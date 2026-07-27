#!/usr/bin/env python3
"""Host-side microbenchmark for `DynamicInferenceContext.initialize_attention_state`.

`initialize_attention_state` is pure CPU work sitting on the per-step decode
critical path (HOSTGAP-S6 attributed ~311 us/step to it at BS256). This harness
builds a context with the Qwen3-30B-A3B inference geometry, drives a synthetic
steady-state decode loop, and reports:

  1. the coarse per-phase breakdown from MCORE_INFER_ATTN_PROF, and
  2. a statement-level breakdown of the phases that dominate.

No model weights are loaded, so a run takes seconds rather than a server start.

Usage:
    MCORE_INFER_ATTN_PROF=1 python dev/moe_fused/harness_attnstate.py --steps 400
"""

import argparse
import os
import time

os.environ.setdefault("MCORE_INFER_ATTN_PROF", "1")
os.environ.setdefault("MCORE_INFER_ATTN_PROF_EVERY", "1000000")  # report manually

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts import dynamic_context as dc
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.inference_request import DynamicInferenceRequest
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer import TransformerConfig

PERF = time.perf_counter_ns


def build_context(args):
    model_config = TransformerConfig(
        num_layers=48,
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_attention_heads=32,
        num_query_groups=4,
        kv_channels=128,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    inference_config = InferenceConfig(
        max_sequence_length=args.max_seq_len,
        max_requests=args.batch_size,
        max_tokens=args.max_tokens,
        buffer_size_gb=args.buffer_gb,
        block_size_tokens=args.block_size,
        num_cuda_graphs=args.num_cuda_graphs,
        use_cuda_graphs_for_non_decode_steps=True,
        enable_chunked_prefill=True,
        unified_memory_level=0,
    )
    return DynamicInferenceContext(model_config=model_config, inference_config=inference_config)


def add_requests(context, batch_size, prompt_len, tokens_to_generate):
    # Prompts must be short: the whole batch is admitted in one prefill step
    # here, whereas the engine chunks prefill across many steps. Prompt length
    # does not affect the host cost being measured -- the block table is always
    # `max_kv_block_count` columns wide -- only the one-off prefill step.
    for i in range(batch_size):
        length = prompt_len + (i % 5)
        context.add_request(
            DynamicInferenceRequest(
                request_id=i,
                prompt_tokens=torch.randint(0, 1000, (length,), dtype=torch.long, device='cpu'),
                sampling_params=SamplingParams(num_tokens_to_generate=tokens_to_generate),
            )
        )


def statement_breakdown(context, reps):
    """Time the individual statements of the phases that the coarse pass flags.

    Runs against the live context in its current (steady-state decode) state, so
    every shape and dtype matches what the engine sees.
    """
    out = {}

    def bench(name, fn):
        fn()  # warm
        t0 = PERF()
        for _ in range(reps):
            fn()
        out[name] = (PERF() - t0) / reps / 1000.0

    active_slice = slice(context.paused_request_count, context.total_request_count)
    query_lengths_view = context.request_query_lengths[active_slice]
    kv_offsets_view = context.request_kv_length_offsets[active_slice]
    block_ids_view = context.request_to_kv_block_ids[active_slice]
    real_bs = query_lengths_view.numel()
    padded_bs = context.padded_batch_dimensions.req_count

    bench(
        "mha.query_lengths_copy",
        lambda: context._cpu_mha_query_lengths.__setitem__(
            slice(None, real_bs), query_lengths_view[:real_bs]
        ),
    )
    bench(
        "mha.cu_query_cumsum",
        lambda: context._cpu_mha_cu_query_seq_lengths.__setitem__(
            slice(1, real_bs + 1), torch.cumsum(query_lengths_view[:real_bs], dim=0)
        ),
    )
    bench(
        "mha.kv_seq_lengths_add",
        lambda: context._cpu_mha_kv_seq_lengths.__setitem__(
            slice(None, real_bs), kv_offsets_view[:real_bs] + query_lengths_view[:real_bs]
        ),
    )
    bench(
        "mha.cu_kv_cumsum",
        lambda: context._cpu_mha_cu_kv_seq_lengths.__setitem__(
            slice(1, real_bs + 1), torch.cumsum(context._cpu_mha_kv_seq_lengths[:real_bs], dim=0)
        ),
    )
    bench(
        "mha.block_table_copy",
        lambda: context._cpu_mha_block_table.__setitem__(
            slice(None, real_bs), block_ids_view[:real_bs]
        ),
    )
    bench("slices.build_active_slices", lambda: context.build_active_slices(padded_bs))
    bench("slices.pad_active_slices", context.pad_active_slices)

    # Split the two `slices` helpers into their individual statements.
    padded_slice = slice(context.paused_request_count, context.paused_request_count + padded_bs)
    for label in context.request_metadata:
        bench(
            f"slices.metadata_copy[{label}]",
            lambda label=label: context.active_request_metadata[label][:padded_bs].copy_(
                context.request_metadata[label][padded_slice], non_blocking=True
            ),
        )
    bench(
        "slices.last_token_idx_cumsum",
        lambda: torch.cumsum(
            context.request_query_lengths[padded_slice],
            dim=0,
            out=context.active_request_last_token_idxs[:padded_bs],
        ),
    )
    bench(
        "slices.last_token_idx_sub",
        lambda: context.active_request_last_token_idxs[:padded_bs].sub_(1),
    )
    n_dec = context.num_decode_requests * (context.num_speculative_tokens + 1)
    bench(
        "slices.logit_idxs_copy_gpu",
        lambda: context.active_logit_idxs[:n_dec].copy_(context._decode_logit_idxs[:n_dec]),
    )
    bench("slices.logit_idxs_zero_gpu", lambda: context.active_logit_idxs[n_dec:].zero_())
    empty = slice(context.active_token_count, context.padded_active_token_count)
    bench(
        "tokenpad.empty_slice_write",
        lambda: context.token_to_block_idx.__setitem__(
            empty, context.kv_block_allocator.dummy_block_idx
        ),
    )
    bench(
        "graph.match_graph_config",
        lambda: dc.CUDAGraphBatchDimensionBuilder.match_graph_config(
            context.batch_dimensions,
            context.cuda_graph_batch_dimensions_list,
            strict=context.is_hybrid_model,
            ep_group=context.expert_model_parallel_group,
            match_ep_token_counts=False,
            ep_zmq_communicator=context._ep_zmq_communicator,
        ),
    )
    bench("xfer.transfer_bookkeeping_to_gpu", context.transfer_bookkeeping_to_gpu)
    # `update_requests` clears `active_attn_metadata` at the end of every step.
    mha = context.graph_attn_metadata["mha_metadata"]
    bench(
        "mha.set_state_data",
        lambda: mha.set_state_data(
            padded_active_request_count=padded_bs, max_seqlen_q=1, max_seqlen_k=mha.max_seqlen
        ),
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=4)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--buffer-gb", type=int, default=40)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-cuda-graphs", type=int, default=-1)
    p.add_argument("--stmt-reps", type=int, default=2000)
    args = p.parse_args()

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0, init_method="tcp://127.0.0.1:29531"
        )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1
    )

    context = build_context(args)
    print(
        f"context: max_requests={context.max_requests} max_tokens={context.max_tokens} "
        f"block_size={context.block_size_tokens} max_kv_block_count={context.max_kv_block_count} "
        f"graph_dims={len(context.cuda_graph_batch_dimensions_list)}",
        flush=True,
    )

    add_requests(context, args.batch_size, args.prompt_len, args.steps + 16)

    # One prefill step, then a steady-state decode loop.
    context.initialize_attention_state()
    context.transfer_bookkeeping_to_gpu()
    n_active = context.total_request_count - context.paused_request_count
    context.update_requests(
        active_requests_mask=torch.ones(n_active, dtype=torch.int32, device='cpu'),
        new_tokens=torch.randint(0, 1000, (n_active,), dtype=torch.long, device='cpu'),
    )

    # Warm the decode path, then reset the profiling counters.
    warmup = 20
    for _ in range(warmup):
        context.initialize_attention_state()
        context.transfer_bookkeeping_to_gpu()
        n_active = context.total_request_count - context.paused_request_count
        context.update_requests(
            active_requests_mask=torch.ones(n_active, dtype=torch.int32, device='cpu'),
            new_tokens=torch.randint(0, 1000, (n_active,), dtype=torch.long, device='cpu'),
        )
    torch.cuda.synchronize()
    for i in range(len(dc._attn_prof_ns)):
        dc._attn_prof_ns[i] = 0
    dc._attn_prof_calls[0] = 0

    step_ns = []
    upd_ns = []
    for _ in range(args.steps):
        t0 = PERF()
        context.initialize_attention_state()
        t1 = PERF()
        context.transfer_bookkeeping_to_gpu()
        t2 = PERF()
        n_active = context.total_request_count - context.paused_request_count
        if n_active == 0:
            break
        context.update_requests(
            active_requests_mask=torch.ones(n_active, dtype=torch.int32, device='cpu'),
            new_tokens=torch.randint(0, 1000, (n_active,), dtype=torch.long, device='cpu'),
        )
        step_ns.append((t1 - t0, t2 - t1))
        upd_ns.append(PERF() - t2)

    torch.cuda.synchronize()
    n = len(step_ns)
    ias = sum(a for a, _ in step_ns) / n / 1000.0
    xfer_outer = sum(b for _, b in step_ns) / n / 1000.0
    print(f"\n=== steady-state decode, {n} steps, active={n_active} ===", flush=True)
    print(f"initialize_attention_state : {ias:8.1f} us/call", flush=True)
    print(f"transfer_bookkeeping (2nd) : {xfer_outer:8.1f} us/call", flush=True)
    print(f"update_requests            : {sum(upd_ns) / n / 1000.0:8.1f} us/call", flush=True)
    print("\n--- coarse phase breakdown (MCORE_INFER_ATTN_PROF) ---", flush=True)
    dc._attn_prof_report()

    print("\n--- statement-level breakdown (us/call) ---", flush=True)
    for name, us in sorted(
        statement_breakdown(context, args.stmt_reps).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {name:38s} {us:8.2f}", flush=True)


if __name__ == "__main__":
    main()
