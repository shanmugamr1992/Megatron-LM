#!/usr/bin/env python3
"""Host-side microbenchmark for the post-sampling decode bookkeeping chain.

HOSTGAP-S6 attributed ~437 us/step of gap G1 to the chain that runs after the
sampled tokens come back from the device:

    active_request_mask  (text_generation_controller.py:1756-1805)   78.0 us
    update_requests      (dynamic_context.py:update_requests)       183.6 us
    an unlabelled engine window between them                       ~152.2 us

This harness drives a real `DynamicInferenceContext` at BS256 through a
synthetic steady-state decode loop, plus a stub engine that runs the real
`post_process_requests`, and reports a statement-level CPU breakdown of each
region. No model weights are loaded, so a run takes seconds.

Usage:
    python dev/moe_fused/harness_updreq.py --steps 300
"""

import argparse
import asyncio
import cProfile
import io
import os
import pstats
import time

import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine, RequestEntry
from megatron.core.inference.inference_request import (
    DynamicInferenceRequest,
    DynamicInferenceRequestRecord,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer import TransformerConfig

PERF = time.perf_counter_ns
_VEC = os.environ.get("MCORE_INFER_VEC_UPDATE_REQS", "0") == "1"
_EMPTY_IDXS = torch.empty(0, dtype=torch.long)


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
    reqs = []
    for i in range(batch_size):
        length = prompt_len + (i % 5)
        req = DynamicInferenceRequest(
            request_id=i,
            prompt_tokens=torch.randint(0, 1000, (length,), dtype=torch.long, device='cpu'),
            sampling_params=SamplingParams(
                num_tokens_to_generate=tokens_to_generate, termination_id=-1
            ),
        )
        req.add_event_add_engine()
        context.add_request(req)
        reqs.append(req)
    return reqs


def build_stub_engine(context, requests, loop):
    """A `DynamicInferenceEngine` with only the fields `post_process_requests` reads."""
    eng = object.__new__(DynamicInferenceEngine)
    eng.context = context
    eng.controller = None
    eng.requests = {
        r.request_id: RequestEntry(
            record=DynamicInferenceRequestRecord.from_request(r), future=loop.create_future()
        )
        for r in requests
    }
    eng.waiting_request_ids = []
    eng.failed_request_ids = []
    eng.finished_request_count = 0
    eng.evicted_request_count = 0
    eng.track_generated_token_events = False
    eng.track_paused_request_events = False
    eng.num_speculative_tokens = 0
    eng.stop_word_finished_request_ids = set()
    eng.stop_word_being_finished_ids = set()
    eng._spec_steps = 0
    eng._ppr_cache_epoch = 0
    eng._ppr_fast_key = None
    eng._ppr_fast_requests = []
    eng._ppr_fast_limits = []
    return eng


def mask_block(context, sampled_tokens_cpu, active_request_count, paused_request_count):
    """Replica of text_generation_controller.py:1756-1805 (the `active_request_mask` range)."""
    active_request_slice = slice(paused_request_count, context.total_request_count)
    active_request_ids = context.request_ids[active_request_slice].long()
    active_sequence_lengths = context.get_active_sequence_lengths()
    active_sequence_lengths += 1
    max_sequence_lengths = context.get_max_sequence_lengths()
    active_request_mask = (
        sampled_tokens_cpu
        != context.active_request_metadata["termination_id"][:active_request_count]
    ).byte() & torch.less(active_sequence_lengths, max_sequence_lengths).byte()
    if _VEC and int(active_request_mask.sum()) == active_request_count:
        finished_idxs = _EMPTY_IDXS
        finished_request_ids = context.request_ids[:0]
    else:
        finished_idxs = (
            torch.nonzero(active_request_mask == 0, as_tuple=True)[0] + paused_request_count
        )
        finished_request_ids = context.request_ids[finished_idxs]
    finished_routing_block_ids = {}
    if context.kv_block_allocator.block_routing and finished_idxs.numel() > 0:
        for fidx in finished_idxs.tolist():
            req_id = int(context.request_ids[fidx].item())
            blocks = context.request_to_kv_block_ids[fidx]
            valid = blocks[blocks >= 0].tolist()
            if valid:
                finished_routing_block_ids[req_id] = valid
    new_sample_copy = sampled_tokens_cpu.clone()
    return active_request_ids, active_request_mask, new_sample_copy, finished_request_ids


def bench(out, name, fn, reps):
    fn()
    t0 = PERF()
    for _ in range(reps):
        fn()
    out[name] = (PERF() - t0) / reps / 1000.0


def mask_statement_breakdown(context, sampled_tokens_cpu, reps):
    """Time the individual statements of the `active_request_mask` range."""
    out = {}
    pc = context.paused_request_count
    arc = context.total_request_count - pc
    active_request_slice = slice(pc, context.total_request_count)

    bench(
        out, "mask.request_ids_long", lambda: context.request_ids[active_request_slice].long(), reps
    )
    bench(out, "mask.get_active_seq_lengths", context.get_active_sequence_lengths, reps)
    bench(out, "mask.get_max_seq_lengths", context.get_max_sequence_lengths, reps)
    asl = context.get_active_sequence_lengths()
    msl = context.get_max_sequence_lengths()
    bench(out, "mask.add_one", lambda: asl.add_(1), reps)
    bench(
        out,
        "mask.ne_termination_byte",
        lambda: (
            sampled_tokens_cpu != context.active_request_metadata["termination_id"][:arc]
        ).byte(),
        reps,
    )
    bench(out, "mask.less_byte", lambda: torch.less(asl, msl).byte(), reps)
    m = (sampled_tokens_cpu != context.active_request_metadata["termination_id"][:arc]).byte()
    n = torch.less(asl, msl).byte()
    bench(out, "mask.and", lambda: m & n, reps)
    mask = m & n
    bench(
        out, "mask.nonzero_finished", lambda: torch.nonzero(mask == 0, as_tuple=True)[0] + pc, reps
    )
    fidx = torch.nonzero(mask == 0, as_tuple=True)[0] + pc
    bench(out, "mask.index_finished_ids", lambda: context.request_ids[fidx], reps)
    bench(out, "mask.clone_sample", lambda: sampled_tokens_cpu.clone(), reps)
    return out


def updreq_statement_breakdown(context, reps):
    """Time the individual statements of `update_requests` in its steady-state decode path."""
    out = {}
    pc = context.paused_request_count
    trc = context.total_request_count
    arc = trc - pc
    ngt = 1 + context.num_speculative_tokens
    mask = torch.ones(arc, dtype=torch.uint8, device='cpu')
    new_tokens = torch.randint(0, 1000, (arc,), dtype=torch.long, device='cpu')
    active_slice = slice(pc, trc)

    bench(
        out,
        "upd.prefill_status_maskscatter",
        lambda: context.request_in_prefill_status_tensor.__setitem__(
            context.request_in_prefill_status_tensor == 1, 0
        ),
        reps,
    )
    bench(
        out,
        "upd.chunked_prefill_idx",
        lambda: context.get_index_of_chunked_prefill_request(safe=True),
        reps,
    )
    bench(out, "upd.count_active_item", lambda: (mask == 1).sum().item(), reps)
    bench(out, "upd.count_finished_item", lambda: (mask == 0).sum().item(), reps)
    bench(out, "upd.reset_attention_state", context.reset_attention_state, reps)
    bench(
        out,
        "upd.needs_new_block_byte",
        lambda: (
            context.request_last_kv_block_offset[pc : arc + pc]
            >= context.block_size_tokens - 1 - context.num_speculative_tokens
        ).byte(),
        reps,
    )
    nnb = (
        context.request_last_kv_block_offset[pc : arc + pc]
        >= context.block_size_tokens - 1 - context.num_speculative_tokens
    ).byte()
    bench(out, "upd.needs_new_block_sum_item", lambda: (nnb == 1).sum().item(), reps)
    bench(
        out, "upd.resume_paused_requests", lambda: context.resume_paused_requests(arc, None), reps
    )
    bench(
        out,
        "upd.evict_overflow",
        lambda: context.evict_overflow_paused_requests(arc, new_tokens, None),
        reps,
    )
    bench(
        out,
        "upd.kv_offsets_add",
        lambda: context.request_kv_length_offsets[active_slice].add_(
            context.request_query_lengths[active_slice]
        ),
        reps,
    )
    # Undo the mutation the previous bench just applied `reps + 1` times.
    context.request_kv_length_offsets[active_slice].sub_(
        context.request_query_lengths[active_slice] * (reps + 1)
    )
    bench(
        out,
        "upd.query_lengths_fill",
        lambda: context.request_query_lengths[active_slice].fill_(ngt),
        reps,
    )
    bench(
        out,
        "upd.old_offsets_clone",
        lambda: context.request_last_kv_block_offset[active_slice].clone(),
        reps,
    )
    old_offsets = context.request_last_kv_block_offset[active_slice].clone()
    bench(
        out,
        "upd.last_block_offset_mod",
        lambda: context.request_last_kv_block_offset.__setitem__(
            active_slice, (old_offsets + ngt) % context.block_size_tokens
        ),
        reps,
    )
    next_tokens = new_tokens
    bench(
        out,
        "upd.token_to_input_ids",
        lambda: context.token_to_input_ids.__setitem__(slice(None, arc), next_tokens),
        reps,
    )
    bench(
        out,
        "upd.token_to_pos_ids",
        lambda: context.token_to_pos_ids.__setitem__(
            slice(None, arc),
            context.request_kv_length_offsets[active_slice].repeat_interleave(ngt)
            + torch.arange(ngt, device='cpu').repeat(arc),
        ),
        reps,
    )
    bench(
        out,
        "upd.token_to_request_idx",
        lambda: context.token_to_request_idx.__setitem__(
            slice(None, arc), torch.arange(pc, trc, device='cpu').repeat_interleave(ngt)
        ),
        reps,
    )
    bench(
        out,
        "upd.token_to_position_in_request",
        lambda: context.token_to_position_in_request.__setitem__(
            slice(None, arc), context.token_to_pos_ids[:arc]
        ),
        reps,
    )
    bench(
        out,
        "upd.token_to_local_position",
        lambda: context.token_to_local_position_within_kv_block.__setitem__(
            slice(None, arc), context.token_to_pos_ids[:arc] % context.block_size_tokens
        ),
        reps,
    )
    bench(
        out,
        "upd.raw_positions",
        lambda: old_offsets[:, None] + 1 + torch.arange(ngt, device='cpu')[None, :],
        reps,
    )
    raw_positions = old_offsets[:, None] + 1 + torch.arange(ngt, device='cpu')[None, :]
    bench(out, "upd.crosses_boundary", lambda: raw_positions >= context.block_size_tokens, reps)
    cb = raw_positions >= context.block_size_tokens
    bench(out, "upd.crosses_boundary_any", lambda: cb.any(), reps)
    bench(
        out,
        "upd.token_to_block_idx",
        lambda: context.token_to_block_idx.__setitem__(
            slice(None, arc), context.request_last_kv_block_id[active_slice].repeat_interleave(ngt)
        ),
        reps,
    )
    return out


def post_process_requests_prototype(engine, request_ids, finished_request_ids, sample, cache):
    """Decision-gate prototype for a `post_process_requests` decode fast path.

    Measurement only — nothing here is wired into the engine. It implements the
    common decode case (no speculative tokens, no stop words, no log probs, no
    top-n log probs, no chunked prefill, no evictions) and reports what the loop
    body would cost with the per-request dict lookup, record indexing, scalar
    list wrap, method call and repeated length probes removed. `cache` holds the
    resolved `(request, token_limit)` pairs, rebuilt only when the request-id
    tuple changes.
    """
    request_ids_list = request_ids.tolist()
    key = tuple(request_ids_list)
    if cache.get("key") != key:
        cache["key"] = key
        cache["pairs"] = [
            (
                engine.requests[rid].record.requests[-1],
                engine.requests[rid].record.requests[-1].sampling_params.num_tokens_to_generate,
            )
            for rid in request_ids_list
        ]
    pairs = cache["pairs"]

    active_request_ids = []
    at_limit = None
    for i, token in enumerate(sample.tolist()):
        request, limit = pairs[i]
        generated = request.generated_tokens
        generated.append(token)
        if len(generated) >= limit:
            if at_limit is None:
                at_limit = []
            at_limit.append(i)
        else:
            active_request_ids.append(request_ids_list[i])
    return active_request_ids, at_limit


_CONTEXT_STATE_TENSORS = (
    "request_ids",
    "request_kv_length_offsets",
    "request_query_lengths",
    "request_output_lengths",
    "request_last_kv_block_offset",
    "request_last_kv_block_id",
    "request_kv_block_counts",
    "request_to_kv_block_ids",
    "request_in_prefill_status_tensor",
)
_TOKEN_STATE_TENSORS = (
    "token_to_input_ids",
    "token_to_pos_ids",
    "token_to_request_idx",
    "token_to_position_in_request",
    "token_to_local_position_within_kv_block",
    "token_to_block_idx",
)


def compare_contexts(a, b, step):
    """Assert two contexts hold identical request and token bookkeeping."""
    bad = []
    for name in ("total_request_count", "paused_request_count", "active_token_count"):
        if getattr(a, name) != getattr(b, name):
            bad.append(f"{name}: {getattr(a, name)} != {getattr(b, name)}")
    if bad:
        raise AssertionError(f"step {step}: " + "; ".join(bad))
    for name in _CONTEXT_STATE_TENSORS:
        if not torch.equal(getattr(a, name), getattr(b, name)):
            n = int((getattr(a, name) != getattr(b, name)).sum())
            bad.append(f"{name}: {n} elements differ")
    n_tok = a.active_token_count
    for name in _TOKEN_STATE_TENSORS:
        x, y = getattr(a, name)[:n_tok], getattr(b, name)[:n_tok]
        if not torch.equal(x, y):
            bad.append(f"{name}: {int((x != y).sum())} of {n_tok} differ")
    alloc_a, alloc_b = a.kv_block_allocator, b.kv_block_allocator
    for name in ("total_avail", "active_count", "paused_count"):
        if getattr(alloc_a, name) != getattr(alloc_b, name):
            bad.append(f"allocator.{name}: {getattr(alloc_a, name)} != {getattr(alloc_b, name)}")
    if bad:
        raise AssertionError(f"step {step}: " + "; ".join(bad))


def run_equivalence(args):
    """Drive two identical contexts, gate OFF and gate ON, through the same script.

    The scripted mask sequence deliberately terminates requests mid-batch and
    runs long enough to cross a KV block boundary (block_size_tokens steps), so
    the finish, pause and resume branches are all exercised under both gates.
    """
    from megatron.core.inference.contexts import dynamic_context as dc

    global _VEC

    torch.manual_seed(1234)
    ctx_off = build_context(args)
    torch.manual_seed(1234)
    ctx_on = build_context(args)
    torch.manual_seed(1234)
    add_requests(ctx_off, args.batch_size, args.prompt_len, args.equiv_steps + 64)
    torch.manual_seed(1234)
    add_requests(ctx_on, args.batch_size, args.prompt_len, args.equiv_steps + 64)
    compare_contexts(ctx_off, ctx_on, -1)

    finished = 0
    for step in range(args.equiv_steps):
        n_active = ctx_off.total_request_count - ctx_off.paused_request_count
        if n_active == 0:
            break
        tokens = torch.randint(0, 1000, (n_active,), dtype=torch.long, device='cpu')
        mask = torch.ones(n_active, dtype=torch.uint8, device='cpu')
        # Terminate a couple of requests mid-batch on a fixed schedule.
        if args.finish_every > 0 and step > 0 and step % args.finish_every == 0:
            for pos in (0, n_active // 3, n_active - 1):
                if pos < n_active:
                    mask[pos] = 0
            finished += int((mask == 0).sum())
        for ctx, gate in ((ctx_off, False), (ctx_on, True)):
            dc._VEC_UPDATE_REQS = gate
            _VEC = gate
            ctx.initialize_attention_state()
            ctx.transfer_bookkeeping_to_gpu()
            pc = ctx.paused_request_count
            mask_block(ctx, tokens, ctx.total_request_count - pc, pc)
            ctx.update_requests(mask.clone(), tokens.clone(), None)
        compare_contexts(ctx_off, ctx_on, step)

    dc._VEC_UPDATE_REQS = _VEC = os.environ.get("MCORE_INFER_VEC_UPDATE_REQS", "0") == "1"
    print(
        f"\nEQUIVALENCE OK: {args.equiv_steps} steps, {finished} requests terminated "
        f"mid-batch, block boundary crossed={args.equiv_steps > args.block_size}, "
        f"final active={ctx_off.total_request_count - ctx_off.paused_request_count}",
        flush=True,
    )


def run_ppr_equivalence(args):
    """Drive two identical stub engines through the same token stream, gate OFF and ON.

    Covers the three cases that matter for the fast path: plain decode steps (fast
    path taken), steps where requests finish (fast path must decline and the
    reference must run), and requests that reach `num_tokens_to_generate` and must
    stop accumulating tokens. After every step both engines are compared on the
    full observable result — returned active ids, finished records, and per-request
    generated tokens, length, status and TTFT.
    """
    import asyncio

    from megatron.core.inference.engines import dynamic_engine as de

    loop = asyncio.new_event_loop()
    limit = max(4, args.equiv_steps // 2)

    def make(gate):
        torch.manual_seed(4321)
        ctx = build_context(args)
        torch.manual_seed(4321)
        reqs = add_requests(ctx, args.batch_size, args.prompt_len, limit)
        return build_stub_engine(ctx, reqs, loop), ctx

    eng_off, ctx_off = make(False)
    eng_on, ctx_on = make(True)

    ids = ctx_off.request_ids[: args.batch_size].long().clone()
    alive = list(range(args.batch_size))
    n_finish_steps = 0
    n_fast_declined = 0

    for step in range(args.equiv_steps):
        if not alive:
            break
        cur_ids = torch.tensor(alive, dtype=ids.dtype)
        tokens = torch.randint(0, 1000, (len(alive),), dtype=torch.long, device='cpu')
        # Finish a few requests mid-batch on a fixed schedule, so the fast path has
        # to decline and hand the step to the reference loop.
        fin = []
        if args.finish_every > 0 and step > 0 and step % args.finish_every == 0:
            fin = [alive[0], alive[len(alive) // 3], alive[-1]]
            fin = sorted(set(fin))
            n_finish_steps += 1
        fin_t = torch.tensor(fin, dtype=ids.dtype)

        out = {}
        for tag, eng, gate in (("off", eng_off, False), ("on", eng_on, True)):
            de._FAST_POST_PROCESS = gate
            out[tag] = eng.post_process_requests(
                cur_ids.clone(), fin_t.clone(), None, 0.0, tokens.clone(), None, None
            )
        if fin:
            n_fast_declined += 1

        act_off, rec_off = out["off"]
        act_on, rec_on = out["on"]
        bad = []
        if list(act_off) != list(act_on):
            bad.append(f"active ids differ ({len(act_off)} vs {len(act_on)})")
        if len(rec_off) != len(rec_on):
            bad.append(f"finished record count {len(rec_off)} != {len(rec_on)}")
        if eng_off.finished_request_count != eng_on.finished_request_count:
            bad.append(
                f"finished_request_count {eng_off.finished_request_count} "
                f"!= {eng_on.finished_request_count}"
            )
        if set(eng_off.requests) != set(eng_on.requests):
            bad.append("live request id sets differ")
        for rid in sorted(set(eng_off.requests) & set(eng_on.requests)):
            a = eng_off.requests[rid].record[-1]
            b = eng_on.requests[rid].record[-1]
            if a.generated_tokens != b.generated_tokens:
                bad.append(f"req {rid}: generated_tokens differ")
            if a.generated_length != b.generated_length:
                bad.append(
                    f"req {rid}: generated_length {a.generated_length} != {b.generated_length}"
                )
            if a.status != b.status:
                bad.append(f"req {rid}: status {a.status} != {b.status}")
            if (a.ttft is None) != (b.ttft is None):
                bad.append(f"req {rid}: ttft set on one side only")
        for ra, rb in zip(rec_off, rec_on):
            if ra[-1].generated_tokens != rb[-1].generated_tokens:
                bad.append("finished record tokens differ")
            if ra[-1].generated_length != rb[-1].generated_length:
                bad.append("finished record generated_length differs")
        if bad:
            raise AssertionError(f"ppr step {step}: " + "; ".join(bad))

        alive = list(act_off)

    de._FAST_POST_PROCESS = os.environ.get("MCORE_INFER_FAST_POST_PROCESS", "0") == "1"
    sample_req = next(iter(eng_off.requests.values())).record[-1]
    n_gen = len(sample_req.generated_tokens)
    print(
        f"\nPPR EQUIVALENCE OK: {args.equiv_steps} steps, {n_finish_steps} steps with "
        f"finishers, {n_fast_declined} fast-path declines, {args.batch_size - len(alive)} "
        f"requests finished, token limit {limit} reached={n_gen >= limit} "
        f"(a surviving request holds {n_gen} generated tokens), final active={len(alive)}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=4)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--buffer-gb", type=int, default=40)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--num-cuda-graphs", type=int, default=-1)
    p.add_argument("--stmt-reps", type=int, default=2000)
    p.add_argument("--profile-steps", type=int, default=100)
    p.add_argument("--equiv", action="store_true", help="run the gate ON/OFF equivalence test")
    p.add_argument(
        "--ppr-equiv",
        action="store_true",
        help="run the post_process_requests fast-path equivalence test",
    )
    p.add_argument("--equiv-steps", type=int, default=300)
    p.add_argument("--finish-every", type=int, default=37)
    args = p.parse_args()

    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", world_size=1, rank=0, init_method="tcp://127.0.0.1:29533"
        )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=1
    )
    loop = asyncio.new_event_loop()

    if args.equiv:
        run_equivalence(args)
        return
    if args.ppr_equiv:
        run_ppr_equivalence(args)
        return

    context = build_context(args)
    reqs = add_requests(context, args.batch_size, args.prompt_len, args.steps + 64)
    engine = build_stub_engine(context, reqs, loop)
    print(
        f"context: max_requests={context.max_requests} block_size={context.block_size_tokens} "
        f"active={context.total_request_count - context.paused_request_count}",
        flush=True,
    )

    def one_step(record=None):
        context.initialize_attention_state()
        context.transfer_bookkeeping_to_gpu()
        pc = context.paused_request_count
        n_active = context.total_request_count - pc
        sampled = torch.randint(0, 1000, (n_active,), dtype=torch.long, device='cpu')

        t0 = PERF()
        ids, mask, new_sample_copy, fin_ids = mask_block(context, sampled, n_active, pc)
        t1 = PERF()
        context.update_requests(mask, new_sample_copy, None)
        t2 = PERF()
        engine.post_process_requests(ids, fin_ids, None, 0.0, sampled, None, None)
        t3 = PERF()
        if record is not None:
            record.append((t1 - t0, t2 - t1, t3 - t2))
        return n_active

    for _ in range(20):
        one_step()

    rec = []
    for _ in range(args.steps):
        if one_step(rec) == 0:
            break

    n = len(rec)
    mask_us = sum(a for a, _, _ in rec) / n / 1000.0
    upd_us = sum(b for _, b, _ in rec) / n / 1000.0
    ppr_us = sum(c for _, _, c in rec) / n / 1000.0
    print(f"\n=== steady-state decode, {n} steps, BS={args.batch_size} ===", flush=True)
    print(f"active_request_mask block  : {mask_us:8.1f} us/step", flush=True)
    print(f"update_requests            : {upd_us:8.1f} us/step", flush=True)
    print(f"post_process_requests      : {ppr_us:8.1f} us/step", flush=True)
    print(f"TOTAL measured chain       : {mask_us + upd_us + ppr_us:8.1f} us/step", flush=True)

    print("\n--- active_request_mask statement breakdown (us/call) ---", flush=True)
    pc = context.paused_request_count
    arc = context.total_request_count - pc
    sampled = torch.randint(0, 1000, (arc,), dtype=torch.long, device='cpu')
    context.initialize_attention_state()
    for name, us in sorted(
        mask_statement_breakdown(context, sampled, args.stmt_reps).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {name:38s} {us:8.2f}", flush=True)

    print("\n--- update_requests statement breakdown (us/call) ---", flush=True)
    for name, us in sorted(
        updreq_statement_breakdown(context, args.stmt_reps).items(), key=lambda kv: -kv[1]
    ):
        print(f"  {name:38s} {us:8.2f}", flush=True)

    print("\n--- post_process_requests gate A/B (same process) ---", flush=True)
    from megatron.core.inference.engines import dynamic_engine as de

    ppr_ids = context.request_ids[pc : context.total_request_count].long()
    ppr_fin = torch.empty(0, dtype=context.request_ids.dtype)
    gate_samples = [
        torch.randint(0, 1000, (arc,), dtype=torch.long, device='cpu') for _ in range(200)
    ]
    gate_us = {}
    for tag, gate in (("gate OFF (reference loop)", False), ("gate ON  (decode fast path)", True)):
        de._FAST_POST_PROCESS = gate
        engine.post_process_requests(ppr_ids, ppr_fin, None, 0.0, gate_samples[0], None, None)
        t0 = PERF()
        for s in gate_samples:
            engine.post_process_requests(ppr_ids, ppr_fin, None, 0.0, s, None, None)
        gate_us[tag] = (PERF() - t0) / len(gate_samples) / 1000.0
        print(f"  {tag:28s} {gate_us[tag]:8.2f} us/step", flush=True)
    off_us, on_us = gate_us["gate OFF (reference loop)"], gate_us["gate ON  (decode fast path)"]
    print(f"  {'saved':28s} {off_us - on_us:8.2f} us/step ({off_us / on_us:.2f}x)", flush=True)
    de._FAST_POST_PROCESS = os.environ.get("MCORE_INFER_FAST_POST_PROCESS", "0") == "1"

    print("\n--- post_process_requests prototype A/B ---", flush=True)
    ab_samples = [
        torch.randint(0, 1000, (arc,), dtype=torch.long, device='cpu') for _ in range(200)
    ]
    engine.post_process_requests(ppr_ids, ppr_fin, None, 0.0, ab_samples[0], None, None)
    t0 = PERF()
    for s in ab_samples:
        engine.post_process_requests(ppr_ids, ppr_fin, None, 0.0, s, None, None)
    real_us = (PERF() - t0) / len(ab_samples) / 1000.0
    cache = {}
    fin_set = set()
    post_process_requests_prototype(engine, ppr_ids, fin_set, ab_samples[0], cache)
    t0 = PERF()
    for s in ab_samples:
        post_process_requests_prototype(engine, ppr_ids, fin_set, s, cache)
    proto_us = (PERF() - t0) / len(ab_samples) / 1000.0
    print(f"  post_process_requests (real)      {real_us:8.2f} us/step", flush=True)
    print(f"  decode fast-path prototype        {proto_us:8.2f} us/step", flush=True)
    print(
        f"  reducible                         {real_us - proto_us:8.2f} us/step "
        f"({100.0 * (1 - proto_us / real_us):.0f}%)",
        flush=True,
    )

    print("\n--- post_process_requests cProfile (%d steps) ---" % args.profile_steps, flush=True)
    ids = context.request_ids[pc : context.total_request_count].long()
    fin = torch.empty(0, dtype=context.request_ids.dtype)
    samples = [
        torch.randint(0, 1000, (arc,), dtype=torch.long, device='cpu')
        for _ in range(args.profile_steps)
    ]
    pr = cProfile.Profile()
    pr.enable()
    for s in samples:
        engine.post_process_requests(ids, fin, None, 0.0, s, None, None)
    pr.disable()
    st = pstats.Stats(pr, stream=(buf := io.StringIO()))
    st.sort_stats("tottime").print_stats(18)
    print(buf.getvalue(), flush=True)


if __name__ == "__main__":
    main()
