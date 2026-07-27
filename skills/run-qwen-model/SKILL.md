---
name: run-qwen-model
description: Run Qwen3-30B-A3B inference with Megatron-Core or vLLM on one OCI 4×GB200 node at batch size 256, and capture matching Nsight Systems profiles. Use for Qwen 30B mcore runs, vLLM runs, baseline benchmarks, or nsys profile capture. Performance optimization belongs to the qwen-model-optimizer agent.
---

# Run Qwen3-30B-A3B on OCI GB200

This skill only runs the fixed comparison workload. Use the
`qwen-model-optimizer` subagent for performance investigation or code changes.

## Fixed comparison

| Setting | Megatron-Core | vLLM |
|---|---|---|
| Cluster | OCI `oci-hsg`, one 4×GB200 node | same |
| Model | Qwen3-30B-A3B, BF16 | same HF weights |
| Batch | 256 gsm8k requests | same |
| Throughput OSL | 1024 | 1024 |
| Profile OSL | 128 (bounded trace) | 128 |
| Parallelism | **TP=1, PP=1, EP=4** | **TP=1, DP=4, expert parallel enabled** |
| Warmup / timed | 2 / 5 | 2 / 5 |

Checkpoint paths:

```bash
export QWEN30B_CKPT=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-mcore
export QWEN30B_TOKENIZER=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf
export QWEN30B_HF="$QWEN30B_TOKENIZER"
```

Before running:

```bash
source ~/.cog/setup.env.oci-hsg
export COG_MEGATRON_REPO=/path/to/Megatron-LM
cog prepare-image --repo "$COG_MEGATRON_REPO" --cluster-name "$COG_CLUSTER_NAME"
cog ensure-env --repo "$COG_MEGATRON_REPO" \
  --cluster-name "$COG_CLUSTER_NAME" \
  --run-name qwen30b-env --gpus 4 --time 00:30:00 \
  --partition "$COG_BATCH_PARTITION"
```

Never download checkpoints. If either path is missing, ask the user.

## 1. Megatron-Core inference

Runs `examples.inference.launch_inference_server` with
`transformer_impl=inference_optimized`, full-iteration CUDA graphs, and the
fixed EP4/TP1 layout.

```bash
EXPERIMENT_ID=MCORE-BASELINE \
EXPERIMENT_HYPOTHESIS="Fresh EP4 mcore baseline" \
QWEN30B_TP=1 QWEN30B_EP=4 QWEN30B_ETP=1 \
BENCH_SIZES_OVERRIDE=256 BENCH_OUTPUT_TOKENS=1024 \
NUM_WARMUP_ITERS=2 NUM_TIMED_ITERS=5 \
bash skills/run-qwen-model/run_qwen_inference.sh \
  qwen3-30b-a3b --checkpoint "$QWEN30B_CKPT"
```

Do not add optimization flags to a baseline run.

## 2. vLLM inference

Runs `vllm serve` with TP1/DP4 and `--enable-expert-parallel`.

```bash
EXPERIMENT_ID=VLLM-BASELINE \
EXPERIMENT_HYPOTHESIS="Fresh vLLM DP4+EP baseline" \
BENCH_BS=256 BENCH_OUTPUT_TOKENS=1024 \
NUM_WARMUP_ITERS=2 NUM_TIMED_ITERS=5 \
bash skills/run-qwen-model/run_qwen_vllm.sh
```

## Nsight Systems profiles

Profiles use BS256 and OSL128 to bound trace size while preserving the same
parallel layouts. Both scripts export `.nsys-rep` and `.sqlite`.

```bash
# mcore EP4/TP1
PROFILE_BS=256 PROFILE_OSL=128 \
QWEN30B_CKPT="$QWEN30B_CKPT" \
bash skills/run-qwen-model/profile_qwen_mcore.sh

# vLLM DP4+EP
PROFILE_BS=256 PROFILE_OSL=128 \
bash skills/run-qwen-model/profile_qwen_vllm.sh
```

Run the profiles sequentially to avoid node contention. Record the run
directory, Slurm job, throughput, latency, TPOT, and trace paths in
`EXPERIMENTS.md`.

For analysis, use:

- `skills/perf_skills/nsight-systems/SKILL.md` for capture and standard reports.
- `skills/perf_skills/nsight-system-analysis/SKILL.md` for mcore/vLLM A/B
  interval-union analysis.

## Records

`EXPERIMENTS.md` is the sole performance ledger. The first two records must be
the fresh vLLM and mcore nsys baselines. Append every later attempt, including
failures and regressions; never rewrite prior records.
