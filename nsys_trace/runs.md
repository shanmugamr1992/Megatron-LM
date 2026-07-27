# Baseline Nsight Systems traces — Qwen3-30B-A3B, 4×GB200 (oci-hsg)

Clean baselines (no optimization changes) for vLLM and Megatron-Core (mcore),
each captured **under Nsight Systems at BS=256, OSL=1024** on a single OCI
4×GB200 node.

## Files in this folder

| File | What |
|------|------|
| `vllm_baseline_osl1024.nsys-rep`  | vLLM native trace (open in Nsight GUI) |
| `vllm_baseline_osl1024.sqlite`    | vLLM exported SQLite (scripted analysis) |
| `mcore_baseline_osl1024.nsys-rep` | mcore native trace (open in Nsight GUI) |
| `mcore_baseline_osl1024.sqlite`   | mcore exported SQLite (scripted analysis) |
| `mcore_s6_tuned_osl128.nsys-rep`  | mcore **after** the session-6 wins (see below) |
| `mcore_s6_tuned_osl128.sqlite`    | same, exported SQLite |
| `mcore_host_s6_osl128.nsys-rep`   | mcore session-6 config with **host visibility** (osrt + CPU sampling) |
| `mcore_host_s6_osl128.sqlite`     | same, exported SQLite (5.9 GB) |
| `mcore-final-jul27.nsys-rep`      | mcore **final optimized config**, all ten gates (see below) |
| `mcore-final-jul27.sqlite`        | same, exported SQLite (345 MB) |

> The `.nsys-rep` / `.sqlite` binaries are large and are git-ignored via
> `nsys_trace/.gitignore`; only this `runs.md` is tracked. Each fetched trace
> also has a `<name>.source.txt` recording the session and remote path it
> came from.

Fetch a trace from a cog session's lustre run dir with:

```bash
dev/moe_fused/fetch_profile.sh <session> <prof-dir|latest> <dest-basename>
# e.g. dev/moe_fused/fetch_profile.sh qwen-comm s6-all-1785028240 mcore_s6_tuned_osl128
```

## `mcore-final-jul27` — final optimized config, all ten gates

The end state of the July optimization campaign: every accepted lever enabled,
at the fixed profile protocol (BS=256, **OSL=128**), so it is directly
comparable to `mcore_s6_tuned_osl128` but *not* to the OSL1024 baselines.

- Session `qwen-updreq`, Slurm job `5616264`, run dir
  `sessions/qwen-updreq/prof/g2-on-1785107105`. Captured by
  `dev/moe_fused/profile_insession.sh`.
- All ten gates on — the seven of `mcore_s6_tuned_osl128` plus
  `MCORE_INFER_INCR_ATTN_STATE=1 MCORE_INFER_VEC_UPDATE_REQS=1
  MCORE_INFER_FAST_POST_PROCESS=1`.
- This is the **gate-ON arm of the QWEN-025 A/B pair**; its gate-OFF partner is
  `sessions/qwen-updreq/prof/g2-off-1785106875`, which is the trace to diff
  against for the last lever's effect.
- Corresponding un-profiled throughput at OSL1024 is **26,430.7 tok/s**
  (77.75% of the vLLM baseline). The 11,159 tok/s in this run's
  `profile_bench.log` is the OSL128-under-nsys number and is not comparable.
- 1,862,685 kernels across all 4 ranks, 124.6 s span, `pragma quick_check` ok.
- **GPU-only trace**: `CUPTI_ACTIVITY_KIND_RUNTIME` and NVTX are present,
  `OSRT_API` and the sampling tables are not. Host gaps show up as idle time
  here but cannot be attributed to call sites from this file.
- Ledger record: `QWEN-025` in `skills/run-qwen-model/EXPERIMENTS.md`.

> Known open question in this trace: residual G1 is ~499 µs/step while the
> entire *measured* host bookkeeping chain is down to 148 µs/step, so most of
> G1 is no longer the bookkeeping chain and has not been isolated. Attributing
> it needs a host-visibility capture, which failed three times in nsys
> finalization on 2026-07-26 (see the recovery notes at the end of this file).

## `mcore_s6_tuned_osl128` — earlier best config (PROFILE-S6)

Captured at the fixed **profile** protocol (BS=256, **OSL=128**), so it is not
comparable to the OSL1024 baselines above; it is the trace behind the
`PROFILE-S6` record in `skills/run-qwen-model/EXPERIMENTS.md`.

- Session `qwen-comm`, Slurm job `5601961`, run dir
  `sessions/qwen-comm/prof/s6-all-1785028240`.
- All seven optimization gates on: `MCORE_FUSE_FC1_ACT=1
  MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1
  MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1`.
- Decode step 9.097 ms; routing/permute down to 430 µs / 98 kernels (4.7%);
  largest remaining item is **1237 µs/step of host gaps between graph replays**
  (21.0% idle).
- **GPU-only trace.** It has `CUPTI_ACTIVITY_KIND_RUNTIME` but no OSRT or
  sampling tables, so the host gaps above cannot be attributed from this file —
  that is what the separate host-visibility capture is for.

## `mcore_host_s6_osl128` — host-visibility capture (HOSTGAP-S6)

Same code, same seven gates, same BS256/OSL128 protocol as
`mcore_s6_tuned_osl128`, but traced with OS-runtime and CPU sampling so the
inter-replay host gaps can be attributed to call sites. This is the trace behind
the `HOSTGAP-S6` record in `skills/run-qwen-model/EXPERIMENTS.md`.

- Session `qwen-host`, Slurm job `5607600`, node `nvl72166-T15`, exec
  `d920241ea00a41618c63a75fefcb9ea2`, run dir
  `sessions/qwen-host/prof/hosts6-1785048883`.
- Produced by `dev/moe_fused/profile_host_insession.sh`; nsys
  2026.3.1.117-263137992252v0.
- Whole-benchmark result under this trace: 6,631.7 tok/s, TPOT 38.60 ms
  (vs 10,454.6 tok/s for the GPU-only `mcore_s6_tuned_osl128` capture). **The
  overhead is not in the decode loop** — the steady-state decode step is 9.330
  ms here vs 9.134 ms in the GPU-only trace (2.1%); the rest falls on model
  load, chunked prefill and CUDA-graph capture.

### nsys flags

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --backtrace=fp \
  --samples-per-backtrace=1 \
  --cpuctxsw=process-tree \
  --python-sampling=true \
  --python-sampling-frequency=1000 \
  --osrt-threshold=1000 \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "$RUN_DIR/mcore_host_profile" \
  $PYBIN -m torch.distributed.run --nproc-per-node 4 ... \
  -m examples.inference.launch_inference_server ...    # server args identical
                                                       # to profile_insession.sh
```

The script preflights this exact flag set against a trivial `python -c` target
and degrades (to `--trace=cuda,nvtx,osrt --sample=process-tree --backtrace=fp
--python-sampling=true`, then to `--trace=cuda,nvtx,osrt --sample=process-tree`)
rather than aborting, so an unsupported flag cannot cost the allocation. The
full flag set was accepted here — see `flagcheck.log` in the run dir.

### Host tables: present vs absent

67 tables total. Present, with row counts:

| Table | Rows |
|---|---:|
| `SAMPLING_CALLCHAINS` | 101,181,751 |
| `SCHED_EVENTS` | 12,437,596 |
| `OSRT_CALLCHAINS` | 5,339,842 |
| `NVTX_EVENTS` | 5,293,465 |
| `OSRT_API` | 5,180,565 |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | 2,663,209 |
| `CUPTI_ACTIVITY_KIND_KERNEL` | 1,918,949 |
| `COMPOSITE_EVENTS` | 1,450,481 |
| `StringIds` | 206,086 |
| `CUPTI_ACTIVITY_KIND_MEMCPY` | 69,781 |
| `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION` | 38,548 |
| `ThreadNames` | 6,551 |

**Absent**: `PYTHON_SAMPLING_CALLCHAINS`, `PYTHON_SAMPLING_RAW`,
`PYTHON_SAMPLING_STRING` (despite `--python-sampling=true` being accepted),
`CUDA_GRAPH_EVENTS`, `CUPTI_ACTIVITY_KIND_GRAPH_TRACE`, `GENERIC_EVENTS`.
Kernels are still recorded individually (`--cuda-graph-trace=node`), and each
carries a `graphId`. So host attribution must go through NVTX ranges plus
native `SAMPLING_CALLCHAINS`/`COMPOSITE_EVENTS` backtraces, not Python-level
frame names. In practice the engine's NVTX ranges (`forward_pass`, `sampling`,
`transfer_samples_to_cpu`, `active_request_mask`, `update_requests`,
`initialize_attention_state`, `transfer_bookkeeping_to_gpu`) make this
sufficient.

### Provenance: QdstrmImporter recovery

nsys finalization deadlocked after the target exited (CLI and agent both parked
in `futex_wait`), so the in-band `.nsys-rep` generation and sqlite export never
ran. `profile_host_insession.sh` bounds that wait (`NSYS_STOP_TIMEOUT`, default
900 s) and then recovers offline: it copies the intermediate
`/tmp/nsys-root/nsys-report-*.qdstrm` next to the target path and runs
`QdstrmImporter --input-file mcore_host_profile.qdstrm` with the importer's own
lib dir on `LD_LIBRARY_PATH` (its `$ORIGIN` rpath is unreliable in this
container). That is why a `.qdstrm` sits beside the report in the run dir.
The recovered file is **not** truncated — verified independently:
`pragma quick_check` returns `ok`, all four ranks are present with ~479 k
kernels each spanning 75.0–293.0 s, four engine PIDs carry ~665 k CUDA-runtime
events each, and OSRT/sampling cover 0–300 s.

### Reading this trace: the decode window is NOT the densest region

The densest kernel region (seconds ~196–241, a clean-looking ~146 ms loop) is
**CUDA-graph capture**, not decode: 25 distinct capture streams, no BS256
LM-head GEMM, zero `index_elementwise_kernel`, and
`_multimem_all_gatherv_3tensor_kernel` averaging 1807 µs because the
symmetric-memory barrier absorbs inter-rank skew while other ranks capture.

The real steady-state decode loops are short bursts at the very end:

| Phase | Window | Steps | Period |
|---|---|---:|---:|
| BS8/OSL32 warmup | 287.514–287.838 s | 44 | 7.53 ms |
| **BS256/OSL128 benchmark** | **291.667–292.982 s** | **146** | **9.064 ms** |

Use `--window 291.75,292.95` and force the anchor to
`nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT` (the once-per-step LM-head
GEMM). Forcing it matters: auto-detection picks `vectorized_elementwise_kernel`
on devices 0 and 2, which straddles the largest gap and silently drops it.
Analyse with `dev/moe_fused/analyze_hostgaps.py`:

```bash
python3 dev/moe_fused/analyze_hostgaps.py gaps  <sqlite> --device 3 \
    --window 291.75,292.95 --anchor nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT
python3 dev/moe_fused/analyze_hostgaps.py attrib <sqlite> --device 3 \
    --window 291.75,292.95 --anchor nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT \
    --prev index_elementwise_kernel --next vectorized_elementwise_kernel --min-us 200
```

The file is 5.9 GB; querying it over ssh on the login node is usually faster
than transferring it. Open read-only
(`sqlite3.connect('file:'+path+'?mode=ro', uri=True)`).

## Results (throughput measured under nsys, BS256 / OSL1024, gsm8k)

| Backend | Throughput | TPOT | Job | Run dir |
|---------|-----------|------|-----|---------|
| vLLM (TP1/DP4+EP) | **32,469.6 tok/s** | 7.88 ms/tok | `5567649` | `runs/vllm-qwen30b-nsys-20260723-130924` |
| mcore (TP1/EP4)   | **19,340.2 tok/s** | 13.24 ms/tok | `5567548` | `runs/qwen-30b-nsys-20260723-125942` |

(These numbers are measured *while profiling under nsys*, so they are lower than
un-profiled throughput; they are self-consistent between the two backends since
both ran under the same nsys settings.)

## Code state — clean, no optimization changes

- **vLLM**: stock `vllm serve` (separate codebase; unaffected by any mcore change).
- **mcore**: commit `808c47535` (`feature/enable-inference-optimized-qwen-moe`) —
  the clean pre-optimization baseline. **None** of the QWEN-001…010 optimization
  changes (FC1+SwiGLU fusion, histogram count, async-serial, dispatcher/backend
  sweeps, router/permute fusion, …) are present. Verified: the synced workspace
  has **no** `FUSE_SWIGLU` / `fuse_fc1_activation`.

  > Note on "main": plain upstream `main` does **not** contain the mcore
  > `inference_optimized` Qwen MoE inference path at all — that path is
  > introduced by `808c47535`, which is exactly the fork point all optimization
  > work branches from. So the clean, *runnable* baseline for this workload is
  > `808c47535`, used here.

Shared config (both):
- Cluster `oci-hsg`, one 4×GB200 node; Qwen3-30B-A3B, BF16.
- BS=256 gsm8k requests, OSL=1024, single profiled iteration (BS8 warmup first).
- nsys: `--trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=node`.
- Checkpoints: `.../agents-space/checkpoints/qwen3-30b-a3b-{mcore,hf}`
  (`.../agents-space` = `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space`).

---

## How I ran vLLM

Submitted from the repo root (creates + submits an sbatch on `oci-hsg`):

```bash
source ~/.cog/setup.env.oci-hsg
PROFILE_BS=256 PROFILE_OSL=1024 bash skills/run-qwen-model/profile_qwen_vllm.sh
# -> vLLM nsys job 5567649, run dir runs/vllm-qwen30b-nsys-20260723-130924
```

Inside the vLLM container the sbatch runs the profiler + server, warmup, the
profiled OSL1024 benchmark, and the sqlite export:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
VENV=/opt/ray_venvs/nemo_rl.experience.sync_rollout_actor.SyncRolloutActor
PY=$VENV/bin/python
VLLM=$VENV/bin/vllm
HF_CKPT=$SCRATCH/checkpoints/qwen3-30b-a3b-hf
PROF_BASE=$EXP/vllm_profile

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "$PROF_BASE" \
  $VLLM serve "$HF_CKPT" --served-model-name qwen \
    --tensor-parallel-size 1 --data-parallel-size 4 --enable-expert-parallel \
    --max-model-len 4096 --max-num-seqs 512 \
    --gpu-memory-utilization 0.9 --trust-remote-code \
    --port 5000 --host 0.0.0.0 &

# after server is READY: BS8 warmup, then the profiled OSL1024 run
$PY -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 --num-iters 1 --num-warmup-iters 0

$PY -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 256 --dataset gsm8k --num-output-tokens 1024 --num-iters 1 --num-warmup-iters 0

# stop nsys, then export sqlite
nsys export --type sqlite --force-overwrite=true \
  --output "$PROF_BASE.sqlite" "$PROF_BASE.nsys-rep"
```

---

## How I ran mcore (clean baseline)

To guarantee "main without any of my changes", I checked out the fork-point
commit into a clean worktree and pointed cog at it (so only clean code is
synced — none of my branches, and none of the untracked repo junk):

```bash
# clean fork-point checkout used as the cog repo
git worktree add --detach ~/mlm-clean-baseline 808c47535   # feature/enable-inference-optimized-qwen-moe

source ~/.cog/setup.env.oci-hsg
export COG_MEGATRON_REPO=~/mlm-clean-baseline
PROFILE_BS=256 PROFILE_OSL=1024 bash skills/run-qwen-model/profile_qwen_mcore.sh
# -> cog submit job 5567548, run dir runs/qwen-30b-nsys-20260723-125942
```

`cog submit` syncs `$COG_MEGATRON_REPO`, then runs (under `srun`, inside the
prepared container) the profiler + EP4/TP1 inference server, warmup, the
profiled OSL1024 benchmark, and the sqlite export:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
CKPT=$SCRATCH/checkpoints/qwen3-30b-a3b-mcore
TOKENIZER=$SCRATCH/checkpoints/qwen3-30b-a3b-hf
PROF_BASE=$RUN_DIR/mcore_profile

QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 \
  --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention \
  --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 \
  --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 \
  --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu \
  --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope \
  --rotary-base 1000000 --qk-layernorm --disable-bias-linear \
  --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion \
  --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 \
  --expert-tensor-parallel-size 1 --inference-moe-token-dispatcher-type nvls \
  --inference-grouped-gemm-backend vllm"

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --force-overwrite=true \
  -o "$PROF_BASE" \
  python -m torch.distributed.run --nproc-per-node 4 --log-dir "$RUN_DIR/torchrun_logs" \
  -m examples.inference.launch_inference_server \
  --load "$CKPT" \
  --dist-ckpt-strictness log_unexpected \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER" \
  --no-use-tokenizer-model-from-checkpoint-args \
  --micro-batch-size 1 --bf16 --te-rng-tracker --inference-rng-tracker \
  --transformer-impl inference_optimized \
  --inference-dynamic-batching \
  --inference-dynamic-batching-unified-memory-level 0 \
  --use-flashinfer-fused-rope \
  --inference-dynamic-batching-max-tokens 4096 \
  --enable-chunked-prefill \
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \
  --inference-dynamic-batching-buffer-size-gb 40 \
  --inference-dynamic-batching-max-requests 256 \
  --inference-dynamic-batching-num-cuda-graphs -1 \
  --cuda-graph-impl local \
  --cuda-graph-scope full_iteration_inference \
  --inference-use-synchronous-zmq-collectives \
  --inference-logging-step-interval 100 \
  --port 5000 \
  $QWEN_MODEL_ARGS &

# after server is READY: BS8 warmup, then the profiled OSL1024 run
python -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 --num-iters 1 --num-warmup-iters 0

python -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 256 --dataset gsm8k --num-output-tokens 1024 --num-iters 1 --num-warmup-iters 0

# stop nsys, then export sqlite
nsys export --type sqlite --force-overwrite=true \
  --output "$PROF_BASE.sqlite" "$PROF_BASE.nsys-rep"
```
