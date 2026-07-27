# Qwen3-30B-A3B optimization ledger

Goal: make Megatron-Core EP4 inference match or exceed the vLLM DP4+EP
throughput on one OCI 4×GB200 node without correctness regressions.

This ledger starts from scratch. Append every experiment, including failures
and regressions. Never edit an earlier result after it is recorded.

## Fixed protocol

| Setting | Value |
|---|---|
| Cluster | OCI `oci-hsg` |
| Hardware | 1 node, 4×GB200 |
| Model | Qwen3-30B-A3B, BF16 |
| Dataset / batch | gsm8k / 256 |
| Throughput workload | OSL1024, 2 warmups, 5 timed iterations |
| Profile workload | OSL128, one short warmup request, one timed request |
| mcore layout | TP=1, PP=1, EP=4, ETP=1 |
| vLLM layout | TP=1, DP=4, `--enable-expert-parallel` |
| Correctness gate | Fixed temperature-0 coherence prompts plus benchmark success |
| Primary metric | Throughput (output tokens/s) |
| Secondary metrics | Average latency and TPOT |

Do not compare results when hardware, checkpoint, batch size, output length,
parallelism, or warmup/timed counts differ.

## Baselines

| ID | Engine | Throughput | Avg latency | TPOT | Job / run | Nsight trace | Status |
|---|---|---:|---:|---:|---|---|---|
| VLLM-BASELINE | vLLM DP4+EP | 23,606.7 tok/s | 1,368.2 ms | 10.844 ms/tok | 5547673 / `vllm-qwen30b-nsys-20260722-093437` | `vllm_profile.nsys-rep`, `.sqlite` | Pass |
| MCORE-BASELINE | mcore EP4/TP1 | 12,346.1 tok/s | 2,590.1 ms | 20.735 ms/tok | 5553135 / `qwen-30b-nsys-20260722-161020` | `mcore_profile.nsys-rep`, `.sqlite` | Pass |

Fresh profile gap: mcore delivers 52.30% of vLLM throughput and is 47.70%
below the target. vLLM is 1.912× faster on this profile workload.

Baseline order is mandatory:

1. Record `VLLM-BASELINE` with Nsight Systems.
2. Record `MCORE-BASELINE` with Nsight Systems.
3. Compute the absolute and percentage gap.
4. Only then modify Megatron-Core.

## Experiment index

| ID | Date | Hypothesis | Changed files / flags | Throughput | Delta vs mcore baseline | Correctness | Job / run | Conclusion |
|---|---|---|---|---:|---:|---|---|---|
| VLLM-BASELINE | 2026-07-22 | Establish the fixed competitor target | none | 23,606.7 | n/a | Benchmark pass | 5547673 | Target established |
| MCORE-BASELINE | 2026-07-22 | Establish the fixed EP4 starting point | `max_requests=256` | 12,346.1 | baseline | Benchmark pass | 5553135 | Starting point established |
| QWEN-001 | 2026-07-22 | Single-kernel FC1+SwiGLU+FC2+topk-reduce mega-fusion beats the 4-kernel vLLM MoE path | `megatron/core/inference/moe/fused_moe_decode.py` (new), `dev/moe_fused/harness.py` (new) | microbench only | n/a (0.68–0.80× kernel) | Numerics pass (max_abs 2.6e-5, allclose) | session `qwen-moe-kernel` | Rejected — fused kernel 20–50% slower than reference; not integrated |
| QWEN-002 | 2026-07-22 | Fusing SiLU(gate)*up into the FC1 GEMM epilogue (removing bounded_silu_mul + the 2N round-trip) speeds up the decode MoE path without hurting FC2 tiling | `vllm_fused_moe.py` (FUSE_SWIGLU), `experts.py` (`fuse_fc1_activation=True`), `dev/moe_fused/harness_fc1.py`, `dev/moe_fused/run_e2e_insession.sh` (new) | **22,741.9 tok/s** (OSL1024) | **+0.55%** vs same-env fusion-off (22,617.7) | Coherent + numerics (max_abs 3.9e-5) | session `qwen-moe-kernel` in-session A/B | Accepted — MoE path 1.25×, e2e +0.55% throughput / −0.55% TPOT, no regression |
| PROFILE-OSL1024 | 2026-07-22 | Re-profile at the real throughput regime (BS256/OSL1024) to find the true vLLM→mcore gap and the dominant decode bottleneck | none (profiling only); `dev/moe_fused/profile_insession.sh` (new), `dev/moe_fused/vllm_osl1024_tput.sbatch` (new) | vLLM **33,994.5** vs mcore **~22,700** tok/s | mcore = **66.8%** of vLLM (**vLLM 1.50×**) | Both coherent | vLLM job `5555787` (tput) + `5555868` (nsys); mcore in-session `prof256a` | Gap is real & large at OSL1024. mcore decode: GPU 79% busy; MoE grouped-GEMM 41%, attn 22%, MoE routing 12% (49k tiny kernels), **exposed EP comm 11.5%**, norm/elt 10%, GPU idle 21%. vLLM uses TRT-LLM fused MoE (1-kernel routing + cutlass bmm + fused finalize, **0 exposed comm**). Next target: routing-kernel storm + exposed EP comm (pending approval) |
| SESSION2-BASE | 2026-07-23 | Re-establish clean OSL1024 baseline in fresh session `qwen-opt` (fusion on, histogram off) before autonomous optimization campaign | none (config = current best) | **22,398.9 tok/s** | baseline for session 2 (−1.5% vs QWEN-002 run, within variance) | Coherent | session `qwen-opt` run `e2ebase` | Clean reference; 65.9% of vLLM 33,994.5 |
| QWEN-003 | 2026-07-23 | Replacing per-pair `atomic_add` in the MoE local-token count kernel with a `tl.histogram` variant (one atomic/bin/CTA) cuts the routing-kernel cost | `permute.py` (`_count_local_tokens_kernel_histogram`, env `MCORE_MOE_HISTOGRAM_COUNT`), `dev/moe_fused/harness_count.py` (new) | microbench only | **0.96× (wash)** | **EXACT integer match** vs reference | session `qwen-opt` `harness_count.py` | Rejected — count-kernel cost is per-launch fixed overhead, not atomic contention; in-kernel rewrite can't help. Default OFF. Real lever = fewer launches + less host-scheduling idle |
| QWEN-004 | 2026-07-23 | flashinfer sampling backend is faster than torch sampling | `run_e2e_cfg.sh` (`--inference-dynamic-batching-sampling-backend flashinfer`) | server crash | n/a | n/a | session `qwen-opt` `flashinfer` | Rejected — incompatible with `full_iteration_inference` CUDA graph capture: `RuntimeError: Generator not registered with the capturing graph` in `flashinfer.sampling`. torch sampling stays. |
| QWEN-005a | 2026-07-23 | `async-sched-mode=serial` overlaps host resolve with next forward, hiding the ~2.3ms/step (21%) GPU idle | `run_e2e_cfg.sh` (`--inference-dynamic-batching-async-sched-mode serial`) | server crash | n/a | n/a | session `qwen-opt` `serial` | Blocked by explicit guards: `ValueError: Async scheduling does not support expert parallelism` (+ separate MoE guard). Guards env-gated for experiment → QWEN-005b |
| QWEN-005b | 2026-07-23 | Guards were merely conservative; async serial works for MoE+EP if opened | env-gated EP+MoE guards in `dynamic_engine.py` + `text_generation_controller.py` (`MCORE_ALLOW_ASYNC_MOE`) | **hang** | n/a | server init OK but **first decode request hangs** (>2min) | session `qwen-opt` `asyncmoe` (cancelled) | Rejected — the guard encodes a real limitation: async serial + nvls alltoall + EP deadlocks on the first decode step. Patch reverted (tree clean). Would need real engine work to support. |
| QWEN-006 | 2026-07-23 | `nccl` AllGather/ReduceScatter inference dispatcher overlaps/costs less than `nvls` | `run_e2e_cfg.sh` (`--inference-moe-token-dispatcher-type nccl`) | **14,677 tok/s** | **0.66× (much worse)** | Coherent | session `qwen-opt` `nccl` | Rejected — nccl pads to worst-case per-rank token count (fixed-count AllGather), inflating comm volume ~2×. `nvls` variable-count stays the best dispatcher. |
| QWEN-007 | 2026-07-23 | `async-sched-mode=serial` (guards opened) hides the between-graph idle at the real OSL1024 regime | env-gated guards + `run_e2e_cfg.sh` | **22,589 tok/s** | **+0.85% (marginal)** | Runs at BS256 (hang was single-request-only) | session `qwen-opt` `async1024` | Marginal. Proves decode at OSL1024 is **NOT idle-bound** (the earlier "21% idle" was an OSL256 prefill artifact). Not worth shipping (unsupported path + tiny gain). Guards left env-gated default-off. |
| QWEN-008 | 2026-07-23 | `CUDA_DEVICE_MAX_CONNECTIONS=8` (was hardcoded 1) lets comm & compute overlap on separate HW queues, hiding exposed NVLS comm | `run_e2e_cfg.sh` (env override) | **~22,556 tok/s** | **~flat (noise)** | Coherent | session `qwen-opt` `maxconn8` | Reject — overlap is bounded by the full-iteration CUDA-graph structure / data deps, not connection count. No effect. |
| QWEN-010 | 2026-07-23 | `torch` grouped-GEMM backend (`torch.nn.functional.grouped_mm`, cuBLAS) beats the vLLM Triton fused-MoE backend on GB200 | `run_e2e_cfg.sh` (`--inference-grouped-gemm-backend torch`) | **18,434 tok/s** | **0.82× (worse)** | Coherent | session `qwen-opt` `gemmtorch` | Rejected — vLLM Triton backend stays best. (`flashinfer` cutlass backend is blocked for SwiGLU; only torch/vllm allowed.) All three backends now evaluated → vLLM is optimal. |
| QWEN-009 | 2026-07-23 | Built-in `--moe-router-fusion` (TE fused softmax+topk) + `--moe-permute-fusion` cut the routing critical path (~18%) | `run_e2e_cfg.sh` EXTRA_SERVER_ARGS | server crash | n/a | n/a | session `qwen-opt` `routperm`/`routfus` | Rejected — both crash: `AssertionError: hidden_size mismatch: 128 vs 8`. TE fused router emits a dense **128-expert** routing map, but `InferenceTopKRouter` (transformer_impl=inference_optimized) uses a dense **top-8** contract for the vLLM/nvls dispatcher. The built-in fusions are wired to the training MoE path only. A hand-written fused softmax+topk would need to honor the top-8 inference contract. |
| PROFILE-DECODE | 2026-07-23 | Get the TRUE per-step decode bottleneck (prior OSL256 totals were prefill-contaminated) | analysis of archived `mcore_osl256.sqlite`, pure-decode window (t0+220s, big-dispatch-free) | n/a | n/a | n/a | local sqlite | **Corrected model** (decode GPU-time share): MoE grouped-GEMM (`_fused_moe_kernel`) **~40% #1**, routing (count/moe_sum/topk/scatter/softmax/meta) ~18%, exposed comm (dispatch 122k + combine 320k) ~16%, attention ~13%, norm/elt ~10%. Kernels overlap across streams → **wall = per-layer critical path** (attn→router→dispatch→GEMM→combine). vLLM wins via TRT-LLM fused MoE (0 exposed comm, fused routing+finalize). Explains QWEN-002 1.25× kernel → +0.55% e2e. |
| CLEANBASE-S3 | 2026-07-24 | Fresh clean un-profiled OSL1024 baseline (SwiGLU fusion OFF) in session `qwen-fuse` before session-3 fusion campaign | none (`MCORE_FUSE_FC1_ACT=0`) | **22,241.5 tok/s** | baseline (session 3) | Coherent | session `qwen-fuse` `cleanbase` | Clean reference; 65.4% of vLLM un-profiled 33,994.5. avg_latency 11,494 ms |
| QWEN-002-CONFIRM | 2026-07-24 | Re-measure QWEN-002 SwiGLU FC1-epilogue fusion (lever #3) cleanly at OSL1024 | `experts.py` (`MCORE_FUSE_FC1_ACT=1`), `vllm_fused_moe.py` FUSE_SWIGLU | **22,269.5 tok/s** | **+0.13% vs OFF** (wash) | Coherent | session `qwen-fuse` `qwen002on` | Confirms QWEN-002: MoE-kernel 1.25× but e2e wash at OSL1024 — FC1 activation is not the decode wall bottleneck. Kept on (free, exact). |
| CLEANBASE-S4 | 2026-07-25 | Fresh clean un-profiled OSL1024 reference in session `qwen-cutlass2` at the session-3 best config, before the grouped-GEMM/cutlass campaign | none (`MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`, vllm backend) | **22,657.7 tok/s** | baseline (session 4); −0.17% vs QWEN-011 (within drift) | Coherent | session `qwen-cutlass2` `ref0-1785010433` | Clean reference; 66.65% of vLLM un-profiled 33,994.5. avg_latency 11,282 ms, TPOT 11.299 ms/tok |
| QWEN-012 | 2026-07-25 | **Decision gate**: quantify grouped-GEMM headroom before writing any kernel — is the ~40% decode share recoverable inefficiency or irreducible expert-weight traffic? | none (analysis); `dev/moe_fused/harness_roofline.py` (new) | n/a (microbench) | n/a | n/a | session `qwen-cutlass` `c05fd5f7` | **The decode grouped GEMM is memory-bound, not FLOP-bound.** Weight traffic 302 MB/layer/rank vs a *measured* 6.081 TB/s streaming-read ceiling ⇒ 49.66 µs floor; production FC1+FC2 = 72.13 µs = **1.45× off roofline**. Achieved 63–83 TFLOP/s on valid FLOPs (~3% of GB200 BF16 peak). Padding waste is real (74.3% dead rows at BLOCK_M=64) but nearly **free**: cutting it 3.89×→1.49× buys only ~1.2×. **⇒ Stage 4 (hand-written CUTLASS/CuTe grouped GEMM) cannot win**: its entire ceiling is 1.45×, and 1.26× of that is reachable by Triton tile retuning alone (→ QWEN-013). Remaining levers are elsewhere (exposed EP comm ~16%, routing ~18%) |
| QWEN-013 | 2026-07-25 | Tiling, not the GEMM implementation, is the gap QWEN-012 found: FC1 (N=768) and FC2 (N=2048) want *opposite* BLOCK_SIZE_N, which vLLM's single shared config cannot express | `vllm_fused_moe.py` (`_get_decode_tuned_configs`, per-GEMM `config_fc1`/`config_fc2`, env `MCORE_MOE_GEMM_TUNE`), `dev/moe_fused/harness_gemmtune.py` (new) | **23,636.0 tok/s** | **+4.32% vs CLEANBASE-S4** | **Bit-exact** (max_abs 0.0 at 128/256/384/512 tokens) + coherent | session `qwen-cutlass2` `gemmtune-1785010791` | **Accepted** — GEMM GPU time 77.65→61.89 µs (1.255×), whole MoE call 100.64→84.31 µs (1.194×), e2e +4.32% with tight variance (23.55–23.67k). Bit-exact because `BLOCK_SIZE_K` is unchanged, so the fp32 K-reduction order is identical. Default OFF; enable with `MCORE_MOE_GEMM_TUNE=1`. Now **69.53%** of vLLM |
| QWEN-013b | 2026-07-25 | The QWEN-013 tuned tiles regress past the decode point, so the fallback threshold should be 384, not 512 | `vllm_fused_moe.py` (`_get_decode_tuned_configs` gate `M > 512` → `M > 384`) | **23,646.0 tok/s** | **+4.36% vs CLEANBASE-S4** (+0.04% vs QWEN-013, i.e. same) | Bit-exact (unchanged kernel) + coherent | session `qwen-cutlass2` `gemmtune2-1785012743` | **Accepted** — CUDA-graph device time vs default tiles is 1.289×/1.197×/1.120×/**0.952×** at 128/256/384/512 tokens, so 512 was a 5% regression inside the tuned range. Decode runs at 256 so e2e is unchanged (23,646.0 vs 23,636.0, within variance); the guard just removes a latent regression for other batch shapes. **Current best config.** |
| QWEN-014 | 2026-07-25 | The already-wired `--inference-grouped-gemm-backend flashinfer` (`flashinfer.fused_moe.cutlass_fused_moe`) accepts BF16 and beats the vLLM Triton path; the ledger's "blocked for SwiGLU" note (QWEN-010) is unsupported by the code | none shipped; `dev/moe_fused/inspect_flashinfer.py`, `dev/moe_fused/harness_flashinfer.py` (new) | microbench only | **0.921× vs QWEN-013** (1.079× vs default tiles) | Numerics pass only with a weight-layout fix (see record) | session `qwen-cutlass2` exec `7590adcb`/`659d2633` | **Rejected on measurement, but the ledger note was wrong.** The kernel *does* support BF16 gated SwiGLU; mcore's wiring is broken in two independent ways (an `ActivationType` mis-map that hard-fails, and a gate/up ordering mismatch that silently corrupts numerics — both root-caused below). After fixing both in the harness, CUDA-graph device time is **90.81 µs vs 83.63 µs** for the retuned Triton path — 8% slower — so fixing the backend would not win. Consistent with QWEN-012: everything is pinned near the weight-bandwidth floor. |
| QWEN-015 | 2026-07-25 | `flashinfer.fused_moe.trtllm_bf16_routed_moe` — the actual TRT-LLM-Gen kernel vLLM wins with — can be dropped in as a new grouped-GEMM backend | none (blocked before any Megatron change); probe in `dev/moe_fused/harness_flashinfer.py` | not reached | n/a | n/a | session `qwen-cutlass2` exec `7590adcb` | **Blocked on weight layout** — not a contract or EP problem. `use_shuffled_weight=False, weight_layout=MajorK` is rejected outright (`BF16 Moe: weight_layout must be BlockMajorK`), and `BlockMajorK` reads `size(3)` of the weights (`IndexError: Index 3 out of bounds for tensor with 3 dimensions`): it requires **4-D pre-shuffled block-major** weights, not mcore's 3-D `[E, 2*ffn, H]`. Path forward and why it is now low priority are in the record. |
| PROFILE-TUNED | 2026-07-25 | Confirm the decode timing composition after QWEN-013 and re-rank the remaining levers | none (profiling only) | n/a | n/a | n/a | session `qwen-cutlass2` `prof/tuned-1785013555` (`mcore_profile.nsys-rep`, `.sqlite`) | Workflow C on one steady-state decode step (BS256/OSL128, device 3, 302-step window): forward pass **9.933 ms**, GPU-busy 7.884 ms, **idle 2.049 ms (20.6%)**, 1362 kernels/step. Share of GPU time: **MoE expert GEMM 2611 µs (32.9%, 96 kernels, 27.2 µs/kern)** — down from ~40% pre-retune — routing/permute 1251 µs (15.8%, 242 kernels), dense GEMM 1092 µs (13.8%), **exposed EP comm 929 µs (11.7%, 96 kernels, 9.7 µs/kern)**, attention 913 µs (11.5%), elementwise 528 µs (6.7%), norm 462 µs (5.8%). **Corroborates QWEN-012/013 independently**: 2611 µs / 48 layers = 54.4 µs per layer for FC1+FC2, matching the ~57 µs microbench, and leaving only 2611−48×49.66 = **229 µs/step (2.3%)** above the weight-bandwidth floor. The GEMM lever is spent; comm + routing + idle are what remain. |
| QWEN-011 | 2026-07-24 | Fuse the MoE indirection-table build (lever #1): merge `_init_sorted_ids` + `_prefix_sum` + `_fill_expert_block_ids` into one `_prefix_fill_init_kernel`, cutting the 5-kernel routing storm to 3 kernels/layer (−96 launches/step) | `vllm_fused_moe.py` (`_prefix_fill_init_kernel`, `_moe_align_block_size_fused`, env `MCORE_MOE_FUSED_ALIGN`), `dev/moe_fused/harness_align.py` (new) | **22,696.4 tok/s** | **+1.92% vs QWEN-002-CONFIRM** (+2.04% vs clean) | **EXACT** (max_abs_diff 0.0, allclose) + coherent | session `qwen-fuse` `fusedalign`, branch `perf/moe-fused-align` | **Accepted** — bit-exact, MoE microbench 1.029×, e2e +1.9% with tight variance (22.65–22.71k). Reduces routing/permute kernel count (lever #1). Now 66.8% of vLLM. |
| QWEN-016 | 2026-07-25 | **Decision gate**: is the 929 µs/step of exposed NVLS EP comm a worthwhile lever — how much is genuinely exposed, is it latency- or bandwidth-bound, and what is the floor under perfect overlap? | none (analysis); `dev/moe_fused/probe_comm.py`, `analyze_comm.py`, `analyze_comm_skew.py`, `harness_comm.py` (new) | n/a (analysis) | n/a | Microbench gate bit-exact (AGV max abs 0.0, RSV max rel 0.0 over 64 trials) | session `qwen-comm` job 5601961, execs `08e6a692`/`04bdfdd6` | **Redirect off this lever.** 100% exposed (interval union = sum), but **latency-bound**: AGV 6.57 µs = 0.72 launch + **5.08 barrier** + 0.77 transfer; RSV 7.88 = 0.72 + **5.03** + 2.13. Bytes are only 127 µs/step of ~693 µs and RSV transfer is already 82% of the NVLink floor, so batching bytes cannot win. Of 1060 µs/step, 632 µs is intrinsic and 428 µs is inter-rank skew (ranks waiting on the slowest rank's expert GEMM — a routing-balance problem). Recoverable critical path is **6.4–7.0% of the step**; RSV-into-FC2 fusion keeps the barrier (≈1.4%) and 2-chunk pipelining adds a barrier per chunk (≈4.3%, and needs concurrent streams under graph capture — the QWEN-008 blocker). CTA count already optimal at the shipped 128. |
| QWEN-018 | 2026-07-25 | **Decision gate**: break the 1251 µs/step routing/permute category down by kernel name — launches, device µs, dispatch gap, and fixed-vs-work split — and compute a wall-time ceiling per candidate fusion before building anything | none (analysis); `dev/moe_fused/analyze_routing.py` (new) | n/a (analysis) | n/a | n/a | session `qwen-comm` job 5601961 (login-node re-analysis of `qwen-cutlass2:prof/tuned-1785013555`) | **Two of the six routing kernels are pathological, and neither is launch-bound.** Per steady-state decode step (40-step mean, device 3): `_moe_sum` 7.79 µs × 48, **`_count_local_tokens` 7.52 µs × 48**, `gatherTopK` 6.05, `_scatter_token_indices` 2.66, router softmax 1.94, `_prefix_fill_init` 1.33, plus a 0.74 µs `torch.zeros` fill = 1531 µs/step of wall including gaps. Inter-kernel dispatch gap is **0.55 µs**, so a removed launch is worth `duration + 0.55`, and the fixed floor per launch is only 0.72+0.55 = **1.27 µs**. Ceilings: **(1) `_moe_sum`→FC2 epilogue = 400 µs = 4.03% gross**, but needs cross-CTA fp32 atomics into `out` plus a zeroing launch — QWEN-001's exact failure shape — so ~3.3% net and possibly negative; **(2) cooperative-grid merge of count→prefix_fill→scatter = 122 µs = 1.23%** before paying two grid syncs ⇒ **gated out**; **(3) new: fold the count + its zeros-fill into `_prefix_fill_init_kernel` = 449 µs = 4.52% gross**, no grid sync needed and integer-exact ⇒ **chosen**. `_count_local_tokens` is slow because at `BLOCK_SIZE=1024` only **2 of its 152 CTAs get work**; QWEN-003 changed the reduction but not that, which is why it measured a wash |
| CLEANBASE-S6 | 2026-07-25 | Fresh same-session OSL1024 reference at the session-4/5 best config, in session `qwen-comm`, before the routing campaign | none (`MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1`) | **23,264.4 tok/s** | baseline (session 6); −1.61% vs QWEN-013b's 23,646.0 (session drift) | Benchmark pass, 5/5 iters | session `qwen-comm` job 5601961, run `e2e/ref-s6-1785024868` | Clean reference; 68.44% of vLLM 33,994.5. avg_latency 10,975 ms, TPOT 11.004 ms/tok. Synced snapshot `workspaces/megatron_lm/6b9355b187072223` verified to **not** contain the session-6 kernel |
| QWEN-019 | 2026-07-25 | Fold the per-expert token count and the `torch.zeros` fill of its counter buffer into `_prefix_fill_init_kernel`, taking the decode indirection-table build from 4 launches to 2. No grid sync is needed because that kernel already has every CTA redundantly recompute the cumsum in registers | `vllm_fused_moe.py` (`_count_prefix_fill_init_kernel`, `_moe_align_block_size_count_fused`, env `MCORE_MOE_FUSED_COUNT`, `_FUSED_COUNT_MAX_TOKENS`), `dev/moe_fused/harness_countfuse.py` (new) | **23,964.4 tok/s** | **+3.01% vs CLEANBASE-S6** | **Bit-exact** (max_abs 0.0 *and* max_rel 0.0 at 128/256/384/512) + tables integer-identical + coherent | session `qwen-comm` job 5601961, run `e2e/countfuse-1785025539` | **Accepted** — align call 16.38→10.24 µs under CUDA-graph replay (1.600×), whole MoE call 86.02→79.96 µs (1.076×), e2e +3.01% with tight variance (23.57–24.30k). TPOT 11.004→10.682 ms/tok. Default OFF; enable with `MCORE_MOE_FUSED_COUNT=1` (requires `MCORE_MOE_FUSED_ALIGN=1`). New best = **70.49%** of vLLM 33,994.5 |
| QWEN-017 | 2026-07-25 | `symm_mem_sync` spins on a system-scope `atom.cas` (a full uncached RMW per attempt); an `ld.acquire.sys` poll + clearing store should cut the 5.05 µs barrier and thus ~485 µs/step | `communication/torch_symm_triton/barrier.py` (`_wait_signal_ldpoll`, env `MCORE_SYMM_BARRIER_LDPOLL`) — **reverted**; `dev/moe_fused/harness_comm.py` | microbench only | **−2.3% (regression)** | Bit-exact both variants (AGV max abs 0.0, RSV max rel 0.0 over 64 trials) | session `qwen-comm` job 5601961, exec `04bdfdd6` | **Rejected** — per-step comm 693.7 µs baseline vs 709.8 µs ldpoll (3 alternating reps each, no overlap). The barrier-only kernel is unchanged at 5.75–5.83 µs vs a 0.72 µs empty kernel, so the 5.05 µs is the 4-way system-scope flag round trip itself, not polling granularity — a hardware/driver floor a Triton rewrite does not move. Patch reverted; `megatron/` back to the session-4 state. |
| QWEN-020 | 2026-07-25 | `_moe_sum_kernel` spends 7.79 µs/layer moving ~6.3 MB — 7× its own bandwidth floor — because the per-topk-slot locality test is a uniform scalar branch on a dependent `routing_map` load, serialising the topk walk. Predicate it into the load mask and widen the K tile to the full hidden size | `vllm_fused_moe.py` (`_moe_sum_kernel_fast`, env `MCORE_MOE_SUM_FAST`, `_FAST_MOE_SUM_MAX_BLOCK_K`) | **24,403.7 tok/s** | **+1.83% vs QWEN-019**, **+4.90% vs CLEANBASE-S6** | **Bit-exact** (max_abs 0.0 *and* max_rel 0.0 at 128/256/384/512) + coherent | session `qwen-comm` job 5601961, run `e2e/moesum-1785026515` | **Accepted** — `_moe_sum` 8.14 → 5.97 µs/layer (1.36×), whole MoE call 80.69 → 78.77 µs at 256 tokens. TPOT 10.682 → 10.490 ms/tok. Same reduction order and same fp32 arithmetic (masked slots add an exact 0.0), so bit-exactness is structural, not luck. Default OFF; enable with `MCORE_MOE_SUM_FAST=1`. New best = **71.79%** of vLLM 33,994.5 |
| QWEN-021 | 2026-07-25 | The router's `torch.softmax` + `torch.topk(sorted=False)` pair costs 436.8 µs/step (4.4%) for a `[256, 128]` fp32 reduction; one CTA per token can do softmax in registers and select the top-8 by max-then-mask in a single kernel | `megatron/core/inference/moe/router_topk.py` (new), `megatron/core/transformer/moe/router.py` (`InferenceTopKRouter._forward` fused-path branch), env `MCORE_ROUTER_FUSED_TOPK`; `dev/moe_fused/harness_routertopk.py` (new) | **25,352.9 tok/s** | **+3.89% vs QWEN-020**, **+8.98% vs CLEANBASE-S6** | **Bit-exact probs and identical expert sets** at 128/256/384/512 × 2 seeds | session `qwen-comm` job 5601961, run `e2e/routertopk-1785027029` | **Accepted, largest single win of the session.** 4 kernels → 1: 16.41 → 4.10 µs under graph replay at 256 tokens (4.00×), eager per-kernel 15.54 → 2.04 µs. TPOT 10.490 → 10.097 ms/tok (−393 µs/step vs a predicted ~312 µs). Default OFF; `MCORE_ROUTER_FUSED_TOPK=1`. New best = **74.58%** of vLLM 33,994.5 |
| QWEN-022 | 2026-07-25 | Now that every CTA streams the routing pairs to build its own histogram (QWEN-019), the scatter can be a second streaming pass in the same kernel — no atomics, no grid sync — taking the decode indirection-table build from 2 launches to 1 | `vllm_fused_moe.py` (`_align_single_kernel`, `_moe_align_block_size_single`, env `MCORE_MOE_FUSED_SCATTER`), `dev/moe_fused/harness_scatterfuse.py` (new) | **25,495.9 / 25,441.4 tok/s** (two runs) | **+0.46% vs QWEN-021** (mean of two), inside the noise band | **Bit-exact** whole-MoE output at 128/256/384/512 + table equality (16 cases: `npp`, `expert_ids`, row multiset, and every row placed under an expert id that owns it) | session `qwen-comm` job 5601961, runs `e2e/scatterfuse-1785027598`, `e2e/scatterfuse-rep2-1785027955` | **Accepted, marginally.** Align call 10.08 → 8.23 µs and whole MoE call 82.17 → 79.56 µs at 256 tokens (1.033×), predicting 1.24%; e2e delivered **+0.35% and +0.55%** in two runs. Both runs beat QWEN-021, so the sign is reliable, but the size is not — the honest attribution is ~0.5%. The candidate is a **wash at 384 tokens (1.007×) and a slight loss at 512 (0.997×)**, since the second streaming pass scales with pairs while the removed launch does not. Default OFF; `MCORE_MOE_FUSED_SCATTER=1` (requires `MCORE_MOE_FUSED_COUNT=1`). New best = **75.00%** of vLLM 33,994.5 |
| PROFILE-S6 | 2026-07-25 | Re-profile at the session-6 configuration (all six gates on) and re-rank, since the step lost ~10% of its wall time and ~820 µs of routing | none (`profile_insession.sh`) | n/a (BS256/OSL128 profile run) | n/a | n/a | session `qwen-comm` job 5601961, `prof/s6-all-1785028240/mcore_profile.{nsys-rep,sqlite}` | **Routing is done as a lever.** One decode step: wall **9.097 ms** (was 9.933), GPU-busy 7.183 (was 7.884), idle 1.914 (21.0%), **1170 kernels** (was 1362). Routing/permute is now **430 µs in 98 kernels** (was 1251 µs in 242). New ranking: MoE expert GEMM 2469 µs (27.1%, closed), **host-side idle 1914 µs (21.0%)**, dense GEMM 1084, comm 1063 (closed), attention 924, elementwise 488, norm 450, routing 430. The idle is concentrated in **three host gaps totalling ~1237 µs/step**: 548 µs after `index_elementwise_kernel`, 386 µs after `vectorized_elementwise_kernel`, 303 µs after `CatArrayBatchedCopy_vectorized` — sampling/detokenize/scheduling between graph replays |
| CLEANBASE-S7 | 2026-07-26 | Fresh same-session OSL1024 reference at the seven-gate session-6 best config, in session `qwen-attnstate`, before the host-side `initialize_attention_state` lever | none (all seven gates on) | **25,805.9 tok/s** | baseline (session 7); +1.22% vs QWEN-022's 25,495.9 (session drift) | Coherent, 5/5 iters | session `qwen-attnstate` job 5613090, run `e2e/ref-s7-1785084185` | Clean reference; **75.91%** of vLLM 33,994.5. Per-iter 25,782.5 / 25,783.6 / 25,814.8 / 25,842.3 / 25,806.7 (spread 0.23%, the tightest reference this campaign has had). avg_latency 9,867 ms, TPOT 9.920 ms/tok |
| QWEN-023 | 2026-07-26 | HOSTGAP-S6's ~311 µs/step of `initialize_attention_state` is recomputation, not computation: at fixed BS256 the request set, sampling metadata, KV block table and CUDA-graph selection are identical for 128 consecutive decode steps, and only the KV sequence lengths advance. Recompute those and reuse the rest, invalidated by a request-layout version counter | `megatron/core/inference/contexts/dynamic_context.py` (`_bump_request_layout_version`, `_incr_attn_state_cache_key/_store/_snapshot/_verify`, `_incremental_attention_state_update`, env `MCORE_INFER_INCR_ATTN_STATE`, `MCORE_INFER_INCR_ATTN_STATE_VERIFY`, profiling gate `MCORE_INFER_ATTN_PROF`), `megatron/core/inference/contexts/attention_context/mha_metadata.py` (`restore_state_data`), `dev/moe_fused/harness_attnstate.py` (new) | **26,032.1 / 26,092.5 tok/s** (two OFF/ON pairs) | **+0.94%** (pairwise +0.88% and +1.01%) | **130 consecutive decode steps bit-identical** (cached vs freshly recomputed, every host and GPU bookkeeping buffer) + temperature-0 coherence output byte-identical to gate-OFF | session `qwen-attnstate` job 5613090, runs `e2e/incr-attn-on-1785085393`, `e2e/incr-attn-on-rep2-*` | **Accepted.** Host CPU per call **174.1 → 51.7 µs (3.37×)**; the residual is the H2D bookkeeping copy, which is irreducible. TPOT 9.915 → 9.823 ms/tok (−92 µs/step). Across all twenty timed iterations the two gates are **fully separated** (slowest ON 25,992.1 > fastest OFF 25,893.9), so the sign is certain even though the size is under 1%. Delivered ~28% of the 3.4% ceiling: the removable phases are ~72% of the call, and the H2D bookkeeping copy that remains cannot be cached. Default OFF; `MCORE_INFER_INCR_ATTN_STATE=1`. New best = **76.67%** of vLLM 33,994.5 |
| CGTRACE-CONTROL | 2026-07-26 | HOSTGAP-S6 could not tell whether the ~900 µs/step of CUDA-graph machinery (a 199 µs `cudaGraphLaunch` plus GPU-side inter-node dispatch) is real or an artifact of `--cuda-graph-trace=node` instrumenting all 1158 graph nodes. Capture the same workload both ways and compare | none (control); `dev/moe_fused/profile_insession.sh` (env `CUDA_GRAPH_TRACE`, plus the hardened nsys stop/qdstrm recovery ported from `profile_host_insession.sh`), `dev/moe_fused/analyze_cgtrace.py` (new) | n/a (two BS256/OSL128 profile runs) | n/a | n/a | session `qwen-attnstate` job 5613090, `prof/cgt-node-1785086713` and `prof/cgt-graph-1785086964` | **The machinery is real; node-mode instrumentation is not inflating the step.** Same rank, same steady-state decode window: step period **8882.5 µs (node) vs 8945.1 µs (graph)** — 0.71% apart and in the *wrong* direction for an instrumentation artifact — and host `cudaGraphLaunch` **190.0 vs 184.7 µs** (2.8%). Node mode traces 1169 kernels/step, matching PROFILE-S6's 1170. **HOSTGAP-S6's wall-time attributions therefore all stand**, and lever 4 (reduce graph node count, ~0.78 µs/node) is live rather than chasing a measurement artifact. Caveat: `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` is too sparsely populated (417 rows total, n=6 in-window on the analyzed device) to independently confirm the *GPU-side* inter-node dispatch component; only the step period and host submit cost are settled |
| HOSTGAP-S6 | 2026-07-26 | Attribute PROFILE-S6's three inter-kernel host gaps to concrete host functions using the recovered host-visibility trace | none (analysis); `dev/moe_fused/analyze_hostgaps.py` (new) | n/a (analysis) | n/a | n/a | session `qwen-host` job 5607600, `prof/hosts6-1785048883` | **All three gaps attributed; the largest is a true serial data dependency, not a scheduling artifact.** Per step (GPU-only trace, mean of 4 ranks, 9.134 ms step / 1965.7 µs idle): **G1 `index_elementwise`→`vectorized_elementwise` 571.4 µs** = `transfer_samples_to_cpu`+`active_request_mask`+`update_requests`+engine window+`initialize_attention_state`, **CPU-bound Python** (100% of samples Running in `_PyEval_EvalFrameDefault`, only 8.6% of the gap in CUDA API, 0.5% in `cudaStreamSynchronize`); **G2 `vectorized_elementwise`→`vectorized_gather` 331.0 µs in 2 instances** (267.9 µs inside `initialize_attention_state`+`transfer_bookkeeping_to_gpu`+`forward_pass` head, **CPU-bound Python**; 63.1 µs inside `sampling`, launch+`cudaStreamSynchronize`); **G3 `CatArrayBatchedCopy`→`rmsnorm_fwd` 305.4 µs** = **one `cudaGraphLaunch` costing 199.1 µs median** for a 1158-node graph (0.172 µs/node), driver-side, host Running in `cuGraphLaunch`. No rank asymmetry (≤4%). Idle splits **1065 µs real host Python (11.7% of step)** vs **900 µs graph machinery (9.9%)** that is `--cuda-graph-trace=node`-artifact-suspect. Async overlap cannot fix G1: the chain is data-dependent on the current step's sampled tokens, which explains QWEN-007's +0.85% |
| QWEN-025 | 2026-07-26 | QWEN-024's decision-gate measurement showed 84% of `post_process_requests` is reducible by collapsing the per-request loop body — unvectorizable is not the same as irreducible. Add a decode fast path that is structurally unable to touch the request termination state machine | `dynamic_engine.py` (`_post_process_requests_decode_fast`, `_ppr_cache_epoch` invalidation at all three record-mutation sites, env `MCORE_INFER_FAST_POST_PROCESS`, `..._VERIFY`), `dev/moe_fused/harness_updreq.py` (`run_ppr_equivalence`, in-process gate A/B) | **26,361.7 tok/s** (three-pair ON mean); best run 26,430.7 | **+0.98%** vs same-session gate-OFF (pairwise +0.99% / +1.26% / +0.68%; +0.59% on a best-4-of-5 trim) | 400-step two-engine equivalence with 10 finisher steps and 200 steps past the token limit; 300 steps under VERIFY; coherence byte-identical | session `qwen-updreq` job 5616264, runs `e2e/ppr-{off,on}{,-r2,-r3}-*`, profiles `prof/g2-off-1785106875` and `prof/g2-on-1785107105` | **Accepted.** `post_process_requests` **211.0 → 27.2 µs/step (7.77×)**; whole post-sampling host chain **372.4 → 148.0 µs/step** across QWEN-024+025. The fast path **declines any step on which a request finishes**, so the termination state machine is unreachable from it — that is the safety argument, not extra testing. cProfile 261 calls/step vs the pre-registered <500 threshold. 13 of 15 ON iterations beat every one of the 15 OFF iterations. G1 −56.1 µs, idle −77.9 µs, kernels/step unchanged at 1169. **Only ~1/3–1/2 of the 183.9 µs host saving converts** (vs ~1:1 for QWEN-024) because `async_bookkeep` was already partly overlapped with GPU work. Default OFF. New best = **77.75%** of vLLM 33,994.5 |
| QWEN-024 | 2026-07-26 | HOSTGAP-S6 lever 2. Measure the three parts of the post-sampling chain separately, name the unlabelled window, and attack only the part that is not per-request Python object churn: `update_requests` and `active_request_mask` are already vectorized, and roughly half their whole-tensor op cost is provably dead work at one generated token per request | `dynamic_context.py` (`_write_decode_token_bookkeeping_fast`, `_write_token_bookkeeping_reference`, `_verify_decode_token_bookkeeping`, env `MCORE_INFER_VEC_UPDATE_REQS`, `..._VERIFY`), `text_generation_controller.py` (no-finisher short-circuit), `dev/moe_fused/harness_updreq.py` (new) | **26,128.2 / 26,270.3 tok/s** (two pairs) | **+0.90% vs CLEANBASE-S8** (pairwise +0.71% and +1.10%) | 400-step two-context equivalence with 30 mid-batch terminations + block-boundary crossing; 300 steps under VERIFY; coherence byte-identical | session `qwen-updreq` job 5616264, runs `e2e/vecupd-on-1785100576`, `e2e/vecupd-on-rep2-1785101960` | **Accepted.** Named the unlabelled engine window: it is `post_process_requests` via `async_bookkeep`, invisible to HOSTGAP-S6 only because the engine's NVTX helper is inert unless `_nvtx_enabled` is set. It is **187.9 µs/step of pure per-request Python object churn** (1,719 `len()`, 256 dict lookups, 256 appends, 3 tensor ops per step) and was **deliberately left alone** per the falsification rule. The other half was op-eliminated: host chain 372.4 → 293.7 µs/step, `update_requests` 137.6 → 74.0 µs (1.86×). TPOT −88 µs/step. Default OFF; `MCORE_INFER_VEC_UPDATE_REQS=1`. New best = **77.28%** of vLLM 33,994.5 |
| CLEANBASE-S8 | 2026-07-26 | Fresh same-session OSL1024 reference at the eight-gate best config, in session `qwen-updreq`, before the `update_requests` host lever | none (all eight gates on) | **25,944.9 tok/s** | baseline (session 8); −0.57% vs QWEN-023's 26,092.5 (session drift) | Coherent, 5/5 iters | session `qwen-updreq` job 5616264, run `e2e/ref-s8-1785099726` | Clean reference; **76.32%** of vLLM 33,994.5. Per-iter 25,669.0 / 25,983.1 / 26,011.2 / 26,023.4 / 26,041.7 — iteration 1 is a 1.3% cold outlier, iterations 2–5 span 0.23%. avg_latency 9,820.6 ms, TPOT 9.867 ms/tok |
| PROFILE-HOST-S6 | 2026-07-25 | Capture a host-visible trace (`osrt` + CPU sampling + Python sampling) so the ~1237 µs/step of host gaps PROFILE-S6 found between CUDA-graph replays can be attributed to concrete host call sites | none (`dev/moe_fused/profile_host_insession.sh`, new; `dev/moe_fused/dispatch_host_profile.sh`, new) | n/a (BS256/OSL128 profile run) | n/a | n/a | session `qwen-comm` job 5601961 **preempted mid-task**; replacement session `qwen-host` job 5607600 on `nvl72166-T15`, exec `d920241ea00a41618c63a75fefcb9ea2` | **Capture dispatched and running, not confirmed complete within the time box.** Job 5601961 was PREEMPTED (not expired) before any capture could run, so the whole capture had to be re-queued on a fresh allocation that only scheduled at the deadline. Artifact target `sessions/qwen-host/prof/hosts6-1785048883/mcore_host_profile.{nsys-rep,sqlite}`. **No gap attribution was performed.** Next session must first check whether the detached exec produced the artifact |

## Session 2 (2026-07-23) — conclusion & recommendation

**Best shippable config stays: legacy async / nvls dispatcher / vLLM grouped-GEMM
backend / FC1+SwiGLU fusion (QWEN-002) = 22,398.9 tok/s (65.9% of vLLM 33,994.5).**

Every accessible knob was swept and rejected (QWEN-003…009). Key learnings:
- Decode at OSL1024 is **compute/comm-bound, not idle-bound** (async serial +0.85%).
- The gap is **structural**: mcore runs the MoE decode as a chain of discrete
  kernels — router gemm → softmax → topk → count/scatter/metadata → **exposed
  NVLS AllGather-V dispatch** → grouped GEMM (FC1+SwiGLU, FC2) → moe_sum →
  **exposed NVLS ReduceScatter-V combine** — all on the per-layer critical path.
  vLLM/TRT-LLM does the equivalent as one fused MoE (fused routing, cutlass
  grouped bmm, fused finalize) with **0 exposed comm**.
- Built-in fusions (`--moe-router-fusion`, `--moe-permute-fusion`) and
  `sampling-backend=flashinfer` are wired to training / non-full-graph paths and
  are **incompatible** with the `inference_optimized` top-8 + full-iteration-graph
  contract.

**To actually close the 1.5× gap (large, multi-session work), in priority order:**
1. **Eliminate exposed NVLS comm** (~16% + it gates the critical path): overlap
   dispatch/combine with expert GEMM via chunked/pipelined experts, or fuse the
   combine ReduceScatter-V with the FC2 epilogue + moe_sum (finalize fusion).
2. **Fuse the routing chain** honoring the inference top-8 contract: one kernel
   for softmax+top8+count+scatter-metadata (cuts ~18% of small serial kernels).
3. **A TRT-LLM-style single fused MoE decode kernel** (dispatch+groupedGEMM+
   SwiGLU+finalize) — the real vLLM-parity path; QWEN-001's full fusion was
   slower, so this needs a cutlass/CUTE grouped-bmm core, not Triton.

## Detailed records

### VLLM-BASELINE — DP4 with expert parallelism

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Establish the fresh vLLM BS256 Nsight target |
| Code revision | `808c475352de6c3693b182048f174736af82356e`; skill files untracked, Megatron source clean |
| Changed files | none |
| Runtime flags | TP1, DP4, `--enable-expert-parallel`, max model length 4096, max sequences 512 |
| Image | `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/images/87e4947c6ce36433.sqsh` |
| Checkpoint / tokenizer | `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/checkpoints/qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/DP4/EP enabled |
| Workload | gsm8k, BS256, OSL128, one BS8/OSL32 warmup request, one timed request |
| Job / run | `5547673`; `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/vllm-qwen30b-nsys-20260722-093437` |
| Throughput | 23,606.708 tok/s |
| Latency / TPOT | 1,368.153 ms / 10.844 ms-token |
| Correctness | Benchmark completed 256/256 requests |
| Nsight artifacts | `vllm_profile.nsys-rep`; `vllm_profile.sqlite` in the run directory |
| Result | vLLM target established |
| Next action | Record the matching mcore EP4 profile |

### MCORE-BASELINE — EP4/TP1

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Establish the fresh mcore EP4 BS256 Nsight starting point |
| Code revision | `808c475352de6c3693b182048f174736af82356e`; skill files untracked, Megatron source clean |
| Changed files | none in Megatron source; profile harness sets `max_requests=256` |
| Runtime flags | TP1, PP1, EP4, ETP1, NVLS dispatcher, vLLM grouped GEMM, inference-optimized transformer, full-iteration inference CUDA graphs |
| Image | Cog dev image `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/images/ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` under the user checkpoint root |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/EP4 |
| Workload | gsm8k, BS256, OSL128, one BS8/OSL32 warmup request, one timed request |
| Job / run | `5553135`; `/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space/runs/qwen-30b-nsys-20260722-161020` |
| Throughput | 12,346.092 tok/s |
| Latency / TPOT | 2,590.135 ms / 20.735 ms-token |
| Correctness | Checkpoint loaded and benchmark completed 256/256 requests |
| Nsight artifacts | `mcore_profile.nsys-rep`; `mcore_profile.sqlite` in the run directory |
| Result | mcore reaches 52.30% of vLLM; 47.70% below target |
| Next action | Analyze the timed-request A/B windows before changing code |

### QWEN-001 — MoE decode mega-fusion (FC1+SwiGLU+FC2+topk-reduce)

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | Fusing the whole MoE expert path into one Triton kernel (removing the `bounded_silu_mul` + `_moe_sum` kernels and two intermediate HBM round-trips) beats the 4-kernel `vllm_fused_moe` path, which is 40.5% of decode GPU-busy time |
| Code revision | branch `perf/moe-fused-decode-gemm` off `808c475352de6c3693b182048f174736af82356e`, dirty |
| Changed files | `megatron/core/inference/moe/fused_moe_decode.py` (new kernel), `dev/moe_fused/harness.py` (new standalone correctness+timing harness) |
| Runtime flags | standalone microbench; Qwen3-30B decode shapes H=2048, moe_ffn=768, 32 local experts, top-8, 256 valid tokens |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | n/a (synthetic weights, reference = production `vllm_fused_moe`) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 (session `qwen-moe-kernel`) |
| Workload | microbench, 10 warmup + 100 timed CUDA-event iters |
| Job / run | session `qwen-moe-kernel`, exec runs `manual1784770098/200/321` |
| Throughput | not measured end-to-end (rejected at microbench) |
| Latency / TPOT | kernel: reference 175 µs vs fused 258 µs (best sweep 0.80×; larger tiles OOM shared memory) |
| Correctness | Pass — max_abs_diff 2.6e-5, max_rel 0.15 on a 1e-4 floor, `allclose(rtol=2e-2,atol=2e-2)` True |
| Nsight artifacts | none (microbench) |
| Result | Rejected — fused kernel is consistently 20–50% slower than the reference in every same-run comparison |
| Next action | Root cause: one CTA per token-block serializes the H=2048 FC2 output loop (vs the reference's N-parallel multi-CTA GEMMs), 3× atomic traffic to `out`, and shared-memory pressure caps tile sizes; the HBM/launch savings are negligible under CUDA graphs. Pivot to either (a) partial FC1+SwiGLU epilogue fusion only, or (b) the 11.5% exposed NVLS all-gatherv/reduce-scatter-v communication |

### QWEN-002 — FC1+SwiGLU epilogue fusion

| Field | Value |
|---|---|
| Date | 2026-07-22 |
| Hypothesis | The 4-kernel MoE path (FC1→2N intermediate→`bounded_silu_mul`→FC2→reduce) wastes a full HBM round-trip of the `[num_valid, 2N]` intermediate and a whole kernel launch. Computing gate & up in the same FC1 program and applying `SiLU(gate)*up` in the fp32 epilogue — writing the `[num_valid, N]` activated intermediate directly — removes both while keeping FC2's N-parallel tiling |
| Code revision | branch `perf/moe-fused-decode-gemm`, dirty |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (add `FUSE_SWIGLU` constexpr to `_fused_moe_kernel`, `fuse_swiglu` to `_invoke_fused_moe_kernel`, `fuse_fc1_activation` path in `vllm_fused_moe`), `megatron/core/transformer/moe/experts.py` (`_vllm_forward` passes `fuse_fc1_activation=True`), `dev/moe_fused/harness_fc1.py` (new A/B harness) |
| Runtime flags | vLLM grouped-GEMM backend, SwiGLU; microbench at Qwen3-30B decode shapes H=2048, moe_ffn=768, 32 local experts, top-8, 256 valid tokens |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | n/a for microbench (synthetic weights; reference = unfused `vllm_fused_moe`) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 (session `qwen-moe-kernel`) |
| Workload | microbench, 10 warmup + 200 timed CUDA-event iters, 3 repeats |
| Job / run | session `qwen-moe-kernel`, exec runs `fc1c…`/`fc1t…` |
| Throughput | **22,741.9 tok/s** (fused) vs **22,617.7 tok/s** (same-env, fusion off) → **+0.55%**. gsm8k BS256 OSL1024, 2 warmup + 5 timed iters |
| Latency / TPOT | e2e TPOT 11.257 (fused) vs 11.319 ms/tok (off) → −0.55%. MoE-path kernel microbench: reference 174–182 µs vs fused 141–143 µs → **1.24–1.27×** (3 repeats) |
| Correctness | Pass — fused vs unfused `vllm_fused_moe`: max_abs_diff 3.9e-5, allclose; e2e coherence prompts coherent (2+2=4, capital of France = Paris) |
| Nsight artifacts | none (in-session A/B benchmark, not profiled) |
| Result | Accepted — correct, MoE kernel 1.25×, e2e +0.55% throughput / −0.55% TPOT, no regression. Unfused path byte-identical (all edits guarded by the `FUSE_SWIGLU` compile-time constexpr) |
| Next action | The 1.25× MoE-kernel win only nets +0.55% e2e at OSL1024, so MoE FC1 activation is not the wall-time bottleneck at the throughput regime — re-profile the OSL1024 decode (not the OSL128 profile trace) to find the true dominant cost before the next change. NOTE: at OSL1024 mcore already reaches ~22.6k tok/s; the ledger's 52% "gap" is an artifact of the OSL128 single-request profile workload |

### QWEN-011 — fused MoE indirection-table build (lever #1)

| Field | Value |
|---|---|
| Date | 2026-07-24 |
| Hypothesis | The decode "routing/permute kernel storm" is dominated by the 5 tiny serial kernels of `_moe_align_block_size_cuda_graphable` (init + count + prefix + fill + scatter) run per MoE layer × 48. Merging init + prefix + fill into one `_prefix_fill_init_kernel` cuts it to 3 kernels/layer (−96 launches/step) and narrows the vLLM gap without touching numerics. |
| Code revision | branch `perf/moe-fused-align` off `perf/moe-routing-fusion` (which carries QWEN-002), dirty |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (`_prefix_fill_init_kernel`, `_moe_align_block_size_fused`, env `MCORE_MOE_FUSED_ALIGN`, wired into `vllm_fused_moe`), `megatron/core/transformer/moe/experts.py` (`MCORE_FUSE_FC1_ACT` env gate), `dev/moe_fused/harness_align.py` (new A/B harness) |
| Runtime flags | `MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`; nvls dispatcher, vLLM grouped-GEMM backend, full-iteration CUDA graphs |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters (throughput); microbench 10 warmup + 200 timed CUDA-event iters (correctness) |
| Job / run | session `qwen-fuse` runs `fusedalign` (e2e) + `acb5ea1b` (microbench) |
| Throughput | **22,696.4 tok/s** vs 22,269.5 (fusion-on) / 22,241.5 (clean) → **+1.92% / +2.04%** |
| Latency / TPOT | avg_latency 11,257 ms vs 11,494 ms (clean) |
| Correctness | **Bit-exact** vs the 5-kernel path: max_abs_diff 0.0, allclose True; e2e coherence prompts coherent (2+2=4, Paris) |
| Nsight artifacts | in-session profile `sessions/qwen-moe-kernel/prof/1784930600/mcore_profile.{nsys-rep,sqlite}` (BS256/OSL128, fused-align ON). Kernel-name counts **confirm the mechanism**: `_init_sorted_ids_kernel`=0, `_prefix_sum_kernel`=0, `_fill_expert_block_ids_kernel`=0 (all removed), `_prefix_fill_init_kernel`=`_count_local_tokens`=`_scatter_token_indices_kernel`=76032 (align now 3 kernels/layer vs 5). microbench MoE-call 142.65→138.57 µs (1.029×) |
| Result | **Accepted** — correct (bit-exact), e2e +1.9% with tight iter variance (22.65–22.71k), no regression. Profile confirms −2 indirection kernels/MoE-layer/step. Default OFF (env-gated) so the untouched path stays byte-identical; enable with `MCORE_MOE_FUSED_ALIGN=1`. |
| Next action | Reduces align 5→3; the remaining serial dep (count → prefix_fill_init → scatter) can only merge further via a cooperative-grid launch (risk: CUDA-graph capture). Profile to confirm per-step kernel-count drop, then target the FC1/FC2 grouped-GEMM efficiency gap (lever #2, needs a cutlass/CUTE grouped bmm — Triton launch count is already minimal at 2/layer). |

### QWEN-012 — grouped-GEMM roofline / padding decision gate

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | The ~40% of decode GPU time in `_fused_moe_kernel` is recoverable GEMM inefficiency, so a cutlass/CuTe BF16 grouped GEMM is worth writing. Measure achieved TFLOP/s, achieved weight bandwidth against a *measured* ceiling, and the indirection-table padding waste before writing any kernel. |
| Code revision | branch `perf/moe-fused-align` off `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, dirty (carries QWEN-002 + QWEN-011) |
| Changed files | `dev/moe_fused/harness_roofline.py` (new, analysis only — no Megatron source change) |
| Runtime flags | microbench at one EP4 rank of decode: hidden 2048, moe_ffn 768, 128 global / 32 local experts, top-8, `num_tokens_hint`=256 (= `local_tokens*ep_size` from `InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate()`), max_tokens 1024 |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` (`nvcr.io/nvidia/pytorch:26.06-py3`) |
| Checkpoint / tokenizer | n/a (synthetic weights at production shapes) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 of the 4-GPU node, session `qwen-cutlass` (job 5598012) |
| Workload | microbench, 20 warmup + 300 timed CUDA-event iters, 3 repeats (spread <1%) |
| Job / run | session `qwen-cutlass` exec `c05fd5f771b44bf7a002886a1242c1c4` |
| Throughput | not applicable (analysis) |
| Latency / TPOT | FC1 (fused SwiGLU) 52.25 µs, FC2 19.88 µs, **total 72.13 µs** per MoE layer per rank |
| Correctness | n/a (no functional change) |
| Nsight artifacts | none (CUDA-event microbench + torch profiler) |
| Result | **Memory-bound, not FLOP-bound — Stage 4 rejected before implementation.** (1) Measured bandwidth on this GB200: d2d copy 6.851 TB/s aggregate, Triton vectorized streaming read **6.081 TB/s** (torch `.sum()` gives 4.05 TB/s and is a poor proxy — do not use it as the ceiling). (2) Every decode step must read all 32 local experts' weights: 302.0 MB per layer per rank ⇒ **49.66 µs bandwidth floor**, vs 72.13 µs measured = **1.45× off roofline**. (3) Achieved 63.4 (FC1) / 83.2 (FC2) TFLOP/s on valid FLOPs — ~3% of GB200 BF16 dense peak; 246/323 TFLOP/s even counting padded rows. (4) Padding waste is large but almost free: `num_tokens_post_padded` vs 527 real local token-expert pairs is 784/1024/2048/4096 at BLOCK_SIZE_M 16/32/64/128 (32.8%/48.5%/**74.3%**/87.1% dead rows), yet dropping BLOCK_SIZE_M from the production 64 to 16 buys only ~1.2×, because each expert's weights are read exactly once per M-tile row-block regardless. (5) A 63-config tile sweep found 58.93 µs (1.225×) with one shared config and 57.24 µs (1.26×) with per-GEMM configs — i.e. **most of the 1.45× ceiling is reachable in Triton**, leaving ≤1.15× for any hand-written kernel. |
| Next action | Do not write a CUTLASS/CuTe grouped GEMM. Land the tile retune (QWEN-013), then redirect to the next-ranked levers from PROFILE-DECODE: exposed EP comm (~16%) and the routing chain (~18%). |

### QWEN-013 — per-GEMM decode tile retune

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | QWEN-012's 1.45× roofline gap is tile-quantized SM occupancy, not the GEMM implementation. `_get_default_config` picks `BLOCK_SIZE_N=128` for both passes, leaving FC1 at ceil(768/128)=6 N-tiles × 32 M-tiles = 192 CTAs on 148 SMs (1.3 waves, ~30% tail idle). FC1 (N=768) needs a *small* `BLOCK_SIZE_N` to manufacture enough CTAs; FC2 (N=2048) has plenty and prefers a *large* one. vLLM shares one config across both passes and cannot express this. |
| Code revision | branch `perf/moe-fused-align` off `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, dirty |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (`_get_decode_tuned_configs`, `_TUNE_DECODE_GEMM` env gate, `config_fc1`/`config_fc2` threaded into the two `_invoke_fused_moe_kernel` calls and their grid sizing), `dev/moe_fused/harness_gemmtune.py` (new A/B + profiler + CUDA-graph harness), `dev/moe_fused/harness_align.py` (fix a module-vs-function import bug, see below) |
| Runtime flags | `MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`; nvls dispatcher, vLLM grouped-GEMM backend, `transformer_impl=inference_optimized`, full-iteration CUDA graphs. Tuned configs: shared `BLOCK_SIZE_M=16`, `BLOCK_SIZE_K=64`, `GROUP_SIZE_M=1`; FC1 `BLOCK_SIZE_N=64`, 4 warps, 3 stages; FC2 `BLOCK_SIZE_N=256`, 8 warps, 4 stages. Applies only for `num_tokens_hint <= 512`; prefill keeps upstream's heuristic. |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` under the user checkpoint root |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters (throughput); microbench 20 warmup + 300 timed iters × 3 repeats and a 50-iter torch-profiler window (attribution) |
| Job / run | session `qwen-cutlass2` (job 5600047) run `gemmtune-1785010791`; reference `ref0-1785010433` (CLEANBASE-S4) |
| Throughput | **23,636.0 tok/s** vs **22,657.7** same-session reference → **+4.32%**. Per-iter 23,553–23,668 tok/s |
| Latency / TPOT | avg_latency 10,802 ms vs 11,282 ms; TPOT 10.831 vs 11.299 ms/tok (−4.14%). Per-kernel GPU time: `_fused_moe_kernel` (both GEMMs) 77.65 → **61.89 µs** (1.255×); whole `vllm_fused_moe` call 100.64 → **84.31 µs** (1.194×) |
| Correctness | **Bit-exact** — max_abs_diff 0.0 and max_rel 0.0 at 128/256/384/512 valid tokens. Expected: `BLOCK_SIZE_K` is unchanged, so the fp32 reduction order along K is identical and only output-tile partitioning changes. e2e coherence prompts coherent (2+2=4, Paris, 3+2 cows = 5 cows) |
| Nsight artifacts | none (torch-profiler per-kernel attribution instead; timing composition change is fully explained by the two GEMM launches) |
| Result | **Accepted** — bit-exact, +4.32% e2e with tight variance, no regression. Default OFF (`MCORE_MOE_GEMM_TUNE`) so the untouched path stays byte-identical. New best mcore = **69.53%** of the vLLM 33,994.5 baseline (was 66.65%). |
| Next action | Two follow-ups, in order: (1) the remaining GEMM headroom is only 1.15× (57–62 µs vs the 49.66 µs floor), so stop here on the GEMM and attack **exposed EP comm (~16%)**; (2) `_get_decode_tuned_configs` is hand-tuned at one shape — if other MoE geometries adopt it, replace the constants with a small autotune keyed on (N, E, M). **Harness caveat worth knowing:** `import megatron.core.inference.moe.vllm_fused_moe as vfm` yields the *function*, not the module, because `moe/__init__.py` does `from .vllm_fused_moe import vllm_fused_moe`. Patching module flags on it silently no-ops, which makes an A/B compare one path against itself. This invalidated my first retune measurement (and affects QWEN-011's *microbench* numbers, though not its e2e result, which used the real env var). Both harnesses now use `importlib.import_module` and assert the object is the module. With the fix, the align A/B re-measures at 148.67 → 124.13 µs eager wall (1.198×). |

### QWEN-014 — flashinfer `cutlass_fused_moe` backend A/B

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | `--inference-grouped-gemm-backend flashinfer` is already wired (`InferenceGroupedMLP._flashinfer_forward`, `experts.py:1057`) and accepts BF16, so QWEN-010's "blocked for SwiGLU" note is wrong and this backend has never actually been benchmarked. Enabling it should beat the vLLM Triton grouped GEMM, since it fuses permute + both GEMMs + activation into one launch. |
| Code revision | branch `perf/moe-fused-align`, dirty. No Megatron source change — the two integration bugs found were reproduced and fixed *in the harness only*, because the measurement then showed the backend is not worth adopting. |
| Changed files | `dev/moe_fused/inspect_flashinfer.py` (new), `dev/moe_fused/harness_flashinfer.py` (new) |
| Runtime flags | microbench at one EP4 rank: hidden 2048, moe_ffn 768, 128 global / 32 local experts, top-8, valid 256, max_tokens 1024, ep_size 4, ep_rank 0 |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh`; flashinfer **0.6.14**, torch 2.13.0a0, CUDA 13.3, GB200 sm100. `flashinfer_jit_cache` / `flashinfer_cubin` are **not** installed, so the first call downloads cubins from `edge.urm.nvidia.com` (~2 min from this cluster; it does have egress). Budget that for CUDA-graph warmup if this path is ever enabled. |
| Checkpoint / tokenizer | n/a (synthetic weights at production shapes; reference = production `vllm_fused_moe`) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200 of the 4-GPU node, session `qwen-cutlass2` (job 5600047) |
| Workload | microbench: numerics single-shot, then 300 CUDA-graph replays × 3 repeats |
| Job / run | session `qwen-cutlass2` execs `7590adcb6e754ed8a7c32a58decb2f9b`, `659d263321e8df73...` |
| Throughput | not measured e2e — rejected at microbench |
| Latency / TPOT | CUDA-graph device time, valid=256: `vllm_fused_moe` default tiles **97.97 µs**, `vllm_fused_moe` QWEN-013 tiles **83.63 µs**, `cutlass_fused_moe` **90.81 µs** → **1.079× vs default, 0.921× vs QWEN-013**. (Eager wall-clock flatters cutlass to 1.17× because it is 1 launch vs 6 at ~12 µs/launch; that number is meaningless under full-iteration CUDA graphs.) |
| Correctness | Passes only after fixing the weight order. `[gate\|up]` (mcore's buffer as-is): max_abs 7.14e-4, max_rel **4.92**. `[up\|gate]`: max_abs 3.87e-5, max_rel 0.171 — bf16 rounding noise. So the kernel wants **up\|gate** (w3\|w1). Both pass a loose `allclose(2e-2)`, which is exactly why this would have shipped silently wrong. |
| Nsight artifacts | none (CUDA-graph event timing + torch profiler) |
| Result | **Rejected on measurement — but QWEN-010's stated reason was wrong, and two real mcore bugs were found.** (1) `_resolve_flashinfer_activation_type` (`experts.py:937-951`) maps `F.silu` → `ActivationType.Silu` **without consulting `config.gated_linear_unit`**, unlike `_resolve_mcore_activation_type` right below it which does. With `Silu` the kernel hard-fails: `fc1_expert_weights.size(1) == fc2_expert_weights.size(2) * mInnerDimMultiplier (1536 vs. 768)` — a non-gated activation expects fc1 out == ffn, but a SwiGLU fc1 emits 2×ffn. The correct enum is `ActivationType.Swiglu`, which works. **This — not any kernel limitation — is why the flashinfer backend appears "blocked for SwiGLU".** (2) The kernel expects fc1 as `[up\|gate]`; `_build_concatenated_weights` produces TE's `[gate\|up]`, so even with the enum fixed the results are numerically wrong. Fixing both is contained (an enum branch + a one-time reordered fc1 buffer, ~200 MB/layer extra since it cannot share storage with the TE `param.data` views), but at 0.921× it would still lose to QWEN-013. |
| Next action | Do not adopt the backend for this workload. Do fix the two bugs as a correctness matter on their own merit, since `--inference-grouped-gemm-backend flashinfer` is currently either a hard crash or silently wrong for every gated-activation MoE: guard `_resolve_flashinfer_activation_type` on `config.gated_linear_unit` (→ `Swiglu`), and reorder fc1 to `[up\|gate]` for that path. File as a separate bug-fix PR, not a perf change. |

### QWEN-015 — TRT-LLM-Gen BF16 fused MoE (`trtllm_bf16_routed_moe`) probe

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | vLLM's BF16 SM100 oracle selects `FLASHINFER_TRTLLM` → `trtllm_bf16_moe` / `trtllm_bf16_routed_moe`, so this is the literal kernel vLLM wins with. The `_routed_` variant takes already-computed routing results, which matches mcore's contract, and it fuses SwiGLU + the topk-weighted finalize. Dropping it in behind an env gate should be the highest-value integration. |
| Code revision | branch `perf/moe-fused-align`, dirty. **No Megatron change made** — blocked at the probe stage. |
| Changed files | `dev/moe_fused/harness_flashinfer.py` (probe only) |
| Runtime flags | `num_experts=128, top_k=8, intermediate_size=768, local_expert_offset=0, local_num_experts=32, routing_method_type=RoutingMethodType.TopK, do_finalize=True`, both `use_shuffled_weight`/`weight_layout` combinations |
| Image | Cog dev image `ceecf5c304a5d8bd.sqsh`; flashinfer 0.6.14 (exposes `trtllm_bf16_moe`, `trtllm_bf16_routed_moe`, `convert_to_block_layout`, `reorder_rows_for_gated_act_gemm`) |
| Checkpoint / tokenizer | n/a (synthetic weights at production shapes) |
| Hardware / layout | OCI `oci-hsg`, 1×GB200, session `qwen-cutlass2` |
| Workload | microbench probe (numerics/latency never reached) |
| Job / run | session `qwen-cutlass2` exec `7590adcb6e754ed8a7c32a58decb2f9b` |
| Throughput | not reached |
| Latency / TPOT | not reached |
| Correctness | not reached |
| Nsight artifacts | none |
| Result | **Blocked on weight layout — and it is not the contract mismatch the plan anticipated.** `weight_layout=MajorK, use_shuffled_weight=False` is rejected by the launcher itself (`trtllm_fused_moe_kernel_launcher.cu:770`: `BF16 Moe: weight_layout must be BlockMajorK`). `weight_layout=BlockMajorK, use_shuffled_weight=True` then indexes `weights.size(3)` and throws `IndexError: Index 3 out of bounds for tensor with 3 dimensions`. So the kernel requires **4-D pre-shuffled block-major** weights; mcore's `[E, 2*ffn, H]` 3-D concatenated buffer cannot be passed in any configuration. Notably the dense top-8 `routing_map`, the `ep_size`/`ep_rank` semantics, and the NVLS symmetric-memory output tensor were *not* the blockers — the API accepts all of those. |
| Next action | Deprioritized, with a concrete recipe if revisited: build a one-time load-time weight pass using flashinfer's own helpers — `reorder_rows_for_gated_act_gemm` on fc1 (which also resolves the gate/up interleave that QWEN-014 found) then `convert_to_block_layout` on both — producing separate 4-D buffers. That breaks `_build_concatenated_weights`' storage sharing with TE's `param.data` views, so it roughly doubles expert-weight residency for the reordered copies (~14.5 GB/rank today) and needs its own `--inference-grouped-gemm-backend` value. **Priority is low now**: QWEN-012 measured the grouped GEMM at 1.15× off the weight-bandwidth floor after QWEN-013, and QWEN-014 measured the sibling cutlass kernel at 0.921× vs the retuned Triton path, so the fused-MoE ceiling here is roughly the ~22 µs of routing + finalize this kernel would absorb, not a GEMM win. Attack exposed EP comm (~16%) first. |

## Session 4 (2026-07-25) — conclusion & recommendation

**Best shippable config is now: nvls dispatcher / vLLM grouped-GEMM backend /
`MCORE_FUSE_FC1_ACT=1` (QWEN-002) / `MCORE_MOE_FUSED_ALIGN=1` (QWEN-011) /
`MCORE_MOE_GEMM_TUNE=1` (QWEN-013 + QWEN-013b) = 23,646.0 tok/s
= 69.56% of vLLM 33,994.5** (was 66.65% at the start of this session).

The session was chartered to pursue "the grouped-GEMM / cutlass lever". The
measurement says that lever is nearly exhausted, and says so three
independent ways:

1. **Roofline (QWEN-012).** The decode grouped GEMM must read 302 MB of expert
   weights per layer per rank against a *measured* 6.081 TB/s streaming-read
   ceiling — a 49.66 µs floor. It ran at 72.13 µs (1.45× off) and now runs at
   ~57 µs (1.15× off). Achieved FLOPs are ~3% of BF16 peak, so this is a
   bandwidth problem and no GEMM implementation can fix it. The dramatic-looking
   74.3% indirection-table padding is nearly free for the same reason.
2. **A hand-written kernel is not the lever (QWEN-013).** The entire 1.45× gap
   was tile-quantized SM occupancy, recovered by giving FC1 and FC2 their own
   `BLOCK_SIZE_N` — 12 lines, bit-exact, +4.36% e2e.
3. **The vendor kernels do not beat it (QWEN-014, QWEN-015).** flashinfer's
   `cutlass_fused_moe` is 0.921× the retuned Triton path in CUDA-graph device
   time, and `trtllm_bf16_routed_moe` cannot accept mcore's weight layout at all.

**Remaining gap to vLLM is 1.44×, and it is no longer in the GEMM.**
PROFILE-TUNED confirms it from the trace side: the MoE GEMM is now 32.9% of
decode GPU time and sits only 229 µs/step (2.3%) above its bandwidth floor,
while exposed EP comm is 11.7%, routing/permute 15.8%, and GPU idle 20.6%.
Per PROFILE-DECODE the decode critical path is
attn → router → **exposed NVLS AllGather-V dispatch** → grouped GEMM →
**exposed NVLS ReduceScatter-V combine**. In priority order:

1. **Exposed NVLS comm (~16%, and it gates the critical path).** vLLM's
   equivalent has *zero* exposed comm. Overlap dispatch/combine with the expert
   GEMM via chunked/pipelined experts, or fuse the combine ReduceScatter-V into
   the FC2 epilogue + `_moe_sum`. This is now the single highest-value target.
2. **The routing chain (~18%).** After QWEN-011 the align is 3 kernels; the
   residual `_count_local_tokens` (8.2 µs) + `_moe_sum` (8.6 µs) + scatter
   (3.1 µs) is ~20 µs of the 84 µs MoE call. Fusing `_moe_sum` into the FC2
   epilogue is the contained piece.
3. **Only then** revisit a fully fused MoE (QWEN-015's recipe), whose remaining
   upside is the routing/finalize overhead, not the GEMM.

## Session 5 (2026-07-25) — exposed NVLS EP comm

### QWEN-016 — EP comm decision gate: exposed, latency-bound, small prize

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | The 929 µs/step of exposed NVLS EP comm is a worthwhile lever. Gate it first: measure how much is genuinely exposed, whether it is latency- or bandwidth-bound, and what the floor is under perfect overlap. |
| Code revision | branch `perf/moe-fused-align`, dirty. **No Megatron change retained** — analysis only. |
| Changed files | `dev/moe_fused/probe_comm.py`, `dev/moe_fused/analyze_comm.py`, `dev/moe_fused/analyze_comm_skew.py`, `dev/moe_fused/harness_comm.py` (all new, analysis harnesses; nothing under `megatron/`) |
| Runtime flags | trace side: none (re-analysis of PROFILE-TUNED). Microbench: `hidden=2048 topk=8 local_tokens=64 per_rank_max=2048 rsv_dtype=float32`, CTA sweep `4,8,16,32,64,128` |
| Image | cog dev image for `oci-hsg`, venv `envs/megatron_lm/dd356431262b5db4` (sqsh tag not re-verified this session) |
| Checkpoint / tokenizer | n/a (trace re-analysis + synthetic-payload microbench at production shapes) |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), EP4/TP1 |
| Workload | trace: PROFILE-TUNED steady-state decode, stream 257, steps 32–96, 48 layers, 4 ranks. Microbench: NVLS AGV/RSV at the production decode shape under CUDA-graph replay, median of repeats |
| Job / run | session `qwen-comm`, Slurm job 5601961 (`batch_long`, 8 h); exec `08e6a692b5164fa1929086fcc0e315d1` (CTA sweep), `04bdfdd605aa49348ff3d031aba70c89` (decomposition) |
| Throughput | not applicable (no Megatron change measured) |
| Latency / TPOT | not applicable |
| Correctness | microbench gate: AGV max\|diff\| 0.000e+00 and RSV max rel err 0.000e+00 over 64 trials vs `all_gather_into_tensor` / analytic rank-scaled reduction |
| Nsight artifacts | source trace `qwen-cutlass2:prof/tuned-1785013555/mcore_profile.sqlite` |
| Result | **Gate answered on all three questions; the prize is smaller than its 11.7% share suggests.** (1) *Exposed*: the per-step union of comm intervals equals their sum — there is no concurrent compute, so it is 100% exposed. Per-step comm totals 928–1219 µs across the four ranks. (2) *Latency-bound, decisively*. Decomposing the production-shape collectives at 128 CTAs: AGV 6.57 µs = launch 0.72 + **barrier 5.08** + transfer 0.77; RSV 7.88 µs = launch 0.72 + **barrier 5.03** + transfer 2.13. Payloads are only 268 KB/rank (AGV) and 512 KB/rank (RSV); against the 900 GB/s/dir NVLink floor (0.894 / 1.748 µs) the RSV transfer is already at 82% of peak and the AGV transfer is *below* the unicast floor because multimem multicast pays one egress for all peers. Byte movement is therefore 127 µs/step out of ~693 µs — **fusing or batching bytes cannot win; only removing or hiding barriers can.** (3) *Floor*. Splitting the trace into intrinsic cost (median release − last arrival) and inter-rank arrival skew: AGV 6.78 µs/kernel = 5.97 intrinsic + 0.76 skew; RSV 15.30 = 7.19 + 8.11. Per step: 1060 µs total = **632 µs intrinsic + 428 µs skew**. The skew is not comm work — it is ranks waiting for the slowest rank's expert GEMM (ranks differ in how many experts receive tokens), so it is a routing-balance problem, not a collective problem. The skew-free microbench independently lands at 693 µs/step, corroborating the 632 µs figure. So the recoverable critical path is **632–693 µs of a 9,933 µs step = 6.4–7.0%**; perfect elimination gives 1.068–1.075× → 25,250–25,420 tok/s = 74.3–74.8% of vLLM 33,994.5. Of that, ~485 µs is 96 symmetric-memory barriers × 5.05 µs. Sub-verdict on the two proposed approaches: **(b) fusing RSV into the FC2 epilogue** removes a launch and the transfer but *keeps the barrier*, ceiling ≈ 2.9 µs/layer = 137 µs/step = **1.4%**. **(a) chunked/pipelined experts** hides a collective behind GEMM but adds one barrier per extra chunk (5.05 µs) against a 14.4 µs/layer collective cost, so a 2-chunk split is worth at best ≈ 9 µs/layer = 430 µs/step = **4.3%**, and only if CUDA-graph capture can express concurrent streams across the MoE dependency chain — which QWEN-008 already identified as the binding constraint. Separately, the CTA count is **already optimal**: the sweep gives AGV 13.54/9.68/8.45/7.89/8.07/6.78 µs and RSV 37.37/21.12/13.43/9.31/8.25/8.05 µs at 4/8/16/32/64/128 CTAs, so the shipped `MAX_NUM_BLOCKS=128` is the best point and reducing CTAs to shrink the per-CTA barrier is strictly worse. A prepared `MCORE_NVLS_{AGV,RSV}_CTAS` override was therefore reverted unused. |
| Next action | **Redirect off this lever.** A 6.4–7.0% ceiling of which 1.4–4.3% is realistically reachable, inside graph-captured dispatcher code, is worse value than routing/permute (1251 µs/step, 15.8%, spread over 242 kernels at 5.2 µs each — a launch-count problem where fusion converts directly into wall time). Take the ledger's item 2: fuse `_moe_sum` into the FC2 epilogue and collapse the residual `_count_local_tokens` / scatter chain. |

### QWEN-017 — load-poll symmetric-memory barrier (rejected)

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | `symm_mem_sync`'s wait loop spins on a system-scope `atom.cas`, one full uncached read-modify-write per attempt, which sets how quickly a rank notices its peers arrived. Replacing it with an `ld.acquire.sys` poll plus a single clearing store should cut the 5.05 µs barrier and thus ~485 µs/step. |
| Code revision | branch `perf/moe-fused-align`, dirty. Change **reverted** after measurement. |
| Changed files | `megatron/core/inference/communication/torch_symm_triton/barrier.py` (added `_wait_signal_ldpoll` + `MCORE_SYMM_BARRIER_LDPOLL` gate, default off) — reverted; `dev/moe_fused/harness_comm.py` (correctness gate) — retained |
| Runtime flags | `MCORE_SYMM_BARRIER_LDPOLL=0` vs `=1`, production shape, 128 CTAs, 3 alternating reps each |
| Image | cog dev image for `oci-hsg`, venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | n/a (collective microbench at production shapes) |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), EP4/TP1 |
| Workload | NVLS AGV + RSV under CUDA-graph replay, median of repeats, 64-trial correctness gate before each timing block |
| Job / run | session `qwen-comm`, job 5601961; exec `04bdfdd605aa49348ff3d031aba70c89` |
| Throughput | not run e2e — rejected at the microbench stage |
| Latency / TPOT | per-step comm (48 layers): baseline 695.7 / 692.2 / 693.3 µs (mean 693.7); ldpoll 710.0 / 710.2 / 709.1 µs (mean 709.8). **+2.3% regression**, no overlap between the two triplets |
| Correctness | bit-exact in both variants: AGV max\|diff\| 0.000e+00, RSV max rel err 0.000e+00 over 64 trials |
| Nsight artifacts | none (microbench uses CUDA events) |
| Result | **Rejected — measured regression.** The barrier-only kernel is unchanged between variants (5.75–5.83 µs at both 256 and 512 threads/CTA, against a 0.72 µs empty kernel), so the ~5.05 µs is the 4-way system-scope flag round trip itself, not polling granularity. The cheaper poll actually costs slightly more, consistently on RSV (7.88 → 8.19 µs). Combined with the CTA-count result in QWEN-016, the barrier looks like a hardware/driver latency floor that a Triton-level rewrite does not move. |
| Next action | Do not pursue further barrier micro-optimisation. Any future attempt on this lever must remove barriers (fewer collectives) or hide them behind compute, not make each one cheaper. |

## Session 6 (2026-07-25) — the routing/permute chain

### QWEN-018 — routing/permute decision gate: per-kernel breakdown and fusion ceilings

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | Routing/permute (1251 µs/step, 15.8%, 242 kernels at 5.2 µs) is a launch-count problem, so removing kernels converts directly into wall time. Gate it first: attribute the 1251 µs to individual kernel names, split each into fixed launch cost vs real work, measure the inter-kernel dispatch gap, and compute a wall-time ceiling per candidate fusion. Reject any candidate below ~1%. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty. **No Megatron change** — analysis only. |
| Changed files | `dev/moe_fused/analyze_routing.py` (new, analysis only; nothing under `megatron/`) |
| Runtime flags | n/a (re-analysis of the PROFILE-TUNED trace, which was captured with `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1`) |
| Image | n/a (login-node `python3` 3.12.13 + stdlib `sqlite3`) |
| Checkpoint / tokenizer | n/a |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, EP4/TP1 (trace); analysis on the login node |
| Workload | PROFILE-TUNED decode window 192.7–195.5 s, device 3, stream 257; 40 steady-state steps averaged, one MoE layer dumped in launch order |
| Job / run | session `qwen-comm`, Slurm job 5601961 (allocation held, analysis ran on the login node); source trace `qwen-cutlass2:prof/tuned-1785013555/mcore_profile.sqlite` |
| Throughput | not applicable (analysis) |
| Latency / TPOT | step wall 9927.2 µs, GPU-busy 7900.0 µs, idle 2027.2 µs (20.4%), 1362 kernels — reproduces PROFILE-TUNED (9.933 / 7.884 / 2.049 ms, 1362) from an independent script |
| Correctness | n/a (no functional change) |
| Nsight artifacts | `qwen-cutlass2:prof/tuned-1785013555/mcore_profile.sqlite`; `dev/moe_fused/analyze_routing.py` output |
| Result | **The routing category is not one problem, it is two pathological kernels plus four cheap ones — and it is not launch-bound.** Per-step, per-kernel (48 launches each unless noted; `WALL` = device time + the dispatch gap that follows): `_moe_sum_kernel` 7.79 µs/k → **400.3 µs**; `_count_local_tokens_kernel_persistent` 7.52 → **387.4**; `gatherTopK` (router top-8) 6.05 → **316.9**; `_scatter_token_indices_kernel` 2.66 → **153.8**; `triton_per_fused__softmax_prep` (router softmax) 1.94 → **119.9**; `_prefix_fill_init_kernel` 1.33 → **90.9**; plus the `torch.zeros` fill of the count buffer (`vectorized_elementwise_kernel`, grid=1) 0.74 → **≈62**. Total **≈1531 µs/step**. Three measurements set the ceilings. (a) *The dispatch gap is 0.55 µs*, uniform across kernels in the graph, so a removed launch is worth `duration + 0.55 µs` — more than the kernel time alone, but far less than the 5.2 µs/kernel average would suggest if you assumed the whole thing were overhead. (b) *The fixed floor per launch is 1.27 µs* (0.72 µs empty kernel from QWEN-016 + 0.55 µs gap), so of the 1531 µs only **366 µs is fixed cost** and **1165 µs is in-kernel work** — pure launch-count reduction can win at most 3.7% and only by removing *every* routing launch. (c) *The 20.4% idle is now explained*: 862 µs of it is two once-per-step host gaps (537 µs after `index_elementwise_kernel`, 325 µs after `CatArrayBatchedCopy_vectorized`, i.e. sampling/detokenize between graph replays) and ~750 µs is 1362 × 0.55 µs of intra-graph node dispatch; there is no large unexplained residue. **Candidate ceilings.** *(1) Fuse `_moe_sum` into the FC2 epilogue* (the ledger's first-ranked): removes 48 × (7.79 + 0.55) = **400 µs = 4.03% gross**. But the topk slots of one token land in different experts' blocks and therefore different M-tiles, so accumulating into `out` needs cross-CTA fp32 atomics (~4 MB/layer of RMW: 512 local pairs × 2048 × 4 B) *and* a zeroing pass over `out[0:valid_tokens]` that today does not exist because `_moe_sum` writes rather than accumulates. That gives back ~70 µs, leaving ~3.3%, it is the exact shape QWEN-001 measured at 0.68–0.80×, and it stops being bit-exact. Held as the fallback, not built first. *(2) Cooperative-grid merge of count → prefix_fill_init → scatter*: the three are serially dependent, so a merge needs a grid-wide sync and keeps all the work; it recovers only the fixed cost, 2 × 48 × 1.27 = **122 µs = 1.23%**, before paying two `grid.sync()`s per layer against a kernel boundary that costs just 0.55 µs — and it needs cooperative launch under CUDA-graph capture. **Gated out on arithmetic.** *(3) New, and the one the breakdown actually points at: fold the token count and its `torch.zeros` fill into `_prefix_fill_init_kernel`.* `_prefix_fill_init_kernel` already has every CTA redundantly recompute the whole 32-wide cumsum in registers, and the count vector is its only input — so recomputing the histogram per CTA needs **no grid sync at all**. Removes 48 × (7.52 + 0.55) + 48 × (0.74 + 0.55) = **449 µs = 4.52% gross**, integer-exact. **Why `_count_local_tokens` costs 7.52 µs to bucket 2048 int32s**: it is neither launch-bound (0.72 µs floor) nor bandwidth-bound (8 KB), it is starved — with `BLOCK_SIZE=1024` and 2048 valid pairs `total_blocks=2`, so exactly **2 of its 152 CTAs receive work**, and each issues 1024 global atomics contending on 32 counters. **This corrects QWEN-003's conclusion.** QWEN-003 replaced those atomics with `tl.histogram` but left `BLOCK_SIZE=1024` untouched, so both variants ran on 2 CTAs; it measured a wash and inferred "the cost is per-launch fixed overhead, not atomic contention, so an in-kernel rewrite cannot help". The device-time measurement says the opposite: 6.8 of the 7.52 µs is in-kernel. |
| Next action | Build candidate (3) behind `MCORE_MOE_FUSED_COUNT` (default off, requires `MCORE_MOE_FUSED_ALIGN`) → QWEN-019. Two levers the gate surfaced but did not pursue, in order: the router pair `gatherTopK` + softmax is **436.8 µs/step = 4.4%** in two kernels (a hand-written fused top-8 router honouring the `inference_optimized` dense top-8 contract; QWEN-009 only ruled out TE's *built-in* fusion, not a hand-written one), and `_moe_sum` at 7.79 µs moves ~6.3 MB, i.e. ~1.05 µs at the measured 6.08 TB/s — it is **7× off its own bandwidth floor**, so restructuring it in place (hoist the 16 per-token scalar `routing_map` loads out of the K loop, parallelise over K as well as tokens) is worth up to 48 × 5.8 = 278 µs = 2.8% *without* the atomics that fusing into FC2 would require. |

### CLEANBASE-S6 — fresh same-session reference

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | Establish a same-session OSL1024 reference at the session-4/5 best config before any session-6 A/B. Session 5 skipped this and session-to-session drift is ~1.5%. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (carries QWEN-002 + QWEN-011 + QWEN-013/013b) |
| Changed files | none |
| Runtime flags | `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1`; nvls dispatcher, vLLM grouped-GEMM backend, `transformer_impl=inference_optimized`, full-iteration CUDA graphs, `CUDA_DEVICE_MAX_CONNECTIONS=1` |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh` (`nvcr.io/nvidia/pytorch:26.06-py3`), venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` under the user checkpoint root |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters |
| Job / run | session `qwen-comm`, Slurm job 5601961; exec `023b75121915449badd4b39ca610f470`; run dir `sessions/qwen-comm/e2e/ref-s6-1785024868` |
| Throughput | **23,264.4 tok/s** (per-iter 23,019.4 / 23,402.0 / 23,189.7 / 23,457.4 / 23,258.9) |
| Latency / TPOT | avg_latency 10,974.7 ms, p50 10,972.5, p99 11,346.9; TPOT 11.004 ms/tok |
| Correctness | Benchmark completed 5/5 iters at 256 requests |
| Nsight artifacts | none (un-profiled throughput run) |
| Result | Reference for session 6 = 68.44% of vLLM 33,994.5. −1.61% vs QWEN-013b's 23,646.0, consistent with the documented ~1.5% session drift; all A/Bs this session are against this number, not against 23,646.0. |
| Next action | Verified the synced snapshot (`workspaces/megatron_lm/6b9355b187072223`) contains no session-6 kernel, so this is exactly the accepted session-5 code. Proceed to QWEN-019. |

### QWEN-019 — fold the token count into the indirection-table build

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | QWEN-018's chosen candidate. The decode indirection-table build spends 4 launches producing a 32-element count vector and then consuming it: a `torch.zeros` fill (0.74 µs), `_count_local_tokens_kernel_persistent` (7.52 µs), `_prefix_fill_init_kernel` (1.33 µs), `_scatter_token_indices_kernel` (2.66 µs). The count's only consumer already has **every CTA redundantly recompute the whole cumsum in registers**, so recomputing the histogram per CTA as well removes the first two launches with no grid sync and no change to the counts. Ceiling 48 × (7.52 + 0.55 + 0.74 + 0.55) = 449 µs = 4.52% of the step. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (`_count_prefix_fill_init_kernel`, `_moe_align_block_size_count_fused`, `_USE_FUSED_COUNT` / `MCORE_MOE_FUSED_COUNT` gate, `_FUSED_COUNT_MAX_TOKENS`, three-way `align_fn` selection), `dev/moe_fused/harness_countfuse.py` (new A/B harness) |
| Runtime flags | `MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`; otherwise identical to CLEANBASE-S6. The fused-count path is restricted to `num_tokens_hint <= 512`; above that the redundant per-CTA read stops being free and the atomic count kernel is used. |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh`, venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters (throughput); microbench 300 CUDA-graph replays × 3 repeats at 128/256/384/512 tokens (timing) and 16 table-equality cases (correctness) |
| Job / run | session `qwen-comm`, job 5601961; microbench exec `d9f7f74a868241d69eb4cc1f4062d40b`; e2e exec `3a807f77358644c89e62866749519802`, run dir `sessions/qwen-comm/e2e/countfuse-1785025539` |
| Throughput | **23,964.4 tok/s** vs **23,264.4** same-session reference → **+3.01%**. Per-iter 23,566.6–24,304.6 |
| Latency / TPOT | avg_latency 10,642.7 ms vs 10,974.7 (−3.03%); TPOT 10.682 vs 11.004 ms/tok (−2.93%). CUDA-graph replay device time at 256 tokens: align call **16.38 → 10.24 µs (1.600×)**, whole `vllm_fused_moe` call **86.02 → 79.96 µs (1.076×)**, i.e. 6.06 µs/layer × 48 = 291 µs/step = 2.93% of the 9.93 ms step — the e2e result matches the microbench prediction to within 0.1 pp. Per-kernel (eager profiler, 100 iters): count 4.76 + zeros 1.20 + prefix_fill 1.76 = 7.72 µs replaced by a single 3.73 µs kernel; align total device time 10.17 → 6.11 µs |
| Correctness | **Bit-exact, and verified two ways.** (1) Table equality over 16 cases (valid ∈ {128,256,384,512} × BLOCK_M ∈ {16,64} × local_expert_start ∈ {0,32}): `num_tokens_post_padded` identical, `expert_ids` identical, and the sorted multiset of `sorted_token_ids[0:npp]` identical (sorted because the scatter's atomics permute rows within an expert block on both paths). (2) Whole-MoE output at all four token counts: max_abs 0.0 **and max_rel 0.0** — per QWEN-014's lesson the relative error is checked, not a loose `allclose`. e2e coherence prompts coherent (2+2=4, Paris, 3+2 cows = 5 cows) |
| Nsight artifacts | none (CUDA-graph event timing + torch profiler; the change is a launch-count/kernel-time change fully explained by the two removed launches) |
| Result | **Accepted** — bit-exact, +3.01% e2e with tight variance, no regression. Default OFF so untouched paths stay byte-identical. New best mcore = **23,964.4 tok/s = 70.49%** of the vLLM 33,994.5 baseline. Realised 291 of the 449 µs ceiling; the shortfall is the new kernel's own cost, 1.33 → 3.73 µs (eager) as each of the 32 CTAs now histograms all 2048 valid pairs instead of reading a 32-element vector. **The mechanism confirms QWEN-018's correction of QWEN-003**: the count kernel's cost was in-kernel, not per-launch, and the fix was to stop running it on 2 of 152 CTAs — not to change how it reduces. |
| Next action | The same breakdown ranks `_moe_sum` next: 7.79 µs/layer to move ~6.3 MB is 7× its own bandwidth floor, and it can be attacked without the cross-CTA atomics that fusing it into FC2 would need → QWEN-020. After that, the router pair (`gatherTopK` + softmax, 436.8 µs/step = 4.4% in two kernels) is the largest remaining routing item. |

### QWEN-020 — predicate the locality test in the topk reduction

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | QWEN-018 measured `_moe_sum_kernel` at 7.79 µs/layer to move ~6.3 MB — 7× the ~1.05 µs its own traffic implies at the 6.08 TB/s of QWEN-012 — so it is neither launch- nor bandwidth-bound. The suspected cause is the `if lid >= 0 and lid < num_local_experts` guard: a uniform scalar branch gated on a dependent global load of `routing_map`, which prevents slot `t`'s data load from overlapping slot `t+1`'s index load and makes the CTA walk the 8 topk slots serially. Predicating the guard into the load mask, and widening `BLOCK_K` from 1024 to the full hidden size so the per-token index loads are issued once instead of `NUM_K_BLOCKS` times, should recover most of that gap **while staying bit-exact** — the reduction order and the fp32 arithmetic are unchanged and masked-off slots contribute an exact 0.0. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (carries QWEN-002 + QWEN-011 + QWEN-013/013b + QWEN-019) |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (`_moe_sum_kernel_fast`, `_USE_FAST_MOE_SUM` / `MCORE_MOE_SUM_FAST` gate, `_FAST_MOE_SUM_MAX_BLOCK_K = 2048`, dispatch in `_moe_sum`), `dev/moe_fused/harness_moesum.py` (new A/B harness) |
| Runtime flags | `MCORE_MOE_SUM_FAST=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`; otherwise identical to CLEANBASE-S6. One new variable vs QWEN-019. |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh` (`nvcr.io/nvidia/pytorch:26.06-py3`), venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters (throughput); microbench 300 CUDA-graph replays × 3 repeats at 128/256/384/512 valid tokens, plus whole-MoE output equality at the same four shapes |
| Job / run | session `qwen-comm`, job 5601961; microbench exec `af464a419c284937a106aae25196359a`; e2e exec `af49c9f54f4a497d8808f50614fe6bf6`, run dir `sessions/qwen-comm/e2e/moesum-1785026515` |
| Throughput | **24,403.7 tok/s** vs **23,964.4** (QWEN-019) → **+1.83%**; vs **23,264.4** (CLEANBASE-S6) → **+4.90%**. Per-iter 24,442.9 / 24,137.8 / 24,580.4 / 24,607.1 / 24,257.1 |
| Latency / TPOT | avg_latency 10,445.6 ms (p50 10,447.9, p99 10,806.4); TPOT **10.490** vs 10.682 ms/tok. Microbench, CUDA-graph replay of the whole `vllm_fused_moe` call: 128 tok 73.10 → 71.59 µs (1.021×), 256 tok 80.69 → 78.77 (1.024×), 384 tok 86.81 → 83.94 (1.034×), 512 tok 102.68 → 98.56 (1.042×). Eager per-kernel at 256 tokens (100 iters): `_moe_sum_kernel` **8.14 → 5.97 µs (1.36×)**, every other kernel in the call unchanged (`_fused_moe_kernel` 62.18 → 61.88, `_count_prefix_fill_init` 4.18 → 4.18, `_scatter_token_indices` 3.07 → 3.03) |
| Correctness | **Bit-exact.** Whole-MoE output at valid ∈ {128, 256, 384, 512}: max_abs 0.0 **and max_rel 0.0** (QWEN-014's lesson — relative error, not a loose `allclose`). e2e coherence prompts coherent (2+2 = 4, Paris, 3+2 cows = 5 cows). Benchmark completed 5/5 iters |
| Nsight artifacts | none (CUDA-graph event timing + torch profiler; the change alters one kernel's internals and no launch counts, which the per-kernel table isolates directly) |
| Result | **Accepted** — bit-exact and positive on every shape measured. New best mcore = **24,403.7 tok/s = 71.79%** of the vLLM 33,994.5 baseline. Default OFF; enable with `MCORE_MOE_SUM_FAST=1`. **Honest accounting of the size of the win:** the microbench says 2.17 µs/layer × 48 = **104 µs/step ≈ 1.0%** of the step, and the graph-replay whole-call delta at 256 tokens is smaller still (1.92 µs × 48 = 92 µs), whereas the e2e TPOT moved 0.192 ms/step (192 µs). The e2e gain is therefore about **2× the microbench prediction**, and QWEN-019's per-iteration range (23,566–24,305) overlaps this run's (24,138–24,607), so part of the measured +1.83% is run-to-run variance rather than kernel time. The change is kept because it is bit-exact and every isolated measurement of it is a strict improvement, but the defensible attribution is ~1%, not 1.83%. QWEN-018's estimated ceiling for restructuring this kernel in place was 278 µs (2.8%); the predication recovered 104 µs of it, so `_moe_sum` at 5.97 µs is still ~6× its bandwidth floor and the remaining gap is the strided `[token, slot, K]` gather itself, not the branch |
| Next action | The router pair is now the largest routing item the QWEN-018 breakdown left standing: `gatherTopK` 6.05 µs + softmax 1.94 µs = 436.8 µs/step = 4.4% in two kernels, for what is only a `[256, 128]` fp32 softmax and a top-8 select → QWEN-021. |

### QWEN-021 — fused softmax + top-8 router selection

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | QWEN-018 left the router pair as the largest untouched routing item: `gatherTopK` 6.05 µs + the compiled softmax 1.94 µs = 436.8 µs/step (4.4%) — for nothing more than a `[256, 128]` fp32 softmax and a top-8 select over 128 experts. `torch.topk` runs a multi-pass radix select sized for large `n`; at 128 candidates per row, one CTA per token can hold the whole row in registers, softmax it, and pick the top 8 by 8 max-then-mask passes. Ceiling: replace ~8.0 µs + 0.55 µs of dispatch gap per layer with a single ~2 µs kernel ⇒ ~312 µs/step ≈ 3.1%. **This is not QWEN-009**, which ruled out TE's *built-in* fused router because it emits a dense 128-expert map while `inference_optimized` needs the dense top-8 contract; this kernel is written to the top-8 contract directly. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (carries QWEN-002 + QWEN-011 + QWEN-013/013b + QWEN-019 + QWEN-020) |
| Changed files | `megatron/core/inference/moe/router_topk.py` (new: `_softmax_topk_kernel`, `fused_softmax_topk`, `can_use_fused_softmax_topk`, `MCORE_ROUTER_FUSED_TOPK` gate, `FUSED_ROUTER_TOPK_MAX_TOKENS = 1024`), `megatron/core/transformer/moe/router.py` (`InferenceTopKRouter._forward` takes the fused path when the contract matches), `dev/moe_fused/harness_routertopk.py` (new A/B harness) |
| Runtime flags | `MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_SUM_FAST=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`; otherwise identical to CLEANBASE-S6. One new variable vs QWEN-020. The fused path is taken only for softmax + pre-softmax + no groups + no scaling factor + no expert bias + no router replay + ≤1024 tokens; anything else (including prefill chunks above 1024 tokens) falls back to the compiled torch path unchanged. |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh` (`nvcr.io/nvidia/pytorch:26.06-py3`), venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters (throughput); microbench 500 CUDA-graph replays × 3 repeats at 128/256/384/512 tokens × 128 experts, correctness at those four shapes × 2 seeds |
| Job / run | session `qwen-comm`, job 5601961; microbench exec `56ae1d844a86427ea60ddce2c73d903c`; e2e exec `2041011b410b45709d455514b32c85f3`, run dir `sessions/qwen-comm/e2e/routertopk-1785027029` |
| Throughput | **25,352.9 tok/s** vs **24,403.7** (QWEN-020) → **+3.89%**; vs **23,264.4** (CLEANBASE-S6) → **+8.98%**. Per-iter 25,012.5 / 25,092.4 / 25,405.9 / 25,625.0 / 25,642.4 — the whole range sits above QWEN-020's whole range (24,137.8–24,607.1), so unlike QWEN-020 this win is outside run-to-run variance |
| Latency / TPOT | avg_latency 10,041.4 ms (p50 10,039.1, p99 10,439.5); TPOT **10.097** vs 10.490 ms/tok → **−393 µs/step** against a predicted ~312 µs. CUDA-graph replay of the router selection alone: 128 tok 16.40 → 4.10 µs, 256 tok **16.41 → 4.10 µs (4.00×)**, 384 tok 18.43 → 4.10 (4.49×), 512 tok 20.47 → 4.11 (4.99×) — the fused kernel is flat in token count over this range while the torch pair is not. Eager per-kernel at 256 tokens: `gatherTopK` 9.12 + `unrolled_elementwise` 2.76 + `softmax_warp_forward` 2.09 + `vectorized_elementwise` 1.57 = **15.54 µs in 4 kernels → 2.04 µs in 1** |
| Correctness | **Bit-exact probabilities and identical expert sets** at 128/256/384/512 tokens × 2 seeds each: the selected expert id multiset per token is equal, max_abs 0.0 and max_rel 0.0 on the probabilities after aligning by expert id, and no duplicate expert ids. (`torch.topk` is called with `sorted=False` during inference, so the *order* of the k results is unspecified on both sides; the fused kernel returns descending score order with ties broken toward the lower expert id.) e2e coherence prompts coherent (2+2 = 4, Paris, 3+2 cows = 5 cows); benchmark 5/5 iters |
| Nsight artifacts | none (CUDA-graph event timing + torch profiler; a 4→1 launch change with per-kernel attribution on both sides) |
| Result | **Accepted — the largest single win of session 6.** New best mcore = **25,352.9 tok/s = 74.58%** of the vLLM 33,994.5 baseline. Default OFF; enable with `MCORE_ROUTER_FUSED_TOPK=1`. The measured e2e gain (393 µs/step) slightly exceeds the 312 µs predicted from QWEN-018's two-kernel accounting because the profile's "router" attribution missed the two elementwise kernels (the `.type_as` cast and one more) that the fused kernel also absorbs — 4 launches removed, not 2. Session total so far: 23,264.4 → 25,352.9 = **+8.98%**, all of it bit-exact. |
| Next action | The last item in the QWEN-018 breakdown that clears the gate is `_scatter_token_indices_kernel` at 2.66 µs + 0.55 µs gap × 48 = 154 µs = 1.6%. Now that QWEN-019 made every CTA stream the pairs anyway, the scatter can be a second streaming pass inside the same kernel — no atomics, no grid sync — taking the indirection-table build to a single launch → QWEN-022. |

### QWEN-022 — one-launch indirection-table build (scatter folded in)

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | The last routing item above the gate is `_scatter_token_indices_kernel`: 2.66 µs + 0.55 µs dispatch gap × 48 = 154 µs/step ≈ 1.6%. QWEN-019 already has CTA `e` stream every valid pair to build a private histogram, so a second streaming pass over the same (L2-resident, 16 KB) pairs lets that CTA place its own rows itself at `excl_e + written + exclusive_cumsum(is_mine)` — no global atomics, no grid sync, and a deterministic table instead of an atomically permuted one. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (carries QWEN-002 + QWEN-011 + QWEN-013/013b + QWEN-019 + QWEN-020 + QWEN-021) |
| Changed files | `megatron/core/inference/moe/vllm_fused_moe.py` (`_align_single_kernel`, `_moe_align_block_size_single`, `_USE_FUSED_SCATTER` / `MCORE_MOE_FUSED_SCATTER` gate, align dispatch), `dev/moe_fused/harness_scatterfuse.py` (new A/B harness) |
| Runtime flags | `MCORE_MOE_FUSED_SCATTER=1` on top of the QWEN-021 set (`MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_SUM_FAST=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_FUSE_FC1_ACT=1`). One new variable. |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh`, venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed iters, **run twice**; microbench 300 replays × 3 repeats at 128/256/384/512 tokens, 16 table-equality cases |
| Job / run | session `qwen-comm`, job 5601961; microbench execs `c7cfd653ca6f4bc1849649746fb1cb48` (first, with a faulty check) and `46060ef1233f407fb929ab82755517d4` (corrected); e2e execs `a31dd240c22d47588ffa58e264180f32` (`e2e/scatterfuse-1785027598`) and `a5557671044f45709e0a933e7b255741` (`e2e/scatterfuse-rep2-1785027955`) |
| Throughput | run 1 **25,441.4 tok/s** (+0.35% vs QWEN-021's 25,352.9), run 2 **25,495.9 tok/s** (+0.56%); mean **25,468.7 = +0.46%**. Per-iter run 1 24,920.5–25,791.6, run 2 24,870.8–25,827.4 — both ranges overlap QWEN-021's (25,012.5–25,642.4) |
| Latency / TPOT | TPOT 10.062 and 10.041 ms/tok vs 10.097 (−35 and −56 µs/step) against a microbench prediction of 125 µs/step. CUDA-graph replay at 256 tokens: align call **10.08 → 8.23 µs (1.224×)**, whole MoE call **82.17 → 79.56 µs (1.033×)**. At other shapes: 128 tok 1.045×, 384 tok 1.007×, **512 tok 0.997× (a slight loss)** — the extra streaming pass scales with pair count while the removed launch does not. Eager per-kernel: `_count_prefix_fill_init` 3.74 + `_scatter_token_indices` 2.38 = 6.12 µs in 2 kernels → `_align_single_kernel` 4.91 µs in 1 |
| Correctness | **Bit-exact whole-MoE output** at 128/256/384/512 (max_abs 0.0 and max_rel 0.0), plus table equality over 16 cases (valid × BLOCK_M ∈ {16,64} × local_expert_start ∈ {0,32}): identical `num_tokens_post_padded`, identical `expert_ids`, identical row multiset, and — the stronger check — every non-sentinel row sits in a block whose expert id actually owns that pair. e2e coherence prompts coherent; both benchmark runs 5/5 iters. **Note on the first harness run:** it reported a table MISMATCH at BLOCK_M=16, which was a bug in the *check*, not the kernel — an expert's rows span several BLOCK_M blocks and the two paths order them differently across that range, so comparing per 16-row block flags a difference that does not exist. The whole-MoE output was bit-exact in that same run, which is what exposed the faulty check. |
| Nsight artifacts | none (CUDA-graph event timing + torch profiler) |
| Result | **Accepted, marginally, and recorded as a weak win rather than a 1.2% one.** Both e2e runs beat QWEN-021, so the sign is reliable; the size (+0.46%) is a third of the microbench prediction and sits inside the per-iteration spread, so the defensible claim is "small but positive". Kept because it is bit-exact, removes a launch, and makes the table deterministic. Default OFF. Best mcore = **25,495.9 tok/s = 75.00%** of the vLLM 33,994.5 baseline (mean of the two runs 25,468.7 = 74.92%). **This is the point where the routing lever runs out**: every routing kernel QWEN-018 listed has now been fused or rewritten except `_moe_sum`, whose remaining gap needs the FC2-epilogue fusion QWEN-018 gated as ~3.3% net and QWEN-001-shaped. |
| Next action | Re-profile at the new configuration before choosing the next lever — the step has lost ~10% of its wall time and roughly 700 µs of routing, so the QWEN-018 ranking is stale → PROFILE-S6. |

### PROFILE-S6 — re-profile at the session-6 configuration

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | The QWEN-018 ranking is stale: the step has lost ~10% of its wall time and roughly 820 µs of routing since it was taken. Re-profile with all six gates on and re-rank before choosing the next lever. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (QWEN-002 + QWEN-011 + QWEN-013/013b + QWEN-019 + QWEN-020 + QWEN-021 + QWEN-022) |
| Changed files | none (`dev/moe_fused/profile_insession.sh`) |
| Runtime flags | `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1` |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh`, venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200 (`nvl72086-T04`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, **OSL128** under nsys (`--trace=cuda,nvtx --cuda-graph-trace=node`), 1 iter after a BS8 warmup; analysis window device 1, 184.9–186.9 s (~220 steady-state decode steps) |
| Job / run | session `qwen-comm`, job 5601961; exec `1a4e8dc90c744a1c9ae0d977ec5ad16e`; run dir `sessions/qwen-comm/prof/s6-all-1785028240` |
| Throughput | 10,454.6 tok/s at OSL128 under nsys (profiling overhead + short OSL; not comparable to the OSL1024 ledger numbers) |
| Latency / TPOT | one steady-state decode step: **wall 9.097 ms** (PROFILE-TUNED: 9.933), **GPU-busy 7.183 ms** (7.884), **idle 1.914 ms = 21.0%** (2.049 ms = 20.6%), **1170 kernels/step** (1362) |
| Correctness | n/a (profile capture) |
| Nsight artifacts | `qwen-comm:prof/s6-all-1785028240/mcore_profile.nsys-rep` and `.sqlite`; `forward_pass.py` and `dev/moe_fused/analyze_routing.py --device 1 --window 184.9,186.9` output |
| Result | **The routing lever is spent.** Per forward pass: MoE expert GEMM 2469.1 µs / 96 kernels (27.1%), dense GEMM 1083.6 / 241 (11.9%), comm 1062.9 / 96 (11.7%), attention 923.8 / 96 (10.2%), elementwise 487.6 / 253 (5.4%), norm 449.5 / 193 (4.9%), **MoE routing/permute 430.4 µs / 98 kernels (4.7%)** — down from 1251 µs / 242 kernels, a 66% cut, and it is now the smallest GPU category except sampling. Counting dispatch gaps, the routing family (including `_align_single_kernel`, which the categorizer files under "other") is ~665 µs/step. Its two survivors are `_moe_sum_kernel_fast` (5.57 µs × 48 = 294 µs with gaps) and `_align_single_kernel` (5.13 × 48 = 272 µs); `_softmax_topk_kernel` costs 1.53 µs × 48 = 99 µs, versus 436.8 µs for the pair it replaced. **The idle is now the ranked next lever and it is host-side, not GPU-side**: of 1914 µs, ~1237 µs sits in three once-per-step gaps — **548 µs** after `index_elementwise_kernel`, **386 µs** after `vectorized_elementwise_kernel`, **303 µs** after `CatArrayBatchedCopy_vectorized` — i.e. sampling, detokenize and scheduling on the host between graph replays. The remainder (~644 µs) is 1170 × 0.55 µs of intra-graph node dispatch, which only fewer kernels can shrink. |
| Next action | Attack the ~1237 µs/step of host gaps, but **profile the host first** — this is not the same experiment as QWEN-005a/QWEN-007 (`async-sched-mode=serial`, which crashed and then measured only +0.85%) or QWEN-004 (`sampling-backend=flashinfer`, incompatible with the full-iteration graph). Capture with `--trace=cuda,nvtx,osrt` plus Python sampling over the decode window and attribute the three gaps to concrete host functions before proposing a change; at 13.6% of the step this is now worth more than any remaining GPU-side category except the (closed) expert GEMM. |

### PROFILE-HOST-S6 — host-visible capture for the inter-replay gaps

| Field | Value |
|---|---|
| Date | 2026-07-25 |
| Hypothesis | PROFILE-S6's ~1237 µs/step of host gaps between CUDA-graph replays (548 µs after `index_elementwise_kernel`, 386 µs after `vectorized_elementwise_kernel`, 303 µs after `CatArrayBatchedCopy_vectorized`) are unattributed because every existing trace is GPU-only. A trace with `osrt` + CPU sampling + Python sampling will name the host functions occupying them and classify each gap as launch overhead, host synchronization, sampling/detokenization, scheduler bookkeeping, or ZMQ round-trip. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (session-6 set unchanged: `vllm_fused_moe.py`, `experts.py`, `router.py`, untracked `router_topk.py`). **No `megatron/` edits made by this task** — verified with `git diff --stat -- megatron`. |
| Changed files | `dev/moe_fused/profile_host_insession.sh` (new — host-visibility variant of `profile_insession.sh`), `dev/moe_fused/dispatch_host_profile.sh` (new — waits for the session to reach `running`, then dispatches the capture detached) |
| Runtime flags | All seven gates: `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1` |
| Image | cog dev image `ceecf5c304a5d8bd.sqsh` (`nvcr.io/nvidia/pytorch:26.06-py3`), venv `envs/megatron_lm/dd356431262b5db4` |
| Checkpoint / tokenizer | `qwen3-30b-a3b-mcore` / `qwen3-30b-a3b-hf` |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/PP1/EP4/ETP1, nvls dispatcher, vllm grouped-GEMM, `transformer_impl=inference_optimized`, `full_iteration_inference` CUDA graphs |
| Workload | gsm8k, BS256, OSL128 under nsys, 1 iter after a BS8/OSL32 warmup (identical to PROFILE-S6 except for the trace flags) |
| nsys version | 2026.3.1.117-263137992252v0 — verified on-node that `--python-sampling`, `--python-sampling-frequency`, `--backtrace`, `--samples-per-backtrace`, `--cpuctxsw`, `--osrt-threshold` are all supported before committing to the command line |
| nsys command line | `nsys profile --trace=cuda,nvtx,osrt --sample=process-tree --backtrace=fp --samples-per-backtrace=1 --cpuctxsw=process-tree --python-sampling=true --python-sampling-frequency=1000 --osrt-threshold=1000 --cuda-graph-trace=node --force-overwrite=true -o <RUN_DIR>/mcore_host_profile $PYBIN -m torch.distributed.run --nproc-per-node 4 ... -m examples.inference.launch_inference_server ...` (server args identical to `profile_insession.sh`). The script preflights this flag set against `python -c print(...)` and degrades to `--trace=cuda,nvtx,osrt --sample=process-tree --backtrace=fp --python-sampling=true` and then to `--trace=cuda,nvtx,osrt --sample=process-tree` if nsys rejects it, so an unsupported flag cannot abort the run. |
| Job / run | Intended session `qwen-comm` job 5601961 — **PREEMPTED** (`raw_state=PREEMPTED`, controller dead) before any capture ran; note `cog jobs get` still reported `RUNNING` while `cog session status` correctly reported `preempted`. Replacement: session `qwen-host`, job **5607600**, node `nvl72166-T15`, partition `batch`, `--time 02:00:00` (started 2026-07-26T06:53 UTC, expires ~08:53 UTC), 4 GPUs. Capture dispatched detached as exec **`d920241ea00a41618c63a75fefcb9ea2`** at 06:53:51 UTC; run dir **`sessions/qwen-host/prof/hosts6-1785048883`**. Exec log: `sessions/qwen-host/exec/runs/d920241ea00a41618c63a75fefcb9ea2/stdout.log`. |
| Throughput | n/a (profile capture) |
| Latency / TPOT | n/a (profile capture) |
| Correctness | n/a (profile capture) |
| Nsight artifacts | **Expected at `sessions/qwen-host/prof/hosts6-1785048883/mcore_host_profile.{nsys-rep,sqlite}` — dispatched and confirmed running, but not confirmed complete.** At hand-off the exec had reached the nsys flag preflight; the server load + BS8 warmup + BS256/OSL128 benchmark + sqlite export had not finished. The script ends with a sanity check that prints row counts for `CUPTI_ACTIVITY_KIND_KERNEL`, `CUPTI_ACTIVITY_KIND_RUNTIME`, `OSRT_API`, `COMPOSITE_EVENTS`, `SAMPLING_CALLCHAINS`, `PYTHON_SAMPLING_CALLCHAINS`, `PYTHON_SAMPLING_STRING`, `SCHED_EVENTS`, `GENERIC_EVENTS` and a `HOST_TABLES_PRESENT=` line — a capture with only `CUPTI_ACTIVITY_KIND_KERNEL` is a failed capture and must be re-run. |
| Result | **Inconclusive — capture not confirmed, no attribution performed.** The task lost its node: session `qwen-comm` job 5601961 was preempted roughly six hours into an eight-hour `batch_long` allocation, before the first capture could be dispatched, and the whole capture had to be re-queued from scratch on a new allocation that did not schedule inside the remaining budget. The deliverable that *was* produced is the reusable capture path: an nsys flag set validated against the installed 2026.3.1 on-node, a profile script that preflights and degrades rather than aborting, and a built-in host-table sanity check. Two operational findings worth carrying: (1) `cog jobs get --job-id` reported `RUNNING` for a preempted session — use `cog session status` to decide whether a session is usable; (2) an unrelated background `rsync` writing into the worktree caused two `WorktreeChangedError` dispatch failures, so no local process may write inside the repo while a `cog session exec` snapshot is in flight. |
| Next action | Check `sessions/qwen-host/exec/runs/d920241ea00a41618c63a75fefcb9ea2/stdout.log` for `PROFILE DONE` and `HOST_TABLES_PRESENT=True`. If present, run Workflow C on the new `.sqlite`, then window the `OSRT_API` / `CUPTI_ACTIVITY_KIND_RUNTIME` / `PYTHON_SAMPLING_CALLCHAINS` tables to the three inter-replay gaps and attribute them. If the exec never ran, re-dispatch `dev/moe_fused/profile_host_insession.sh` on a fresh session — it is unchanged and ready. |

### HOSTGAP-S6 — host attribution of the three inter-replay gaps

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | PROFILE-S6's three largest inter-kernel gaps (after `index_elementwise_kernel`, `vectorized_elementwise_kernel`, `CatArrayBatchedCopy_vectorized`, ~1237 µs/step together) can be attributed to named host functions from the PROFILE-HOST-S6 capture, and each can be classified as CPU-bound host execution, blocking synchronization, driver launch latency, allocator activity, a genuine GPU dispatch gap, or profiler overhead. |
| Code revision | `37d1adcf09390c2593ad18b8dbe3b3cdb7e25514`, branch `perf/moe-fused-align`, dirty (session-6 set). **No `megatron/` edits** — analysis only. |
| Changed files | `dev/moe_fused/analyze_hostgaps.py` (new — `gaps` brackets every inter-kernel gap in the kernel-union complement by the kernel ending at its start and the kernel starting at its end; `attrib` aggregates CUPTI runtime, OSRT, NVTX and native sampling backtraces over all instances of one bracketed gap class) |
| Runtime flags | n/a (analysis of existing traces) |
| Image | n/a |
| Checkpoint / tokenizer | n/a |
| Hardware / layout | OCI `oci-hsg`, 1×4 GB200, TP1/PP1/EP4/ETP1 (as captured) |
| Workload | Analysis of the BS256/OSL128 decode loop in two traces of the same seven-gate configuration |
| Job / run | session `qwen-host` job 5607600, exec `d920241ea00a41618c63a75fefcb9ea2`, `sessions/qwen-host/prof/hosts6-1785048883` |
| Throughput | n/a. For the record, the host-traced benchmark itself reports 6,631.7 tok/s / TPOT 38.60 ms (vs PROFILE-S6's 10,454.6 tok/s) — but that overhead lands almost entirely **outside** the decode loop (model load, chunked prefill, graph capture); the steady-state decode step is only 2.1% slower under host tracing. |
| Latency / TPOT | Steady-state decode step: **9.134 ms** (GPU-only) vs **9.330 ms** (host trace) = 1.021×. Idle 1965.7 vs 2128.0 µs = 1.083×. |
| Correctness | n/a (analysis) |
| Nsight artifacts | Host: `qwen-host:prof/hosts6-1785048883/mcore_host_profile.{nsys-rep,sqlite}` (615 MB / 5.9 GB). GPU-only reference: `nsys_trace/mcore_s6_tuned_osl128.sqlite`. Analysis: `dev/moe_fused/analyze_hostgaps.py` |

**Trace provenance and window — two corrections that had to be made first.**
The sqlite is intact (`pragma quick_check` ok) and complete despite the offline
`QdstrmImporter` recovery: all four ranks are present with ~479 k kernels each
spanning 75.0–293.0 s, four engine PIDs with ~665 k CUDA-runtime events each,
and OSRT/sampling covering 0–300 s. Two things were *not* as expected:

1. **The densest kernel region is not the decode loop.** Seconds 196–241 look
   like a steady loop (a `_fused_metadata_kernel` every ~146 ms, 48 layers per
   period) but it is **CUDA-graph capture**: 25 distinct capture streams, no
   BS256 LM-head GEMM, `index_elementwise_kernel` count **zero**, and
   `_multimem_all_gatherv_3tensor_kernel` averaging **1807 µs** (vs 15.8 µs in
   the GPU-only trace) because the symmetric-memory barrier absorbs inter-rank
   skew while the other ranks are capturing. The real BS256/OSL128 decode loop
   is a 1.3 s burst at **291.67–292.98 s** (146 anchor intervals, period 9.064
   ms), preceded by the BS8/OSL32 warmup at 287.51–287.84 s (44 steps, 7.53 ms).
   Analysing 217–241 s would have produced a 16×-inflated fiction.
2. **`PYTHON_SAMPLING_*` and the graph tables are absent** despite
   `--python-sampling=true`. Attribution therefore uses NVTX ranges +
   `CUPTI_ACTIVITY_KIND_RUNTIME` + `OSRT_API` + native `SAMPLING_CALLCHAINS`,
   which turned out to be sufficient because the engine emits per-phase NVTX.

**Method.** Kernel-union complement per anchor-to-anchor step, anchored on the
once-per-step LM-head GEMM `nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT` on
all four devices in both traces (forcing the same anchor matters: auto-detection
picked `vectorized_elementwise_kernel` on devices 0/2 of the host trace, which
straddles G1 and silently dropped it). Each gap is bracketed by both neighbours.
Windows: GPU-only 184.9–186.9 s (126 steps), host 291.75–292.95 s (117 steps).
The GPU-only trace is the source of truth for magnitude, the host trace for
attribution.

**Per-step gap budget (GPU-only trace, µs/step, per rank).**

| dev | step ms | busy ms | idle µs | G1 | G2A | G2B | G3 | G4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9.135 | 7.195 | 1940.2 | 559.0 | 63.7 | 259.1 | 303.8 | 48.8 |
| 1 | 9.135 | 7.181 | 1953.7 | 566.8 | 62.4 | 264.0 | 305.7 | 49.9 |
| 2 | 9.134 | 7.141 | 1993.8 | 582.7 | 63.1 | 275.8 | 308.6 | 54.1 |
| 3 | 9.133 | 7.158 | 1975.0 | 576.9 | 63.1 | 273.0 | 303.6 | 52.7 |
| **mean** | **9.134** | **7.169** | **1965.7** | **571.4** | **63.1** | **267.9** | **305.4** | **51.4** |

Every rank is within 4% of every other on every row — **no rank asymmetry**, as
expected when the four EP ranks are barrier-coupled by the multimem collectives.
The five bracketed gaps are **1259.3 µs/step = 13.79% of the step and 64.1% of
the idle**; the residual 706.4 µs is 1158 × ~0.61 µs of intra-graph node
dispatch. Host-trace equivalents (mean of ranks) are G1 641.2, G2 378.5, G3
308.2 µs — inflation 1.12× / 1.14× / 1.01×, so the host trace neither invents
nor hides these gaps.

**Gap 1 — `index_elementwise_kernel` → `vectorized_elementwise_kernel`.**
571.4 µs/step GPU-only (host: median 635.4, mean 640.9, p10 585.0, p90 676.1,
116 instances, 0.99/step). Verdict **(i) CPU-bound Python**. NVTX covers 72.5%:
`update_requests` 183.6 µs (28.6%), `initialize_attention_state` 151.7 (23.7%),
`active_request_mask` 78.0 (12.2%), `transfer_samples_to_cpu` 41.2 (6.4%),
`sampling` tail 10.5 (1.6%); the remaining 27.5% is an unlabelled engine window
between `update_requests` and `initialize_attention_state` (median 152.2 µs)
that is itself 39.6%-leaf `_PyEval_EvalFrameDefault` with `at::_ops::_to_copy`
in 34% of stacks. Only **55.1 µs (8.6%)** of the gap is inside any CUDA API
(`cudaMemcpyAsync` ×2 = 41.7 µs, `cudaLaunchKernel` 8.6, `cudaStreamSynchronize`
**3.4 µs = 0.5%**), and only 32.9 µs (5.1%) inside any syscall on the engine
thread — so ~86% is plain userspace execution. All 161 samples landing in these
windows have `threadState = Running`; of the 142 on the engine thread, 100% have
`_PyEval_EvalFrameDefault` and `_PyObject_MakeTpCall` somewhere in the stack,
with `at::native::copy_` in 12.7%. Source:
`text_generation_controller.py:1750` (`transfer_samples_to_cpu`, D2H at `:1723`),
`:1756`–`1805` (`active_request_mask`), `:1807`–`1811` (`update_requests` →
`dynamic_context.py:3515`), `:644`–`649` (`initialize_attention_state` →
`dynamic_context.py:2128`, whose own comment at `:643` reads "100% CPU
computation").

**Gap 2 — `vectorized_elementwise_kernel` → `vectorized_gather_kernel`, twice
per step, bimodal.** Instance **2B** (267.9 µs GPU-only; host median 301.1, mean
310.4) is NVTX-covered 97.7% by `initialize_attention_state` 159.9 µs (51.5%),
`forward_pass` head 99.0 (31.9%) and `transfer_bookkeeping_to_gpu` 44.3 (14.3%);
CUDA API is 35.9 µs (11.5%), engine-thread syscalls 7.0 µs (2.3%), and all 76
samples are Running with 100% `_PyEval_EvalFrameDefault` — verdict **(i)
CPU-bound Python**, the same bookkeeping chain as G1 continued past the first
GPU op it emits. Instance **2A** (63.1 µs GPU-only; host median 70.8) sits
entirely inside the `sampling` NVTX range and is 34.4% CUDA API
(`cudaLaunchKernel` 15.6 µs = 21.8%, `cudaStreamSynchronize` 8.7 µs = 12.1%),
with `at::_ops::index_Tensor::call` / `at::native::index_kernel` in 55.6%/44.4%
of its (only 9) samples — verdict **(i)+(iii)**, advanced-indexing dispatch plus
a short blocking stream sync. Source: `text_generation_controller.py:652`–`654`
(`transfer_bookkeeping_to_gpu` → `dynamic_context.py:2411`), `:790`/`:1864`
(`forward_pass`), `:1902`/`:2018`–`2023` (`sampling`; `torch.argmax` at `:2019`,
`.cpu()` at `:2022`).

**Gap 3 — `CatArrayBatchedCopy_vectorized` → `rmsnorm_fwd_tuned_kernel` (the
next graph replay).** 305.4 µs/step GPU-only (host median 311.5, mean 314.3, 117
instances). Verdict **(iii) driver graph-launch latency**. It is 100% inside
`forward_pass`, and **63.5% of it is a single `cudaGraphLaunch` call**: 132 calls
in the window, median **199.1 µs**, p10 188.6, p90 211.9, unimodal (so it is not
a graph-switch/upload artifact — one graph id, 140, carries 135,528 of the
window's kernels). Plus `cudaMemcpyAsync` 18.4 µs (5.8%) and ~96 µs of Python.
Engine-thread syscalls are 0.7%; all 78 samples are Running, 56.6% with
`cuGraphLaunch` as the deepest resolved frame and 57.9% with
`at::cuda::CUDAGraph::replay()` in the stack. At 1158 nodes per replay that is
**0.172 µs of host submit time per graph node**, on the critical path before the
first node executes.

**Aggregate, and the profiling-overhead correction.** GPU-busy is 7.169 ms =
78.5% of the 9.134 ms step; host gap is 1965.7 µs = **21.5%**. Splitting that
idle by confidence:

| Component | µs/step | % of step | Confidence |
|---|---:|---:|---|
| G1 + G2A + G2B + G4 (host Python + sampling dispatch) | 953.8 | 10.4% | Real — samples Running in the interpreter |
| G3 minus its `cudaGraphLaunch` (Python + H2D in `forward_pass`) | 111.4 | 1.2% | Real |
| `cudaGraphLaunch` inside G3 | 194.0 | 2.1% | **Artifact-suspect** |
| Intra-graph inter-node dispatch (1158 × ~0.61 µs) | 706.4 | 7.7% | **Artifact-suspect** |
| **Total idle** | **1965.7** | **21.5%** | |

Overhead was corrected by comparing the two traces step-for-step rather than by
assuming a factor: the host capture adds only **2.1%** to the decode step
(9.134 → 9.330 ms) and 8.3% to the idle, even though it costs 37% of
whole-benchmark throughput, because its cost falls on load/prefill/capture. The
one caveat that does **not** cancel is that *both* traces used
`--cuda-graph-trace=node`, which forces CUPTI to instrument every graph node;
the 194 µs `cudaGraphLaunch` and the 706 µs of inter-node dispatch — 900 µs/step,
9.9% of the step, 45.8% of the idle — are exactly the quantities that flag
distorts, and this analysis cannot separate them from the real cost. A
`--cuda-graph-trace=graph` control capture settles it for the price of one
profile run.

**What the recoverable upside is worth.** Taking the current best mcore of
25,495.9 tok/s (QWEN-022, 75.00% of vLLM's 33,994.5) and scaling by
`9134 / (9134 − Δ)` — an upper bound that assumes perfect removal and that the
OSL128 composition transfers to the OSL1024 throughput regime:

| Removed | Δ µs | ceiling tok/s | % of vLLM | Share of the remaining gap |
|---|---:|---:|---:|---:|
| G1 | 571.4 | 27,197 | 80.0% | 20.0% |
| G2 (both) | 331.0 | 26,454 | 77.8% | 11.3% |
| G3 | 305.4 | 26,378 | 77.6% | 10.4% |
| G1 + G2B (the serial bookkeeping chain) | 839.3 | 28,067 | 82.6% | 30.3% |
| All five bracketed gaps | 1259.3 | 29,573 | 87.0% | 48.0% |
| All idle | 1965.7 | 32,488 | 95.6% | 82.3% |

**The finding that reframes the lever.** G1 and G2B are not a scheduling
artifact that overlap could hide. The chain is
`graph(N) → logits(N) → argmax(N) → D2H tokens(N) → active_request_mask(N) →
update_requests(N) → initialize_attention_state(N+1) → H2D → graph(N+1)`, and
every link needs the previous one's result: the host cannot run step N+1's
bookkeeping during step N's graph replay because that bookkeeping consumes step
N's sampled tokens. This is a genuine serial data dependency, and it explains
QWEN-007 cleanly — `async-sched-mode=serial` measured +0.85% not because the
idle was illusory but because with a single request stream there is nothing to
overlap it *with*. The lever is therefore to make the CPU work cheaper or move
it onto the GPU, not to hide it.

**Ranked candidate next levers** (upside is a measured ceiling, not a prediction;
none of these has been measured):

1. **Make `initialize_attention_state` incremental** — ~311 µs/step (152 µs in
   G1 + 160 µs in G2B), pure CPU by construction. In steady-state decode the
   request set is fixed for 128 steps and only the sequence lengths increment,
   so most of the per-step attention metadata is recomputed unchanged. Ceiling
   ~3.4% (→ ~77.6% of vLLM); realistic maybe half. **Risk low-medium**: CPU-only,
   no numerics exposure, but correctness depends on invalidating the cache on
   every request add/pause/finish. Falsified if a fresh profile shows the G1 and
   G2B windows unchanged.
2. **Cut `update_requests` + `active_request_mask` + the unlabelled engine
   window** — 183.6 + 78.0 + ~175 = ~437 µs/step of Python over 256 requests.
   Ceiling 4.8% (→ ~78.8%). **Risk low**: no numerics, no guards, purely
   additive with everything else.
3. **`--cuda-graph-trace=graph` control capture** — no throughput at all, but it
   decides whether the 900 µs/step (9.9% of the step, 45.8% of the idle) of
   graph machinery is real or instrumentation. **Risk none, cost one profile
   run.** This gates lever 4 and should be run before it.
4. **Reduce graph node count** — each removed node is worth ~0.172 µs of host
   submit plus ~0.61 µs of GPU inter-node dispatch ≈ 0.78 µs, so 100 nodes ≈
   78 µs ≈ 0.85%. **Risk medium** and sharply diminishing; this is the fusion
   campaign that QWEN-019…022 already mined.
5. **Move per-step request bookkeeping onto the GPU** (persistent device-side
   batch state, no D2H→Python→H2D round trip per step), which is what vLLM does.
   Ceiling is most of the 839 µs chain, ~9.2%. **Risk high** — a large engine
   rewrite, multi-session.

Explicitly *not* recommended: re-litigating `async-sched-mode=serial`. The
dependency analysis above says the idle it targets is not overlappable with a
single request stream, which is consistent with its measured +0.85%.

| Field | Value |
|---|---|
| Result | **Supported — all three gaps attributed with numbers, and one taxonomy verdict each.** G1 (571 µs) and G2B (268 µs) are CPU-bound Python in the engine's per-step bookkeeping chain; G2A (63 µs) is sampling-path launch plus a short stream sync; G3 (305 µs) is 63.5% a single 199 µs `cudaGraphLaunch`. No rank asymmetry. The idle budget is 1065 µs of real host work and 900 µs of graph machinery whose reality is gated on a `--cuda-graph-trace=graph` control. The campaign-level conclusion is that the largest remaining lever is a serial data dependency, so it must be made cheaper rather than hidden. |
| Next action | Run the `--cuda-graph-trace=graph` control capture (lever 3, one profile run) to fix the denominator, then implement lever 1 (incremental `initialize_attention_state`) and measure it e2e at OSL1024 against CLEANBASE-S6 with the full seven-gate set. |

### Session 6 — conclusion & recommendation

The routing/permute lever is closed, and it delivered. Against the same-session
reference CLEANBASE-S6 (23,264.4 tok/s), the four accepted changes take mcore to
**25,495.9 tok/s = 75.00%** of the fixed vLLM OSL1024 baseline of 33,994.5, a
**+9.6%** session gain — and every one of them is bit-exact, so none of it was
bought with numerics.

| Change | Mechanism | e2e |
|---|---|---:|
| QWEN-019 `MCORE_MOE_FUSED_COUNT` | count + zero-fill folded into the table build (4 launches → 2) | +3.01% |
| QWEN-020 `MCORE_MOE_SUM_FAST` | topk-reduction locality test predicated instead of branched | ~+1% (measured +1.83%, partly variance) |
| QWEN-021 `MCORE_ROUTER_FUSED_TOPK` | softmax + top-8 select in one CTA-per-token kernel (4 launches → 1) | **+3.89%** |
| QWEN-022 `MCORE_MOE_FUSED_SCATTER` | scatter folded in as a second streaming pass (2 launches → 1) | +0.46% |

Reproduce the best configuration with all seven gates:
`MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1
MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1
MCORE_MOE_FUSED_SCATTER=1`.

Two methodological points worth carrying forward. First, QWEN-018's gate was
right to reject the two candidates the ledger had ranked first and second on
arithmetic alone — the win came from the third candidate the *measurement*
surfaced, and from the router pair the gate flagged but did not rank. Second,
microbench-to-e2e transfer was reliable for launch removals (QWEN-019 predicted
2.93% and got 3.01%; QWEN-021 predicted ~3.1% and got 3.89%) and unreliable for
in-kernel rewrites (QWEN-020 and QWEN-022 both landed at roughly a third to
double their predictions), so keep repeating e2e runs whose predicted effect is
under ~1%.

PROFILE-S6 re-ranks the step: routing is now 4.7% and the largest addressable
item is **1237 µs/step of host gaps between graph replays** (21.0% idle overall).
That is a host-side problem and needs a host-side profile before any code
change.

Append records using this exact structure:

```markdown
### QWEN-NNN — short name

| Field | Value |
|---|---|
| Date | YYYY-MM-DD |
| Hypothesis | One measurable claim |
| Code revision | Commit SHA and clean/dirty state |
| Changed files | Exact paths, or `none` for baseline |
| Runtime flags | Exact non-default flags |
| Image | Immutable image path/tag |
| Checkpoint / tokenizer | Exact paths |
| Hardware / layout | Cluster, GPUs, TP/PP/EP/ETP/DP |
| Workload | Dataset, batch, OSL, warmups, timed iterations |
| Job / run | Slurm job ID and run directory |
| Throughput | tokens/s |
| Latency / TPOT | ms / ms-token |
| Correctness | Prompt outputs and benchmark status |
| Nsight artifacts | `.nsys-rep`, `.sqlite`, analysis output |
| Result | Supported / rejected / inconclusive |
| Next action | One prioritized follow-up |
```

## Session 7 (2026-07-26) — host-side bookkeeping

### CLEANBASE-S7 — same-session seven-gate reference

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | Establish a same-session OSL1024 reference before measuring a sub-1% host-side lever |
| Code revision | `37d1adcf0`, branch `perf/moe-fused-align`, dirty with the session-6 change set |
| Changed files | none |
| Runtime flags | `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1` |
| Image | `agents-space/images/ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `agents-space/checkpoints/qwen3-30b-a3b-mcore` / `-hf` |
| Hardware / layout | OCI `oci-hsg`, 1 node 4×GB200, TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed |
| Job / run | job 5613090, `sessions/qwen-attnstate/e2e/ref-s7-1785084185` |
| Throughput | **25,805.9 tok/s** — per-iter 25,782.5 / 25,783.6 / 25,814.8 / 25,842.3 / 25,806.7 |
| Latency / TPOT | 9,867.5 ms / 9.920 ms/tok |
| Correctness | Coherent on all three temperature-0 prompts; benchmark 5/5 |
| Nsight artifacts | none (throughput run) |
| Result | Clean reference at **75.91%** of vLLM 33,994.5. Spread across the five iterations is 0.23%, so a 0.5% effect is resolvable in this session |
| Next action | A/B the incremental `initialize_attention_state` lever against this reference |

### QWEN-023 — incremental `initialize_attention_state`

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | The ~311 µs/step HOSTGAP-S6 attributed to `initialize_attention_state` is recomputation of step-invariant metadata; caching it behind a request-layout version counter removes most of it without changing a single generated token |
| Code revision | `37d1adcf0`, branch `perf/moe-fused-align`, dirty (session-6 set plus this change) |
| Changed files | `megatron/core/inference/contexts/dynamic_context.py`, `megatron/core/inference/contexts/attention_context/mha_metadata.py`, `dev/moe_fused/harness_attnstate.py` (new) |
| Runtime flags | seven gates as CLEANBASE-S7, plus `MCORE_INFER_INCR_ATTN_STATE=1` |
| Image / checkpoint / hardware / workload | identical to CLEANBASE-S7 |
| Job / run | job 5613090, `e2e/incr-attn-on-1785085393` and `e2e/incr-attn-on-rep2-*`; host microbenchmark exec `a5d20392e75848b9855366cd7844c850` |
| Throughput | **26,032.1** (per-iter 26,052.6 / 26,018.7 / 26,075.7 / 25,992.1 / 26,021.4) and **26,092.5** on an independent repeat pair |
| Latency / TPOT | 9,780.4 ms / 9.834 ms/tok; repeat 9.811 ms/tok (−92 µs/step vs reference on two-run means) |
| Correctness | 130 consecutive decode steps bit-identical under `MCORE_INFER_INCR_ATTN_STATE_VERIFY=1`; temperature-0 coherence output byte-identical to gate-OFF |
| Nsight artifacts | none needed — the effect is host wall time, measured directly |
| Result | **Accepted, +0.94%** (two OFF/ON pairs, non-overlapping per-iteration distributions) |
| Next action | Cut `update_requests` + `active_request_mask` + the unlabelled engine window (~437 µs/step, HOSTGAP-S6 lever 2) with the same version-counter machinery |

**Pre-optimization CPU breakdown.** Measured with `MCORE_INFER_ATTN_PROF=1`
(monotonic-ns marks around each phase) and with a standalone host harness
(`dev/moe_fused/harness_attnstate.py`) that drives a real
`DynamicInferenceContext` at BS256 through 400 steady-state decode steps with no
model loaded. The two agree to within 12%. Per call, 256 active requests:

| Phase | µs/call | What it is |
|---|---:|---|
| `slices` | 73.9 | build + pad the active-request slices, sampling-metadata copies, logit indices |
| `mhameta` | 32.2 | KV/query lengths, both cumsums, block table, `set_state_data` |
| `xfer_inner` | 26.9 | H2D copy of the bookkeeping buffer |
| `graphmatch` | 12.8 | batch dimensions + `match_graph_config` |
| `tokenpad` | 12.0 | stamp the token padding slots |
| `maxlen` | 12.0 | max-seqlen scalars |
| `pre` + `paddims` + `tail` | 1.8 | pending Mamba ops, padded dimensions, epilogue |
| **TOTAL** | **171.5** | |

The cost is **concentrated, not thin**: `slices` alone is 43%, and inside it two
statements — `build_active_slices` 29.3 µs and `pad_active_slices` 26.7 µs —
are a third of the whole call. Every one of those top statements recomputes a
value that is provably identical to the previous decode step. That is what made
the lever viable; had the profile come back flat across fifty 3 µs statements
the design below would not have been worth building.

**Design.** A single monotonic `_request_layout_version` counter is bumped by
every path that changes *which* request occupies a slot, a request's sampling
metadata, or its KV block table. `initialize_attention_state` takes the fast
path only when the current cache key equals the stored one, where the key is
`(layout_version, total_request_count, paused_request_count, active_token_count,
kv_block_allocator.total_avail, chunked_prefill_request_id,
num_speculative_tokens)`. The version is the guard; the other six are
independent structural sentinels, so a missed bump would additionally have to
coincide with an unchanged request count, token count and block-allocator
occupancy to escape detection. The fast path then recomputes only the KV
sequence lengths and their cumsum — statements copied character-for-character
from the full path so the buffers stay bit-identical — re-stamps the token
padding sentinels, rebinds `state_data` through a new
`MHAMetadata.restore_state_data` (needed because `reset_attention_state` clears
the max-seqlen scalars every step), and issues the H2D transfer. Everything
else is reused.

**Invalidation audit.** Thirteen mutation paths, each bumping the version:

| Path | Event covered |
|---|---|
| `add_request` | request add, including a chunked-prefill chunk |
| `resume_paused_requests` (inside `if resume_request_count > 0`) | resume |
| `_swap_book_keeping_tensors` | pause/resume slot swap |
| `_move_book_keeping_tensors` | slot compaction |
| `resolve_requests` | end-of-step finish/retire |
| `release_memory_blocks_from_request_indexes` | evict, block release |
| `prepare_requests` (inside `if num_new_blocks > 0`) | new KV block allocated mid-decode |
| `add_dummy_requests_parallel` (inside `if requests`) | dummy padding requests |
| `add_dummy_requests_for_cudagraph_capture` | graph capture padding |
| `add_dummy_requests_for_expert_parallel_step` | EP dummy step |
| `initialize_all_tensors` | tensor state (re)allocation |
| `reset_tensors`, `reset_metadata` | context reset |

Advancing a request's KV length is deliberately *not* a bump — that is the one
thing the fast path recomputes. Four conditions bail out to the full path
outright rather than relying on the key: `construct_graph_dimensions is not
None` (any graph-dimension change), `is_expert_parallel_dummy_cuda_graph_step`,
`is_hybrid_model` (Mamba state), and `num_prefill_requests != 0`. The cache is
additionally never *stored* while `is_creating_cuda_graphs` is set or when the
step did not use a CUDA graph. The class-scope declaration of every cache
attribute means any construction path starts cold and invalid.

**Correctness evidence.** `MCORE_INFER_INCR_ATTN_STATE_VERIFY=1` runs the fast
path, snapshots everything it produced, then recomputes the whole thing from
scratch and asserts equality of: both MHA query-length buffers and cumsums, both
KV-length buffers and cumsums, the block table, `active_request_last_token_idxs`,
`active_logit_idxs`, all seven `active_request_metadata` tensors, the entire GPU
bookkeeping buffer, the padded token/request counts, the padded batch
dimensions, the graph-selection flag, both max-seqlen scalars, and the
`(address, shape, dtype)` binding of every `state_data` view. **130 consecutive
decode steps passed with zero mismatches.** End to end, the three temperature-0
coherence completions are byte-identical between gate ON and gate OFF.

**Performance.** Host CPU per call **174.1 → 51.7 µs, a 3.37× reduction**; the
coarse breakdown collapses to `graphmatch=0.2 slices=1.1 tokenpad=0.1
mhameta=0.4 maxlen=0.1`, with the 48.2 µs residual being the KV-length
recompute plus the H2D bookkeeping copy, which cannot be cached. E2e:

| Run | Gate | Throughput | Per-iter |
|---|---|---:|---|
| `ref-s7` | OFF | 25,805.9 | 25,782.5 / 25,783.6 / 25,814.8 / 25,842.3 / 25,806.7 |
| `incr-attn-on` | ON | **26,032.1** | 26,052.6 / 26,018.7 / 26,075.7 / 25,992.1 / 26,021.4 |
| `ref-s7-rep2` | OFF | 25,831.7 | 25,710.8 / 25,893.9 / 25,863.9 / 25,875.4 / 25,815.4 |
| `incr-attn-on-rep2` | ON | **26,092.5** | 26,157.8 / 26,085.1 / 26,096.3 / 26,106.8 / 26,017.0 |

Pairwise the lever is +0.88% and +1.01%; on the two-run means (OFF 25,818.8, ON
26,062.3) it is **+0.94%**. Across all twenty timed iterations the slowest ON
iteration (25,992.1) is still faster than the fastest OFF iteration (25,893.9),
so the two distributions do not overlap at all and the sign is certain despite
the effect being under 1%. TPOT 9.915 → 9.823 ms/tok, i.e. **−92 µs/step**.
New best = **76.67%** of vLLM 33,994.5.

**Why it delivered 0.94% and not the 3.4% ceiling.** The harness measures one
call at 174.1 µs and the fast path removes 122.4 µs of it, but the per-step
saving observed in TPOT is 92 µs — about three quarters of a single call's
worth, against HOSTGAP-S6's 311 µs/step spread over two windows. So the ceiling
was overstated for two compounding reasons: the H2D bookkeeping transfer is
~27 µs of every call and is not cacheable (the GPU buffer must be rewritten
every step), and the second of the two per-step windows is evidently not all
`initialize_attention_state` doing cacheable work. The honest read is that the
remaining host-side upside lives in `update_requests` and the engine window,
not here.

### CGTRACE-CONTROL — is the graph machinery real?

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | The ~900 µs/step HOSTGAP-S6 attributed to CUDA-graph machinery is an artifact of `--cuda-graph-trace=node`; capturing with `=graph` will shrink it |
| Code revision | `37d1adcf0`, branch `perf/moe-fused-align`, dirty |
| Changed files | `dev/moe_fused/profile_insession.sh` (`CUDA_GRAPH_TRACE` env, bounded nsys stop + qdstrm recovery), `dev/moe_fused/analyze_cgtrace.py` (new) |
| Runtime flags | seven gates plus `MCORE_INFER_INCR_ATTN_STATE=1`, identical in both captures |
| Workload | gsm8k, BS256, OSL128, 1 warmup request + 1 timed request |
| Job / run | job 5613090, `prof/cgt-node-1785086713`, `prof/cgt-graph-1785086964` |
| Nsight artifacts | `mcore_profile.{nsys-rep,sqlite}` in both run dirs; analysis exec `d6f572f2613645cdbb05226dbc8e5e98` |
| Result | **Hypothesis rejected — the machinery is real** |
| Next action | Treat HOSTGAP-S6's ranking as sound; the next lever is `update_requests` (lever 2), not graph-node reduction (lever 4), on measured size |

| Metric (same rank, steady decode window) | `=node` | `=graph` | Δ |
|---|---:|---:|---:|
| Step period | 8882.5 µs | 8945.1 µs | −0.71% |
| Host `cudaGraphLaunch` (median) | 190.0 µs | 184.7 µs | +2.8% |
| Traced kernels per step | 1169 | 54 (graph interior hidden) | — |
| Steady steps analyzed | 91 | 88 | — |

The step period is the number that matters, and it does not move: if CUPTI's
per-node instrumentation were adding ~900 µs to a 9 ms step, removing it would
have shortened the step by 10%, and instead the graph-mode capture came out
0.71% *slower*, which is ordinary run-to-run variance. The 190 µs
`cudaGraphLaunch` survives the control almost unchanged, so it is genuine
driver-side submit cost for a 1158-node graph. Every wall-time attribution in
PROFILE-S6 and HOSTGAP-S6 therefore stands as measured.

One limit worth recording: `CUPTI_ACTIVITY_KIND_GRAPH_TRACE` in the graph-mode
capture holds only 417 rows for the whole run (6 inside the analyzed window),
so the GPU-side "inter-node dispatch" half of the 900 µs cannot be
independently confirmed from this control. The step period and the host submit
cost are settled; that component is not.

### CLEANBASE-S8 — same-session eight-gate reference

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | Establish a same-session OSL1024 reference before measuring the second host-side lever |
| Code revision | `37d1adcf0`, branch `perf/moe-fused-align`, dirty with the session-7 change set |
| Changed files | none |
| Runtime flags | `MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1 MCORE_INFER_INCR_ATTN_STATE=1` |
| Image | `agents-space/images/ceecf5c304a5d8bd.sqsh` |
| Checkpoint / tokenizer | `agents-space/checkpoints/qwen3-30b-a3b-mcore` / `-hf` |
| Hardware / layout | OCI `oci-hsg`, 1 node 4×GB200 (`nvl72039-T01`), TP1/PP1/EP4/ETP1 |
| Workload | gsm8k, BS256, OSL1024, 2 warmup + 5 timed |
| Job / run | job 5616264, `sessions/qwen-updreq/e2e/ref-s8-1785099726` (and `ref-s8-rep2-1785101268`) |
| Throughput | **25,944.9 tok/s** — per-iter 25,669.0 / 25,983.1 / 26,011.2 / 26,023.4 / 26,041.7. Repeat: **25,983.9** — per-iter 26,021.5 / 25,976.0 / 26,037.9 / 25,983.6 / 25,901.2 |
| Latency / TPOT | 9,820.6 ms / 9.867 ms/tok; repeat 9.852 ms/tok |
| Correctness | Coherent on all three temperature-0 prompts; benchmark 5/5 |
| Nsight artifacts | none (throughput run) |
| Result | Clean reference at **76.32%** of vLLM 33,994.5 (two-run mean 25,964.4 = 76.38%) |
| Next action | A/B the reduced-op `update_requests` lever against this reference |

The first run's iteration 1 is a 1.3% cold outlier; iterations 2–5 span 0.23%.
The repeat run does not show it and spans 0.53% across all five. Two-run mean
25,964.4 tok/s is the number the lever below is judged against.

### QWEN-024 — reduced-op post-sampling bookkeeping

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | HOSTGAP-S6 lever 2. The ~437 µs/step of host work after the sampled tokens return is three different things, not one: measure `update_requests`, `active_request_mask` and the unlabelled engine window separately, name the window, and only then decide whether any of it is attackable |
| Code revision | `37d1adcf0`, branch `perf/moe-fused-align`, dirty (session-7 set plus this change) |
| Changed files | `megatron/core/inference/contexts/dynamic_context.py` (`_write_decode_token_bookkeeping_fast`, `_write_token_bookkeeping_reference`, `_verify_decode_token_bookkeeping`, `_decode_req_idx_arange` cache, two count reductions, env `MCORE_INFER_VEC_UPDATE_REQS`, `MCORE_INFER_VEC_UPDATE_REQS_VERIFY`), `megatron/core/inference/text_generation_controllers/text_generation_controller.py` (`_empty_finished_idxs`, `_empty_finished_ids`, no-finisher short-circuit), `dev/moe_fused/harness_updreq.py` (new) |
| Runtime flags | eight gates as CLEANBASE-S8, plus `MCORE_INFER_VEC_UPDATE_REQS=1` |
| Image / checkpoint / hardware / workload | identical to CLEANBASE-S8 |
| Job / run | job 5616264, `e2e/vecupd-on-1785100576` and `e2e/vecupd-on-rep2-*`; harness execs `eaca22426f654bfb938bd68c13c7bff8` (gate OFF breakdown), `94604a2f79824182bef77456c9a2b7ee` (equivalence, verify, gate ON) |
| Throughput | **26,128.2** and **26,270.3 tok/s** on two independent OFF/ON pairs |
| Latency / TPOT | 9,745.4 ms / 9.798 ms/tok; repeat 9,691.7 ms / 9.745 ms/tok (−88 µs/step vs the reference on two-run means) |
| Correctness | 400-step two-context equivalence with 30 mid-batch terminations and a KV block-boundary crossing; 300 consecutive steps under `MCORE_INFER_VEC_UPDATE_REQS_VERIFY=1`; temperature-0 coherence output byte-identical to gate-OFF |
| Nsight artifacts | OSL128 A/B pair, `prof/g1-off-1785102888` and `prof/g1-on-1785103157`, `mcore_profile.{nsys-rep,sqlite}`; analysis exec `20bc3a28e2344c9b8f51aae8c90f6982`; window finder `dev/moe_fused/find_decode_window.py` (new) |
| Result | **Accepted, +0.90%** (two OFF/ON pairs: +0.71% and +1.10%), and the G1 window shrank by the predicted amount |
| Next action | The remaining host-side upside is `post_process_requests`, which this record shows is not of the same kind; see the ranked next levers below |

**Pre-optimization CPU breakdown — and what the unlabelled window actually is.**
Measured with `dev/moe_fused/harness_updreq.py`, which drives a real
`DynamicInferenceContext` at BS256 through a synthetic steady-state decode loop
and calls the real `DynamicInferenceEngine.post_process_requests` against a stub
engine holding 256 real `DynamicInferenceRequest` objects. No model is loaded,
so a full breakdown takes about 90 seconds. Per step, 256 active requests:

| Region | µs/step | HOSTGAP-S6 | Nature |
|---|---:|---:|---|
| `active_request_mask` | 46.9 | 78.0 | ~11 whole-tensor CPU ops on 256 elements |
| `update_requests` | 137.6 | 183.6 | ~22 whole-tensor CPU ops on 256 elements |
| `post_process_requests` | 187.9 | ~152.2 (unlabelled) | per-request Python object churn |
| **TOTAL** | **372.4** | **~413.8** | |

The harness runs 10–20% cheaper than the in-server measurement across all three
regions, which is the expected direction (no server threads, no allocator
pressure, no profiler), and the three ratios agree to within 15% of each other.

**The unlabelled engine window is `DynamicInferenceEngine.post_process_requests`,
reached through `async_bookkeep`.** It appeared unlabelled in HOSTGAP-S6 because
the controller emits NVTX through `torch.cuda.nvtx.range_push` directly, which is
always live, while the engine emits through `megatron.core.utils.nvtx_range_push`,
which is inert unless `_nvtx_enabled` is set — so the engine's own `bookkeeping`
and `detokenization` ranges were simply never pushed in that capture. The
measured 187.9 µs/step also matches the window's 152.2 µs median once the
harness discount is applied. Its `cProfile` composition, over 100 steps:

| Item | calls/step | What it is |
|---|---:|---|
| `post_process_requests` body | 1 | 61% of total time is straight-line interpreter work in the per-request loop |
| `builtins.len` | 1,718.8 | ~6.7 length probes per request per step |
| `get_request` | 256 | `self.requests[id].record[-1]` dict + list indexing |
| `list.append` | 256 | `active_request_ids.append` |
| `builtins.isinstance` | 256 | the `if not isinstance(tokens, list)` scalar wrap |
| `_check_stop_words_for_request_post_append` | 256 | returns immediately (no stop words configured) |
| `DynamicInferenceRequestRecord.__getitem__` | 256 | record `[-1]` |
| tensor ops | 3 | three `.tolist()` calls for the whole step |

**The falsification criterion, honored.** The pre-agreed rule was: vectorize if
the cost is a uniform per-request loop expressible as tensor ops, and stop if it
is per-request Python object churn. The measurement splits the chain cleanly in
half and the two halves fall on opposite sides of that line.

`post_process_requests` is **churn, and it was left alone**. It is 187.9 µs/step
— the single largest item in the chain — and it does essentially no tensor work:
three `.tolist()` calls for the entire step, against 256 dict lookups, 256 list
appends, 256 method calls and ~1,719 `len()` probes. Nothing here vectorizes.
Making it cheaper means adding a fast path through the request state machine —
token appending, termination, stop words, log-prob routing — which is exactly the
EOS/length handling the task flagged as dangerous, for a benefit that would have
to be re-argued from scratch. It is not touched by this change.

The other half is **neither** of the two anticipated cases. `update_requests` and
the `active_request_mask` block contain no per-request Python loop at all; they
are already vectorized. Their cost is ~33 whole-tensor operations on 256-element
CPU tensors at 2–16 µs each, and roughly half of that work is *provably dead* in
the decode regime. Three statements in `update_requests` cost 30.0 µs/step
between them and, at `num_speculative_tokens == 0`, compute nothing:

| Statement | µs/step | Why it is dead at one generated token per request |
|---|---:|---|
| `token_to_pos_ids` write | 16.0 | `repeat_interleave(1)` is identity, and it adds `torch.arange(1).repeat(256)` — a 256-element zero vector |
| `raw_positions` + `crosses_boundary` + `.any()` | 10.1 | builds a `[256, 1]` tensor whose only consumer is an `else` branch that `num_speculative_tokens == 0` makes unreachable |
| `token_to_request_idx` write | 7.3 | `repeat_interleave(1)` is identity over a `torch.arange` that is a pure function of the request-slot bounds |

So the lever here is **op elimination**, not vectorization. That is a third
category the dichotomy did not anticipate, and it is the low-risk one: it changes
no per-request semantics, touches no termination logic, and every removed
operation is removed because its result is bit-identical to a cheaper one.

**Design.** One env gate, `MCORE_INFER_VEC_UPDATE_REQS`, default OFF. Note that
the mechanism is deliberately unlike QWEN-023: nothing is cached across steps
except a `torch.arange` that depends only on the request-slot bounds, because
`update_requests` consumes the current step's freshly sampled tokens and there is
no cross-step reuse to exploit. Five changes:

1. The per-token bookkeeping tail of `update_requests` is factored into
   `_write_token_bookkeeping_reference`, kept verbatim, and a new
   `_write_decode_token_bookkeeping_fast` used only when
   `num_speculative_tokens == 0`. The fast form drops the three dead statements
   above, updates `request_last_kv_block_offset` without the defensive clone
   (whose only other consumer was `raw_positions`), and reuses a cached
   `torch.arange(paused_request_count, total_request_count)` invalidated by a
   bounds compare.
2. `finished_request_count` is derived as `numel() - active_request_count`
   instead of a second full comparison and reduction. The mask is 0/1 by
   construction — it is an `&` of two `.byte()` comparisons in the controller —
   so the complement is exact.
3. `active_requests_requiring_new_block` stays a bool tensor instead of being
   cast with `.byte()`. Every downstream use (`torch.nonzero`, `== 0`, scalar
   assignment, `sum`) is dtype-agnostic.
4. Its population count uses `int(tensor.sum())` rather than `(tensor == 1).sum().item()`.
5. In the controller, when the mask says nothing finished this step, the
   `torch.nonzero` scan and the advanced-index gather it feeds are skipped in
   favour of cached empty tensors of the same dtype and device.

**Correctness evidence.** Three independent checks, all on the eight-gate config.

*Two-context equivalence, 400 steps.* Two `DynamicInferenceContext` objects are
built identically and driven through the same scripted mask sequence, one with
the gate off and one on, comparing after every step: `total_request_count`,
`paused_request_count`, `active_token_count`, nine request bookkeeping tensors
(`request_ids`, `request_kv_length_offsets`, `request_query_lengths`,
`request_output_lengths`, `request_last_kv_block_offset`,
`request_last_kv_block_id`, `request_kv_block_counts`,
`request_to_kv_block_ids`, `request_in_prefill_status_tensor`), all six
`token_to_*` tensors, and the KV block allocator's `total_avail`,
`active_count` and `paused_count`. The script **terminates three requests
mid-batch every 37 steps** — 30 terminations in total, leaving 226 of 256 active
— and runs past the 256-token block boundary so the pause / new-block / resume
branches all execute under both gates. Zero mismatches. This is the check that
directly answers the off-by-one-in-termination risk: the finished-request count,
the finished index set, and the resulting slot compaction are all compared
element-wise on the steps where requests actually finish.

*Verify mode, 300 steps.* `MCORE_INFER_VEC_UPDATE_REQS_VERIFY=1` runs the fast
tail, snapshots the six token buffers plus `request_last_kv_block_offset` and
`active_token_count`, restores the inputs, runs the reference tail, and asserts
equality. 300 consecutive steps, zero mismatches.

*End to end.* The three temperature-0 coherence completions are byte-identical
between gate ON and gate OFF.

**Performance.** Host CPU per step, from the harness at BS256:

| Region | Gate OFF | Gate ON | Δ |
|---|---:|---:|---:|
| `active_request_mask` | 46.9 | 34.9 | −12.0 |
| `update_requests` | 137.6 | 74.0 | −63.6 (1.86×) |
| `post_process_requests` | 187.9 | 184.7 | unchanged, as intended |
| **chain total** | **372.4** | **293.7** | **−78.7** |

E2e:

| Run | Gate | Throughput | Per-iter |
|---|---|---:|---|
| `ref-s8` | OFF | 25,944.9 | 25,669.0 / 25,983.1 / 26,011.2 / 26,023.4 / 26,041.7 |
| `vecupd-on` | ON | **26,128.2** | 25,856.1 / 26,141.3 / 26,233.9 / 26,141.6 / 26,272.3 |
| `ref-s8-rep2` | OFF | 25,983.9 | 26,021.5 / 25,976.0 / 26,037.9 / 25,983.6 / 25,901.2 |
| `vecupd-on-rep2` | ON | **26,270.3** | 26,291.7 / 26,281.1 / 26,279.5 / 26,302.1 / 26,197.4 |

Pairwise the lever is **+0.71%** and **+1.10%**; on the two-run means (OFF
25,964.4, ON 26,199.3) it is **+0.90%**. Nine of the ten ON iterations are
faster than every one of the ten OFF iterations; the single exception is the
cold first iteration of the first ON run (25,856.1), and both OFF runs show the
same cold-first-iteration pattern. Restricted to iterations 2–5 the two
distributions do not overlap at all (slowest ON 26,141.3 > fastest OFF
26,041.7). TPOT 9.860 → 9.771 ms/tok on two-run means, i.e. **−88 µs/step**.
New best = **77.28%** of vLLM 33,994.5 on the best run, **77.07%** on the
two-run mean.

**Why it delivered less than the 4.8% ceiling.** The ceiling assumed the whole
437 µs/step chain could vanish. Half of it is `post_process_requests`, which this
change deliberately does not touch, so the reachable half was ~185 µs/step of
which the harness says 78.7 µs was removed — 42% of the reachable half, 18% of
the whole chain. TPOT moved 9.860 → 9.771 ms/tok on two-run means, i.e.
−88 µs/step, against a harness-measured saving of 78.7 µs. The two agree to
within 12%, in the direction the harness discount predicts, which is the
strongest evidence that the host saving is what produced the throughput change
rather than session drift.

**Profile confirmation — HOSTGAP-S6's falsification test for this lever class.**
HOSTGAP-S6 said a lever of this kind is falsified if a fresh profile shows the G1
window unchanged. Two BS256/OSL128 captures were taken back to back in the same
session at the same eight-gate code, differing only in the gate, and analysed
with `analyze_hostgaps.py gaps` on the forced LM-head anchor
`nvjet_sm100_tst_512x64_64x3_2x1_2cta_v_bz_TNT`. The G1 bracket is now
`index_elementwise_kernel -> vectorized_gather_kernel` rather than
`-> vectorized_elementwise_kernel`, because QWEN-023 changed which kernel the
bookkeeping chain emits first; it is the same gap class (largest gap, ~1.0
instances per step).

| Metric (mean of devices 0 and 3) | Gate OFF | Gate ON | Δ |
|---|---:|---:|---:|
| G1 window | 640.6 µs | 562.3 µs | **−78.3 µs** |
| Step period | 8.9425 ms | 8.8530 ms | −89.5 µs (−1.00%) |
| Total idle | 1757.0 µs | 1679.2 µs | −77.8 µs |
| GPU-busy | 7.2015 ms | 7.2045 ms | +3.0 µs (noise) |
| Kernels per step | 1169 | 1169 | 0 |
| G3 (`CatArrayBatchedCopy` → `rmsnorm`) | 295.6 / 293.0 | 296.8 / 293.2 | unchanged |
| G2A (`vectorized_elementwise` → `vectorized_gather`) | 67.1 / 60.4 | 65.9 / 63.6 | unchanged |

Four independent measurements agree: the harness says 78.7 µs of host work was
removed, the G1 window shrank 78.3 µs, the step period shortened 89.5 µs and
TPOT shortened 88 µs. GPU-busy time and the per-step kernel count are identical,
and the two gaps this change does not touch did not move, so the saving is
host-side and localized exactly where it was designed to be. Not falsified.

**Ranked next levers, updated.** (Lever 1 below was then implemented in the same
session and accepted as QWEN-025; the decision-gate measurement that justified
starting it is recorded here.)

1. **`post_process_requests` fast path** — 187.9 µs/step harness, ~152 µs/step
   in-server, now the largest single host item in the decode loop. A fast path
   for the common decode case (no speculative tokens, no stop words, no log
   probs, not chunked prefill, request not finishing) could collapse the loop
   body to a few operations. Ceiling ~1.7% e2e. **Risk high** — it is the request
   termination state machine. Falsified if a `cProfile` of the fast path does not
   cut the per-step call count below ~500.

   **Decision-gate measurement, taken with the time left in this session**
   (harness exec `4c0a584d22704d4ab1d8c5a7223bc8c2`, gate ON, BS256, 200
   timed calls against the real engine method and against an in-harness
   prototype of the decode fast path). The prototype implements the common
   decode case only and removes the per-request dict lookup, the record `[-1]`
   indexing, the `isinstance` scalar wrap, the `_check_stop_words_...` call and
   the repeated `len()` probes, resolving `(request, token_limit)` pairs once
   per request-id set instead of once per step:

   | Path | µs/step |
   |---|---:|
   | `post_process_requests`, as shipped | 213.8 |
   | decode fast-path prototype | 33.8 |
   | **reducible** | **180.0 (84%)** |

   So the lever clears its own gate decisively: 180 µs/step against an 8.85 ms
   step period is **~2.0% e2e**, roughly twice what QWEN-024 delivered, and it
   is the largest remaining host item. The mechanism is neither caching
   (QWEN-023) nor op elimination (QWEN-024) but a third thing — collapsing a
   per-request Python loop body — so it needs its own correctness argument, and
   the termination path is exactly where that argument is hard. Note the
   prototype is a *measurement*, not an implementation: it does not handle stop
   words, log probs, chunked prefill, speculative tokens or eviction, and those
   must fall back to the existing loop.
2. **Reduce graph node count** (HOSTGAP-S6 lever 4, now unblocked by
   CGTRACE-CONTROL) — ~0.78 µs per removed node, so 100 nodes ≈ 0.85%.
   **Risk medium**, sharply diminishing.
3. **Move per-step request bookkeeping onto the GPU** (HOSTGAP-S6 lever 5) —
   ceiling ~9.2%, **risk high**, multi-session.

### QWEN-025 — decode fast path for `post_process_requests`

| Field | Value |
|---|---|
| Date | 2026-07-26 |
| Hypothesis | The decision-gate measurement in QWEN-024 says 84% of `post_process_requests` is reducible without any tensor work, by collapsing the per-request loop body. If that translates, it is worth ~2.0% e2e — twice QWEN-024 |
| Code revision | branch `perf/moe-fused-align`, dirty (session-7 set, QWEN-024, and this change) |
| Changed files | `megatron/core/inference/engines/dynamic_engine.py` (`_post_process_requests_decode_fast`, `_ppr_cache_epoch` invalidation at the three request-record mutation sites, env `MCORE_INFER_FAST_POST_PROCESS`, `..._VERIFY`), `dev/moe_fused/harness_updreq.py` (`run_ppr_equivalence`, in-process gate A/B) |
| Runtime flags | eight gates plus `MCORE_INFER_VEC_UPDATE_REQS=1`, plus `MCORE_INFER_FAST_POST_PROCESS=1` |
| Image / checkpoint / hardware / workload | identical to CLEANBASE-S8 |
| Job / run | job 5616264, e2e `ppr-off-1785104484` / `ppr-on-1785104751` / `ppr-off-r2-1785105441` / `ppr-on-r2-1785105667` / `ppr-off-r3-1785105923` / `ppr-on-r3-1785106159`; harness execs `e620b4429e3945e6a4b0ab61d731c372` (equivalence), `81ab8e8ee09840eebf825420633b2efc` (gate A/B + verify) |
| Throughput | **26,361.7 tok/s** on the three-pair ON mean; best single run **26,430.7** |
| Latency / TPOT | 9,658.7 ms / 9.7111 ms/tok on the ON mean, against 9.8060 OFF — **−94.9 µs/step** |
| Correctness | 400-step two-engine equivalence including 10 finisher steps and 200 steps past the token limit; 300 steps under `MCORE_INFER_FAST_POST_PROCESS_VERIFY=1`; temperature-0 coherence byte-identical |
| Nsight artifacts | OSL128 A/B pair `prof/g2-off-1785106875` and `prof/g2-on-1785107105`; analysis exec `0ec0392b8c1d47c5a099cee585d777a4`. Two follow-on host-visibility captures (`prof/hosts9-*`, `prof/hosts9b-*`) failed to import; see the note under the next-lever list |
| Result | **Accepted, +0.98%** on three-pair means (+0.59% on a best-4-of-5 trim), ON wins all three pairs |
| Next action | Host-side decode work is now 148 µs/step against 372 µs before QWEN-024; the next lever is graph node count, not the host chain |

**Why this was started despite QWEN-024's stop verdict.** QWEN-024 declined to
touch `post_process_requests` because it is per-request Python object churn and
therefore not *vectorizable* — that was the pre-agreed falsification criterion
and it still holds. What the QWEN-024 decision-gate measurement then showed is
that being unvectorizable is not the same as being irreducible: an in-harness
prototype of the plain decode case ran at 33.8 µs/step against the real
method's 213.8 µs, i.e. 84% of the cost is dict lookups, record indexing, an
`isinstance` scalar wrap, a stop-word call that returns immediately, and ~1,719
`len()` probes, none of which the plain decode case needs. That is a third
mechanism, distinct from QWEN-023's caching and QWEN-024's op elimination:
collapsing a loop body.

**Design — the safety property comes first.** One env gate,
`MCORE_INFER_FAST_POST_PROCESS`, default OFF, dispatching to
`_post_process_requests_decode_fast` at the top of `post_process_requests`. The
fast path returns `None` and the full reference loop runs whenever anything is
not the plain decode case. The declining conditions are: speculative decoding
or accepted tokens, log probs, top-n log probs, finished routing block ids,
token event tracking, any pending stop-word state, chunked prefill in flight, a
TPOT sample due this step (`step_time > 0`), any eviction, **or any request
finishing this step**.

That last condition is the design's whole safety argument. The task flagged an
off-by-one in EOS/length handling as the dangerous failure mode, and the honest
way to remove that risk is not to test it harder but to never enter it: the
fast path is structurally unreachable on any step where a request finishes, so
the termination state machine — pop, future resolution, routing
reconstruction, status, finish event — is only ever executed by the original,
well-tested code. At BS256 steady-state decode essentially every step has no
finisher, so nearly all the work is still avoided; the OSL1024 benchmark takes
the fast path on all but a handful of its ~1,024 steps.

What remains in the fast path is one bounded append per request:

```python
if num_generated < limits[i]:
    generated.append(token)
```

which is the exact one-token specialization of the reference trim — the
reference computes `keep = num_tokens_to_generate - len(generated_tokens)` and
slices `tokens[:keep]`, which for a single token keeps it iff the request is
below its limit and drops it entirely otherwise. The first-token TTFT sample is
preserved inline on the `num_generated == 0` branch.

The `(request, token_limit)` pairs are resolved once per active request-id set
rather than once per step, keyed on `(epoch, tuple(request_ids))`. The epoch is
bumped at all three sites that can change which object `record[-1]` resolves
to — the two `record.checkpoint()` calls (recompute-suspend and eviction) and
the `RequestEntry` insertion in `_add_request` — so a stale resolution is not
representable rather than merely unlikely. A configured stop word on any
request also declines, since the post-append scan is the one piece of
per-request work here that is not a bounded append.

**Correctness evidence.** Three independent checks.

*Two-engine equivalence, 400 steps.* Two stub engines are built with identical
request sets and driven through the same token stream, one gate OFF and one ON,
compared after every step on the full observable result: returned active id
list, finished record count and contents, `finished_request_count`, the live
request-id set, and per request `generated_tokens`, `generated_length`, `status`
and whether TTFT is set. Three requests are finished mid-batch every 37 steps
(10 finisher steps, 30 requests finished, all 10 correctly declined by the fast
path), and `num_tokens_to_generate` is set to 200 against 400 steps so every
surviving request spends 200 consecutive steps *at* its token limit — the
append-suppression branch — and both gates agree that it holds exactly 200
tokens at the end. Zero mismatches. This is the check that answers the
off-by-one risk directly, on both sides of the limit and on the finish steps.

*Verify mode, 300 steps.* `MCORE_INFER_FAST_POST_PROCESS_VERIFY=1` independently
recomputes the expected post-state of every request from the pre-state and the
sampled token before running the fast loop, then asserts equality. 300
consecutive steps, zero mismatches.

*End to end.* The three temperature-0 coherence completions are byte-identical
between gate ON and gate OFF, character for character across all three prompts.

**Performance.** Host CPU per step at BS256, both arms measured in the same
process against the same engine so the comparison carries no process-to-process
noise:

| Region | Gate OFF | Gate ON | Δ |
|---|---:|---:|---:|
| `post_process_requests` | 211.0 | 27.2 | **−183.9 (7.77×)** |

And the whole post-sampling chain, against the pre-QWEN-024 starting point:

| Region | Before QWEN-024 | After QWEN-024 | After QWEN-025 |
|---|---:|---:|---:|
| `active_request_mask` | 46.9 | 34.9 | 34.1 |
| `update_requests` | 137.6 | 74.0 | 73.7 |
| `post_process_requests` | 187.9 | 184.7 | **40.3** |
| **chain total** | **372.4** | **293.7** | **148.0** |

The pre-registered falsification test for this lever was "falsified if a
`cProfile` of the fast path does not cut the per-step call count below ~500."
Measured: **261 calls/step** (7,831 over 30 steps), of which 256 are the single
`len()` per request, against ~1,719 `len()` probes alone before. Not falsified.

E2e, three OFF/ON pairs run back to back in one session:

| Pair | Gate | Throughput | Per-iter |
|---|---|---:|---|
| 1 | OFF | 26,054.0 | 26,251.6 / 26,249.3 / 26,241.1 / 26,274.0 / 25,283.4 |
| 1 | ON | **26,313.1** | 26,104.8 / 26,306.3 / 26,431.3 / 26,418.0 / 26,307.9 |
| 2 | OFF | 26,101.0 | 26,183.2 / 26,218.8 / 26,253.2 / 26,195.0 / 25,664.3 |
| 2 | ON | **26,430.7** | 26,442.2 / 26,416.0 / 26,428.8 / 26,426.4 / 26,440.3 |
| 3 | OFF | 26,164.4 | 25,883.8 / 26,270.7 / 26,270.5 / 26,228.0 / 26,173.0 |
| 3 | ON | **26,341.2** | 26,404.3 / 26,333.4 / 26,243.2 / 26,376.0 / 26,349.9 |

Pairwise **+0.99%, +1.26%, +0.68%**; on three-pair means (OFF 26,106.5, ON
26,361.7) **+0.98%**. The task set +0.5% as the noise floor, so the headline
number clears it, but each OFF run contains one slow outlier iteration (25,283.4
and 25,664.3 in the fifth position, 25,883.8 in the first) which flatters the
mean comparison. Dropping the slowest iteration from every run gives OFF
26,234.0 against ON 26,388.7, **+0.59%** — still above the floor, and the
honest lower bound.

The distribution-level evidence is stronger than either mean. Pooling all 15 OFF
and 15 ON iterations, **13 of the 15 ON iterations are faster than every one of
the 15 OFF iterations** (the two exceptions are 26,104.8, the cold first
iteration of pair 1, and 26,243.2). ON also has visibly lower spread — pair 2's
five ON iterations span 0.10% — because the removed work included the variance,
not just the mean. ON wins all three pairs.
New best = **77.75%** of vLLM 33,994.5 on the best run, **77.55%** on the
three-pair mean.

**Profile confirmation, and why only part of the host saving converts.** Two
BS256/OSL128 captures back to back, same code, differing only in the gate,
analysed with `analyze_hostgaps.py gaps` on the forced LM-head anchor, mean of
devices 0 and 3:

| Metric | Gate OFF | Gate ON | Δ |
|---|---:|---:|---:|
| G1 (`index_elementwise` → `vectorized_gather`) | 555.5 µs | 499.4 µs | **−56.1 µs** |
| Total idle | 1717.5 µs | 1639.6 µs | −77.9 µs |
| Step period (median) | 8.8645 ms | 8.8050 ms | −59.5 µs |
| GPU-busy | 7.202 ms | 7.219 ms | +17 µs (noise) |
| Kernels per step | 1169 | 1169 | 0 |
| G3 (`CatArrayBatchedCopy` → `rmsnorm`) | 302.1 | 294.9 | −7.2 |
| G2A (`vectorized_elementwise` → `vectorized_gather`) | 65.8 | 61.5 | −4.3 |

The saving is host-side and localized: the kernel count is bit-identical, GPU-busy
does not move, and the idle reduction is concentrated in G1.

But note the conversion ratio, which is the interesting result and is *worse*
than QWEN-024's. The harness says 183.9 µs of host work was removed; G1 shrank
56.1 µs, total idle 77.9 µs, the profiled step period 59.5 µs, and e2e TPOT
94.9 µs. QWEN-024 removed 78.7 µs of host work and got 78.3 µs out of G1 — very
nearly 1:1 — because `update_requests` sits directly in the serial dependency
between the sampled tokens returning and the next step's launches. Only about a
third to a half of this change converts, which says most of
`post_process_requests` was already partly overlapped with GPU execution: it
runs from `async_bookkeep`, so the interpreter was working through it while the
device still had queued work, and only the exposed tail was ever on the critical
path. **The corollary for future levers: harness-measured host CPU savings are
an upper bound, and how much converts depends on where in the step the work
sits, not on how much of it there is.** The 2.0% projection from the
decision-gate measurement assumed a 1:1 conversion and was therefore too
optimistic by roughly a factor of two; the measured +0.98% is what it is worth.

**Ranked next levers, after QWEN-025.**

1. **Reduce graph node count** (HOSTGAP-S6 lever 4, unblocked by
   CGTRACE-CONTROL) — ~0.78 µs per removed node, so 100 nodes ≈ 0.85%. Now the
   top-ranked lever by expected value. **Risk medium**, sharply diminishing.
   Falsified if removing nodes does not move the profiled step period.
2. **Re-attribute the residual G1** — 499 µs/step remains in G1 but the whole
   measured host chain is now only 148 µs/step, so the majority of G1 is
   something HOSTGAP-S6's attribution folded in but this campaign has not
   isolated. Pure measurement, no code risk, and it is what decides whether any
   further host-side lever exists at all. Blocked on the nsys host-capture
   failure recorded above; unblock it by bisecting the host flag set against a
   short capture, starting with `--cpuctxsw=process-tree`.
3. **G3, the `CatArrayBatchedCopy` → `rmsnorm` gap** — 294.9 µs/step and
   *untouched by the last three levers*, which all moved G1. HOSTGAP-S6
   attributed it to a single 199 µs `cudaGraphLaunch` for a 1158-node graph,
   which makes it the same lever as graph node count rather than an independent
   one. **Risk unknown**; confirm the attribution still holds before treating it
   as separate.
4. **Move per-step request bookkeeping onto the GPU** (HOSTGAP-S6 lever 5) —
   ceiling now lower than the original ~9.2%, since QWEN-024 and QWEN-025
   together removed 224 µs/step of the host chain. **Risk high**, multi-session.

Note the residual G1 is still 499 µs/step against a measured host chain of only
148 µs/step, so most of what is left in G1 is *not* the post-sampling
bookkeeping chain any more. Attributing the rest of G1 is a prerequisite for
claiming any further ceiling on host-side work.

**Attempted and failed in the leftover time: re-attributing G1 at the new
config. Host-visibility capture is currently broken, and two plausible causes
are now ruled out.** Three captures with `profile_host_insession.sh` at the
QWEN-025 config, all unrecoverable in the same way — nsys finalization
deadlocked after the target exited, the bounded stop expired, and the
intermediate qdstrm failed to import with
`QuadDCommon::IncompleteFileException` from `verifyHeader`:

| Run | OSL | `--python-sampling` | qdstrm | Outcome |
|---|---:|---|---:|---|
| `prof/hosts9-1785107698` | 128 | on | 366 MB | import failed |
| `prof/hosts9b-1785109992` | 96 | on | 347 MB | import failed, stream byte-stable 30 s first |
| `prof/hosts9c-1785110873` | 96 | **off** | 219 MB | import failed |

*Hypothesis 1, copy-while-growing race — falsified.* The recovery path was
changed to wait for the source qdstrm size to hold steady before copying. The
second capture copied a stream that had been byte-stable for 30 s and still
failed `verifyHeader`, so the deadlocked nsys never writes the stream's
terminating section at all. The file is structurally incomplete, not truncated
in transit, and no amount of waiting recovers it. The settle loop is retained
only because it is harmless and documents the ruled-out cause.

*Hypothesis 2, event volume from the Python sampler — falsified.* Dropping
`--python-sampling` (new `PYTHON_SAMPLING=0` knob) cut the capture from 347 MB
to 219 MB and it failed identically. Neither the sampler nor sheer volume is the
trigger.

This does *not* affect any QWEN-025 result: the GPU-only script
(`profile_insession.sh`) finalized cleanly twice in this same session and
produced the `g2-off`/`g2-on` pair the profile table above is built from. The
deadlock is specific to the host-visibility flag set, and it is not universal —
HOSTGAP-S6 obtained a usable host trace from this same script on an earlier
allocation, so something environmental differs. Remaining untested suspects, in
order: `--cpuctxsw=process-tree`, then `--sample=process-tree` itself, then
`osrt` tracing. **Do not spend a third session's leftover time on blind retries
of the full flag set** — bisect the flags against a short capture first, since
each full attempt costs ~14 minutes.

## Optimization rules

1. Profile and classify before proposing a code change.
2. Change one performance variable at a time.
3. Preserve the fixed protocol.
4. Validate correctness before accepting throughput.
5. Revert regressions or correctness failures.
6. Record the result before beginning another experiment.
7. Stop when mcore meets or exceeds `VLLM-BASELINE`, then rerun both baselines
   once to confirm parity under identical conditions.
