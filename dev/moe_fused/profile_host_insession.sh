#!/usr/bin/env bash
# Host-visibility variant of profile_insession.sh.
#
# Identical workload/protocol to profile_insession.sh (BS256, OSL128, TP1/PP1/
# EP4/ETP1, nvls dispatcher, vllm grouped-GEMM, full_iteration_inference CUDA
# graphs), but the nsys trace adds OS-runtime, CPU-sampling and Python-sampling
# so the host gaps between graph replays can be attributed to real call sites.
set -uo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
PROFILE_BS="${PROFILE_BS:-256}"
PROFILE_OSL="${PROFILE_OSL:-128}"

SCRATCH=/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space
CKPT=$SCRATCH/checkpoints/qwen3-30b-a3b-mcore
TOKENIZER=$SCRATCH/checkpoints/qwen3-30b-a3b-hf
export CHECKPOINT_LOAD_PATH="$CKPT"

VENV=$SCRATCH/envs/megatron_lm/dd356431262b5db4/.venv
PYBIN=$VENV/bin/python

SESSION_HANDLE="${SESSION_HANDLE:-qwen-comm}"
RUN_DIR=$SCRATCH/sessions/${SESSION_HANDLE}/prof/${PROF_TAG:-host}-$(date +%s)
mkdir -p "$RUN_DIR/torchrun_logs"
EXTRA="$RUN_DIR/extra_pkgs"; mkdir -p "$EXTRA"
export PYTHONPATH="$EXTRA:${PYTHONPATH:-}"
$PYBIN -m pip install --quiet --no-cache-dir --target="$EXTRA" hypercorn aiohttp 2>/dev/null || true
echo "RUN_DIR=$RUN_DIR"; echo "PYBIN=$PYBIN"
echo "GATES: FUSE_FC1_ACT=${MCORE_FUSE_FC1_ACT:-<unset>} FUSED_ALIGN=${MCORE_MOE_FUSED_ALIGN:-<unset>} GEMM_TUNE=${MCORE_MOE_GEMM_TUNE:-<unset>} FUSED_COUNT=${MCORE_MOE_FUSED_COUNT:-<unset>} SUM_FAST=${MCORE_MOE_SUM_FAST:-<unset>} ROUTER_FUSED_TOPK=${MCORE_ROUTER_FUSED_TOPK:-<unset>} FUSED_SCATTER=${MCORE_MOE_FUSED_SCATTER:-<unset>}"

SERVER_LOG="$RUN_DIR/server.log"
PROF_BASE="$RUN_DIR/mcore_host_profile"

command -v nsys >/dev/null 2>&1 || { echo "ERROR: nsys not found"; exit 3; }
nsys --version

# Host-visibility trace flags. Validated against a trivial target first: an
# unsupported flag aborts the launch, and a failed launch costs the whole run.
# PYTHON_SAMPLING=0 drops the Python sampler, which is the largest contributor
# to the event count; two 350 MB captures with it on deadlocked in nsys
# finalization and left an unterminated qdstrm that QdstrmImporter rejects.
# CPUCTXSW=none is the next step of that bisection after PYTHON_SAMPLING=0.
PYTHON_SAMPLING="${PYTHON_SAMPLING:-1}"
CPUCTXSW="${CPUCTXSW:-process-tree}"
HOST_FLAGS=(
  --trace=cuda,nvtx,osrt
  --sample=process-tree
  --backtrace=fp
  --samples-per-backtrace=1
  --cpuctxsw="$CPUCTXSW"
  --osrt-threshold=1000
  --cuda-graph-trace=node
  --force-overwrite=true
)
if [[ "$PYTHON_SAMPLING" == "1" ]]; then
  HOST_FLAGS+=(--python-sampling=true --python-sampling-frequency=1000)
fi
echo "===== nsys flag preflight ====="
if nsys profile "${HOST_FLAGS[@]}" -o "$RUN_DIR/flagcheck" \
     $PYBIN -c "print('flagcheck ok')" > "$RUN_DIR/flagcheck.log" 2>&1; then
  echo "PREFLIGHT OK (full host flags)"
else
  echo "PREFLIGHT FAILED with full host flags; log:"; tail -30 "$RUN_DIR/flagcheck.log"
  HOST_FLAGS=(
    --trace=cuda,nvtx,osrt
    --sample=process-tree
    --backtrace=fp
    --python-sampling=true
    --cuda-graph-trace=node
    --force-overwrite=true
  )
  if nsys profile "${HOST_FLAGS[@]}" -o "$RUN_DIR/flagcheck2" \
       $PYBIN -c "print('flagcheck ok')" > "$RUN_DIR/flagcheck2.log" 2>&1; then
    echo "PREFLIGHT OK (reduced host flags)"
  else
    echo "PREFLIGHT FAILED (reduced); log:"; tail -30 "$RUN_DIR/flagcheck2.log"
    HOST_FLAGS=(--trace=cuda,nvtx,osrt --sample=process-tree --cuda-graph-trace=node --force-overwrite=true)
    echo "FALLING BACK to minimal host flags"
  fi
fi
echo "NSYS_HOST_FLAGS=${HOST_FLAGS[*]}"

QWEN_MODEL_ARGS="--model-provider gpt --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --num-experts 128 --moe-router-topk 8 --moe-ffn-hidden-size 768 --moe-grouped-gemm --moe-router-dtype fp32 --moe-router-pre-softmax --moe-token-dispatcher-type alltoall --swiglu --normalization RMSNorm --norm-epsilon 1e-6 --position-embedding-type rope --rotary-base 1000000 --qk-layernorm --disable-bias-linear --untie-embeddings-and-output-weights --no-gradient-accumulation-fusion --make-vocab-size-divisible-by 1187 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --expert-model-parallel-size 4 --expert-tensor-parallel-size 1 --inference-moe-token-dispatcher-type nvls --inference-grouped-gemm-backend vllm"

nsys profile \
  "${HOST_FLAGS[@]}" \
  -o "$PROF_BASE" \
  $PYBIN -m torch.distributed.run --nproc-per-node 4 --log-dir "$RUN_DIR/torchrun_logs" \
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
  --inference-dynamic-batching-max-requests 256 \
  --inference-dynamic-batching-cuda-graph-sizing-distribution exponential \
  --inference-dynamic-batching-async-sched-mode legacy \
  --inference-dynamic-batching-sampling-backend torch \
  --enable-chunked-prefill \
  --seq-length 4096 --max-position-embeddings 4096 --inference-max-seq-length 4096 \
  --inference-dynamic-batching-buffer-size-gb 40 \
  --inference-dynamic-batching-num-cuda-graphs -1 \
  --cuda-graph-impl local \
  --cuda-graph-scope full_iteration_inference \
  --inference-use-synchronous-zmq-collectives \
  --inference-logging-step-interval 100 \
  --port 5000 \
  $QWEN_MODEL_ARGS \
  > "$SERVER_LOG" 2>&1 &
NSYS_PID=$!

READY=0
for i in $(seq 1 360); do
  if grep -q "Running on http://0.0.0.0:5000" "$SERVER_LOG" 2>/dev/null; then READY=1; break; fi
  if ! kill -0 $NSYS_PID 2>/dev/null; then echo "SERVER/NSYS DIED"; tail -120 "$SERVER_LOG"; exit 1; fi
  sleep 5
done
if [[ "$READY" != "1" ]]; then echo "SERVER TIMEOUT"; tail -120 "$SERVER_LOG"; kill $NSYS_PID 2>/dev/null; exit 1; fi
echo "===== SERVER READY (profiling) ====="

$PYBIN -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size 8 --dataset gsm8k --num-output-tokens 32 \
  --num-iters 1 --num-warmup-iters 0 || true

echo "===== PROFILED BENCHMARK BS=$PROFILE_BS OSL=$PROFILE_OSL ====="
$PYBIN -u tests/performance_tests/client/static_benchmark.py \
  --server-url "http://localhost:5000/v1" --model qwen \
  --batch-size $PROFILE_BS --dataset gsm8k --num-output-tokens $PROFILE_OSL \
  --num-iters 1 --num-warmup-iters 0 2>&1 | tee "$RUN_DIR/profile_bench.log"

echo "===== stopping nsys ====="
# nsys finalization can deadlock (both CLI and agent parked in futex_wait) after
# the target exits, so never `wait` unbounded here: that burns the whole
# allocation and leaves a 0-byte report. Bound the wait, then recover offline
# from the intermediate .qdstrm, which holds the full stream either way.
STOP_TIMEOUT="${NSYS_STOP_TIMEOUT:-900}"
kill -INT $NSYS_PID 2>/dev/null || true
STOPPED=0
for i in $(seq 1 "$STOP_TIMEOUT"); do
  if ! kill -0 $NSYS_PID 2>/dev/null; then STOPPED=1; break; fi
  sleep 1
done
if [[ "$STOPPED" == "1" ]]; then
  wait $NSYS_PID 2>/dev/null || true
  echo "NSYS_EXITED_CLEANLY=1"
else
  echo "NSYS_STOP_TIMEOUT after ${STOP_TIMEOUT}s (pid $NSYS_PID still alive) — will recover from qdstrm"
fi
ls -la "$RUN_DIR"/mcore_host_profile.* || true

if [[ ! -s "$PROF_BASE.nsys-rep" ]]; then
  echo "===== recovering report from intermediate qdstrm ====="
  rm -f "$PROF_BASE.nsys-rep"
  QDSTRM=$(ls -t "${TMPDIR:-/tmp}"/nsys-root/nsys-report-*.qdstrm /tmp/nsys-root/nsys-report-*.qdstrm 2>/dev/null | head -1)
  if [[ -z "$QDSTRM" ]]; then
    echo "ERROR: no intermediate qdstrm found; trace is unrecoverable"; exit 4
  fi
  # A deadlock can park nsys *before* it flushes, leaving a stream the importer
  # rejects as incomplete however long we wait. A second SIGINT is what actually
  # drives the flush (it recovered a 387 MB stream by hand once), so send one and
  # watch whether the file grows before deciding the trace is lost.
  QD_BEFORE=$(stat -c %s "$QDSTRM" 2>/dev/null || echo 0)
  echo "qdstrm before second SIGINT: $QD_BEFORE bytes"
  kill -INT $NSYS_PID 2>/dev/null || true
  sleep 5
  # The deadlocked nsys is still appending to this file. Copying it while it
  # grows yields an IncompleteFileException from QdstrmImporter (seen once at
  # 366 MB), so wait for the size to hold steady before taking the copy.
  QD_SETTLE="${NSYS_QDSTRM_SETTLE:-30}"
  QD_SETTLE_MAX="${NSYS_QDSTRM_SETTLE_MAX:-900}"
  prev=-1; stable=0
  for i in $(seq 1 "$QD_SETTLE_MAX"); do
    cur=$(stat -c %s "$QDSTRM" 2>/dev/null || echo 0)
    if [[ "$cur" == "$prev" && "$cur" != "0" ]]; then
      stable=$((stable + 1))
      [[ "$stable" -ge "$QD_SETTLE" ]] && break
    else
      stable=0
    fi
    prev="$cur"
    sleep 1
  done
  QD_AFTER=$(stat -c %s "$QDSTRM" 2>/dev/null || echo 0)
  echo "QDSTRM=$QDSTRM ($QD_AFTER bytes, size stable for ${stable}s)"
  if [[ "$stable" -lt "$QD_SETTLE" ]]; then
    echo "WARNING: qdstrm still growing after ${QD_SETTLE_MAX}s; import may fail"
  fi
  if [[ "$QD_AFTER" -le "$QD_BEFORE" ]]; then
    echo "WARNING: qdstrm did not grow after the second SIGINT ($QD_BEFORE -> $QD_AFTER)."
    echo "         The stream is probably unterminated and QdstrmImporter will reject it."
    echo "         Bisect the host flag set on the next attempt (start with --cpuctxsw)."
  fi
  cp "$QDSTRM" "$PROF_BASE.qdstrm"
  NSYS_ROOT=$(dirname "$(dirname "$(readlink -f "$(command -v nsys)")")")
  IMPORTER=$(ls "$NSYS_ROOT"/host-linux-*/QdstrmImporter 2>/dev/null | head -1)
  if [[ -z "$IMPORTER" ]]; then
    echo "ERROR: QdstrmImporter not found under $NSYS_ROOT"; exit 4
  fi
  # Run through the importer's own lib dir; its $ORIGIN rpath is unreliable here.
  LD_LIBRARY_PATH="$(dirname "$IMPORTER"):${LD_LIBRARY_PATH:-}" \
    "$IMPORTER" --input-file "$PROF_BASE.qdstrm" || { echo "ERROR: QdstrmImporter failed"; exit 4; }
  kill -9 $NSYS_PID 2>/dev/null || true
fi

if [[ ! -s "$PROF_BASE.nsys-rep" ]]; then
  echo "ERROR: no non-empty .nsys-rep to export"; exit 4
fi
echo "===== exporting sqlite ====="
if [[ ! -s "$PROF_BASE.sqlite" ]]; then
  nsys export --type sqlite --force-overwrite=true --output "$PROF_BASE.sqlite" "$PROF_BASE.nsys-rep"
else
  echo "sqlite already produced by report generation; skipping export"
fi
ls -la "$PROF_BASE.sqlite"
echo "===== host-table sanity check ====="
$PYBIN - "$PROF_BASE.sqlite" <<'PYEOF'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
names = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
want = ["CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
        "OSRT_API", "COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS",
        "PYTHON_SAMPLING_CALLCHAINS", "PYTHON_SAMPLING_STRING",
        "SCHED_EVENTS", "GENERIC_EVENTS"]
for t in want:
    if t in names:
        try:
            n = db.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        except Exception as e:
            n = f"err {e}"
        print(f"  {t:34s} rows={n}")
    else:
        print(f"  {t:34s} MISSING")
print("HOST_TABLES_PRESENT=" + str(any(t in names for t in ("OSRT_API","PYTHON_SAMPLING_CALLCHAINS","COMPOSITE_EVENTS"))))
PYEOF
echo "===== PROFILE DONE ====="
echo "REP=$PROF_BASE.nsys-rep"
echo "SQLITE=$PROF_BASE.sqlite"
