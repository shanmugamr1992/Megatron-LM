#!/usr/bin/env bash
# Wait for the qwen-host session to become RUNNING, then dispatch the
# host-visibility nsys capture detached so it survives the local client.
set -uo pipefail
source ~/.cog/setup.env.oci-hsg
export COG_MEGATRON_REPO=/Users/shanmugamr@nvidia.com/Megatron-LM
export PATH="$COG_MEGATRON_REPO/dev/moe_fused/nomux:$PATH"
HANDLE="${HANDLE:-qwen-host}"

for i in $(seq 1 200); do
  S=$(cog session status --session-handle "$HANDLE" --cluster-name "$COG_CLUSTER_NAME" 2>&1 \
      | python3 -c 'import json,sys
raw=sys.stdin.read(); i=raw.find("{")
try:
    d=json.loads(raw[i:])["session"]; print(d["state"])
except Exception: print("unknown")')
  echo "$(date -u +%T) state=$S"
  if [[ "$S" == "running" ]]; then break; fi
  if [[ "$S" == "failed" || "$S" == "preempted" || "$S" == "completed" ]]; then
    echo "SESSION DEAD ($S)"; exit 1
  fi
  sleep 15
done

CMD='MCORE_FUSE_FC1_ACT=1 MCORE_MOE_FUSED_ALIGN=1 MCORE_MOE_GEMM_TUNE=1 MCORE_MOE_FUSED_COUNT=1 MCORE_MOE_SUM_FAST=1 MCORE_ROUTER_FUSED_TOPK=1 MCORE_MOE_FUSED_SCATTER=1 SESSION_HANDLE='"$HANDLE"' PROF_TAG=hosts6 PROFILE_BS=256 PROFILE_OSL=128 bash dev/moe_fused/profile_host_insession.sh'

for attempt in 1 2 3 4 5; do
  RAW=$(cog session exec --session-handle "$HANDLE" --cluster-name "$COG_CLUSTER_NAME" \
        --repo "$COG_MEGATRON_REPO" --command "$CMD" --detach 2>&1)
  REQ=$(printf '%s' "$RAW" | python3 -c 'import json,sys
raw=sys.stdin.read(); i=raw.find("{")
try: print(json.loads(raw[i:])["execution"]["request_id"])
except Exception: print("")')
  if [[ -n "$REQ" ]]; then
    echo "DISPATCHED request_id=$REQ"
    echo "STDOUT=$COG_SCRATCH_ROOT/sessions/$HANDLE/exec/runs/$REQ/stdout.log"
    exit 0
  fi
  echo "dispatch attempt $attempt failed:"; printf '%s\n' "$RAW" | tail -c 800
  sleep 20
done
echo "ALL DISPATCH ATTEMPTS FAILED"; exit 1
