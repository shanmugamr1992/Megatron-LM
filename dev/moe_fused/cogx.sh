#!/usr/bin/env bash
# Local helper: run a command in the cog session and print its stdout/stderr.
#   bash dev/moe_fused/cogx.sh <session-handle> <timeout-sec> '<command>'
#
# Uses `--detach` + polling rather than a foreground wait: cog's foreground
# waiter ships a multi-kilobyte script over ssh, which trips macOS's ssh
# multiplexing ("mm_send_fd: sendmsg(1): Message too long"). The detached
# dispatch and the exec-status poll are both short enough to survive it.
set -uo pipefail
source ~/.cog/setup.env.oci-hsg
export COG_MEGATRON_REPO=/Users/shanmugamr@nvidia.com/Megatron-LM
export PATH="$COG_MEGATRON_REPO/dev/moe_fused/nomux:$PATH"
HANDLE="$1"; TIMEOUT="$2"; shift 2; CMD="$*"

RAW=$(cog session exec --session-handle "$HANDLE" --cluster-name "$COG_CLUSTER_NAME" \
  --repo "$COG_MEGATRON_REPO" --command "$CMD" --detach 2>&1)
REQ=$(printf '%s' "$RAW" | python3 -c 'import json,sys
raw=sys.stdin.read(); i=raw.find("{")
try:
    print(json.loads(raw[i:])["execution"]["request_id"])
except Exception:
    print(""); sys.stderr.write(raw[-1500:])')
if [[ -z "$REQ" ]]; then echo "### DISPATCH FAILED"; exit 1; fi
RUNDIR="$COG_SCRATCH_ROOT/sessions/$HANDLE/exec/runs/$REQ"
echo "### request_id=$REQ"
echo "### stdout=$RUNDIR/stdout.log"

DEADLINE=$(( $(date +%s) + TIMEOUT ))
EXITC=""
while [[ $(date +%s) -le $DEADLINE ]]; do
  EXITC=$(ssh -o BatchMode=yes "$COG_SSH_HOST" "cat '$RUNDIR/exit_code' 2>/dev/null" | tr -d '[:space:]')
  [[ -n "$EXITC" ]] && break
  sleep "${POLL_SEC:-20}"
done
echo "### exit_code=${EXITC:-TIMEOUT}"
ssh -o BatchMode=yes "$COG_SSH_HOST" \
  "tail -c ${TAIL_BYTES:-40000} '$RUNDIR/stdout.log' 2>/dev/null; \
   echo '### ---- STDERR ----'; tail -c ${ERR_BYTES:-6000} '$RUNDIR/stderr.log' 2>/dev/null"
[[ "${EXITC:-1}" == "0" ]] || exit 1
