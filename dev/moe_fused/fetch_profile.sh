#!/usr/bin/env bash
# Copy an nsys profile from a cog session's lustre run dir into ./nsys_trace/.
#
# Usage:
#   dev/moe_fused/fetch_profile.sh <session> <prof-dir|latest> <dest-basename>
#
# Examples:
#   dev/moe_fused/fetch_profile.sh qwen-comm latest              mcore_host_s6_osl128
#   dev/moe_fused/fetch_profile.sh qwen-comm s6-all-1785028240   mcore_s6_tuned_osl128
#
# ControlMaster is forced off: cog's multiplexed connection dies on large
# payloads on macOS (see dev/moe_fused/nomux/ssh).
set -euo pipefail

SESSION="${1:?session handle, e.g. qwen-comm}"
PROF_DIR="${2:?prof subdir under sessions/<session>/prof, or 'latest'}"
DEST_BASE="${3:?destination basename, e.g. mcore_host_s6_osl128}"

LOGIN="${COG_LOGIN_HOST:-oci-hsg-cs-001-login-02.nvidia.com}"
SCRATCH="${COG_SCRATCH:-/lustre/fsw/portfolios/coreai/users/shanmugamr/agents-space}"
REMOTE_PROF_ROOT="$SCRATCH/sessions/$SESSION/prof"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/nsys_trace"

SSH_OPTS=(-o ControlMaster=no -o ControlPath=none -o BatchMode=yes)
rsh() { ssh "${SSH_OPTS[@]}" "$LOGIN" "$@"; }

if [[ "$PROF_DIR" == "latest" ]]; then
  PROF_DIR="$(rsh "ls -1t '$REMOTE_PROF_ROOT' | head -1")"
  [[ -n "$PROF_DIR" ]] || { echo "ERROR: no prof dirs under $REMOTE_PROF_ROOT" >&2; exit 1; }
  echo "latest -> $PROF_DIR"
fi

REMOTE="$REMOTE_PROF_ROOT/$PROF_DIR"
echo "remote: $LOGIN:$REMOTE"
rsh "ls -lh '$REMOTE'/*.nsys-rep '$REMOTE'/*.sqlite" || {
  echo "ERROR: expected .nsys-rep and .sqlite in $REMOTE" >&2; exit 1; }

mkdir -p "$DEST_DIR"
for ext in nsys-rep sqlite; do
  src="$(rsh "ls -1 '$REMOTE'/*.$ext | head -1")"
  [[ -n "$src" ]] || { echo "ERROR: no .$ext in $REMOTE" >&2; exit 1; }
  echo "==> $DEST_DIR/$DEST_BASE.$ext"
  rsync -h --progress -e "ssh ${SSH_OPTS[*]}" "$LOGIN:$src" "$DEST_DIR/$DEST_BASE.$ext"
done

# Keep provenance next to the binaries; runs.md is the tracked index.
cat > "$DEST_DIR/$DEST_BASE.source.txt" <<EOF
session:   $SESSION
prof dir:  $REMOTE
host:      $LOGIN
fetched:   $(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF

ls -lh "$DEST_DIR/$DEST_BASE".*
