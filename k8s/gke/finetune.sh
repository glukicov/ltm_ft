#!/usr/bin/env bash
# Run ltm-ft experiments on the GKE L4 and copy the results back into outputs/.
#
#   k8s/gke/finetune.sh "run --task checkerboard --name l4/checkerboard" "precision --name l4/precision/checkerboard"
#
# Each argument is one `ltm-ft` invocation, run in order in a single pod (the weights download once). collect.sh then
# copies the outputs back and writes outputs/<first --name segment>/timeline-<job>.json: Job creation, GPU node
# scale-up, image pull, container start and every command's duration, so cluster overhead is part of the measurement.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=k8s/gke/env.sh
source "$HERE/env.sh"
export JOB="${JOB:-ltm-ft-$(date -u +%Y%m%d-%H%M%S)}"
K() { kubectl --context "$CTX" -n "$NAMESPACE" "$@"; }
[[ $# -gt 0 ]] || { echo "usage: $0 \"run --task ... --name l4/...\" [...]" >&2; exit 2; }

script="$(mktemp)"
trap 'rm -f "$script"' EXIT
# shellcheck disable=SC2016 # single quotes on purpose: these lines expand inside the pod, not here
{
  echo 'set -euo pipefail'
  echo 'stamp() { echo "PHASE $1 $(date -u +%Y-%m-%dT%H:%M:%SZ)"; }'
  echo 'stamp container_started'
  echo 'python -c "import torch; print(\"cuda:\", torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
  i=0
  for cmd in "$@"; do
    i=$((i + 1))
    echo "stamp command_${i}_start"
    echo "ltm-ft $cmd 2>&1 | grep -v -E 'Fetching|Warning'"
    echo "stamp command_${i}_end"
  done
  echo 'stamp all_done'
  echo 'echo FINETUNE_DONE'
  echo 'sleep 3600'
} > "$script"
K create configmap ltm-ft-experiments --from-file=experiments.sh="$script" --dry-run=client -o yaml | K apply -f -

# shellcheck disable=SC2016 # envsubst takes the variable names literally
envsubst '${JOB} ${NAMESPACE} ${IMAGE}' < "$HERE/job-finetune.yaml" | K apply -f -
echo "job $JOB submitted"
exec "$HERE/collect.sh" "$JOB"
