#!/usr/bin/env bash
# Wait for a fine-tuning Job to finish, copy its outputs into outputs/, write its timeline, delete the Job.
# Safe to re-run (e.g. after a laptop disconnect) as long as the pod is still in its post-run sleep.
#
#   k8s/gke/collect.sh ltm-ft-20260914-090533
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
# shellcheck source=k8s/gke/env.sh
source "$HERE/env.sh"
JOB="${1:?usage: $0 <job-name>}"
K() { kubectl --context "$CTX" -n "$NAMESPACE" "$@"; }

until pod="$(K get pods -l job-name="$JOB" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)" && [[ -n "$pod" ]]; do
  sleep 5
done
echo "pod $pod"
last=""
# grep -c reads the whole stream: grep -q would exit early and SIGPIPE kubectl, which pipefail reports as failure.
until [[ "$(K logs "$pod" 2>/dev/null | grep -c FINETUNE_DONE || true)" -gt 0 ]]; do
  phase="$(K get pod "$pod" -o jsonpath='{.status.phase}')"
  if [[ "$phase" == Failed || "$phase" == Succeeded ]]; then
    echo "pod ended ($phase) before FINETUNE_DONE:" >&2
    K logs "$pod" --tail 30 >&2 || true
    exit 1
  fi
  now="$(K logs "$pod" 2>/dev/null | grep -E 'PHASE|test:|restored' | tail -1 || true)"
  [[ -n "$now" && "$now" != "$last" ]] && { echo "$now"; last="$now"; }
  sleep 20
done

mkdir -p "$ROOT/outputs"
log="$ROOT/outputs/.gke-$JOB.log" # untracked (dot-prefixed): the raw pod log, for debugging
K logs "$pod" > "$log"
grep -E "PHASE|cuda:" "$log"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
K cp "$pod:/work/outputs" "$staging/outputs"
cp -R "$staging/outputs/." "$ROOT/outputs/"

# The commands, as written into the pod script by finetune.sh ("ltm-ft <args> 2>&1 | ...").
K get configmap ltm-ft-experiments -o jsonpath='{.data.experiments\.sh}' |
  sed -n -E 's/^ltm-ft (.*) 2>&1 .*$/\1/p' > "$staging/commands.txt"
prefix="$(grep -o -E -- '--name [^ /]+' "$staging/commands.txt" | head -1 | awk '{print $2}')"
created="$(K get job "$JOB" -o jsonpath='{.metadata.creationTimestamp}')"
node="$(K get pod "$pod" -o jsonpath='{.spec.nodeName}')"
node_created="$(kubectl --context "$CTX" get node "$node" -o jsonpath='{.metadata.creationTimestamp}')"
K get events --field-selector "involvedObject.name=$pod" -o json > "$staging/events.json"
mkdir -p "$ROOT/outputs/${prefix:-l4}"
python3 "$HERE/timeline.py" --job "$JOB" --created "$created" --node-created "$node_created" --log "$log" \
  --events "$staging/events.json" --commands-file "$staging/commands.txt" > "$ROOT/outputs/${prefix:-l4}/timeline-$JOB.json"
echo "wrote outputs/${prefix:-l4}/timeline-$JOB.json"

K delete job "$JOB" --wait=false
echo "deleted job $JOB; the L4 node scales down after ~10 min idle"
