# Contributing

Issues and pull requests are welcome: another synthetic task, a different set of trainable
modules, a result that does not reproduce on your machine, or a fix.

## Setup and checks

```bash
uv sync                                        # project + dev tools (ruff, mypy, pytest), pinned in uv.lock
uv run ruff check . && uv run ruff format --check .
uv run mypy                                    # strict
uv run pytest                                  # tiny random TabFM on CPU: no checkpoint download
shellcheck -x k8s/gke/*.sh k8s/gke/kctl docker/requirements.sh
docker/requirements.sh                         # after changing dependencies: regenerate the image's requirements
```

CI runs the same checks on every push and pull request, with CPU-only PyTorch.

## Things that are easy to break by accident

1. **Step 0 must be the stock model.** `prepare_for_finetuning` only flips `requires_grad`; it never
   changes a dtype. TabFM is precision-sensitive: computing its encoders in float32 (or under
   `torch.autocast`) changes predictions by several points. `tests/test_ltm_ft.py` asserts that a
   prepared model predicts exactly like the untouched one, and that training episodes produce the
   same logits as `TabFMClassifier` does at inference.

2. **The test set is scored twice.** `ltm-ft tune` generates no test rows at all; choose
   hyperparameters with it. `ltm-ft run` scores test once before and once after fine-tuning.

3. **The results tables come from `outputs/`.** Each `run` writes `outputs/<name>/results.json`
   and `summary.md`, and the README numbers are copied from them. If you change the method, rerun
   and commit the new outputs.

4. **GKE runs cost money until `down.sh`.** The L4 pool scales to zero between Jobs, but the cluster,
   its system node and the image repository bill until `k8s/gke/down.sh` deletes them. Your project id
   comes from `gcloud` or `$PROJECT` and is never written into the repository; `timeline.py` keeps
   project, node and pod names out of committed JSON. The container runs the same `ltm-ft` commands
   as the laptop, so a code change needs `REBUILD=1 k8s/gke/up.sh` before the next GKE run.

5. **Weights are never committed.** `--save-weights` writes `*.safetensors` under `outputs/`, which
   is gitignored: fine-tuned weights derive from TabFM's non-commercial checkpoint.

## Licences

Contributions are accepted under the repository's [MIT licence](LICENSE). TabFM's weights are not
part of this repository and stay under Google's `tabfm-non-commercial-v1.0` licence.
