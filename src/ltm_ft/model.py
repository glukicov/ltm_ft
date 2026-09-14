"""Load TabFM and choose which of its weights fine-tuning may change.

TabFM's classifier is ~1.64B parameters, and 1.62B of them sit in the in-context-learning (ICL)
transformer: 24 blocks of width 2048 that attend from every row to the labelled rows. The cell,
column and row embedders in front of it are tiny by comparison.

Full fine-tuning with AdamW needs weights + gradients + two optimiser moments, ~26 GB in float32
for this model - more than a 24 GB laptop has. So we fine-tune the *last* `n_blocks` ICL blocks
plus the final norm and decoder head ("last layers", one of the partial fine-tuning schemes that
Rubachev et al. 2025, "On Finetuning Tabular Foundation Models", found to perform close to full
fine-tuning). Everything else stays frozen in bfloat16.

Optionally (`encoders=True`) the small cell/column/row encoders in front of the ICL transformer
are fine-tuned too - the "embeddings" variant in the same paper. That costs a backward pass through
all 24 ICL blocks, but only ~20M extra trainable parameters.

Precision matters more than you would expect. TabFM is designed to *compute* in bfloat16, and on
the checkerboard task the same checkpoint computed in float32 loses ~11 points of ROC AUC; even
`torch.autocast` changes its predictions. So the model is never upcast: trainable weights stay
bf16 inside the model, forward and backward passes are exactly the stock computation, and the
optimiser keeps float32 master copies (`Float32Master` in finetune.py) so that small AdamW updates
are not lost to bf16 rounding. The fine-tuned model is a plain bf16 TabFM that the stock
`TabFMClassifier` uses exactly like the original checkpoint.

Frozen parameters do not require gradients, so autograd records nothing (and keeps no
activations) until the first trainable module: no custom forward pass is needed.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
from torch import nn


def pick_device(device: str = "") -> str:
    """The requested device, else Apple-silicon GPU (MPS) when available, else CPU."""
    if device:
        return device
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_classifier(device: str, dtype: torch.dtype | None = torch.bfloat16) -> nn.Module:
    """The pre-trained TabFM classification checkpoint (first call downloads ~6.6 GB).

    `dtype=None` keeps the checkpoint's float32 weights (the library's "debugging / quality
    comparison" mode); the default bfloat16 is what the model is designed to compute in.
    `use_cache=False`: the tabfm loader otherwise hands every caller the same process-wide
    instance, and we are about to modify this one.
    """
    model = tabfm_v1_0_0.load(model_type="classification", device=device, dtype=dtype, use_cache=False)
    return cast(nn.Module, model)


def icl_blocks(model: nn.Module) -> nn.ModuleList:
    return cast(nn.ModuleList, cast(Any, model).icl_predictor.tf_icl.blocks)


ENCODER_MODULES = ("cell_embedder", "col_embedder", "row_interactor", "col_embedder_2", "row_interactor_2")


def trainable_modules(model: nn.Module, n_blocks: int, encoders: bool = False) -> list[nn.Module]:
    """The last `n_blocks` ICL blocks, the final RMSNorm and the decoder; with `encoders`, also the
    cell/column/row encoders that turn each table row into the ICL transformer's input (~20M parameters).
    """
    blocks = icl_blocks(model)
    if not 0 <= n_blocks <= len(blocks):
        raise ValueError(f"n_blocks must be in [0, {len(blocks)}], got {n_blocks}")
    icl = cast(Any, model).icl_predictor
    modules: list[nn.Module] = [*list(blocks)[len(blocks) - n_blocks :], icl.ln, icl.decoder]
    if encoders:
        modules += [cast(nn.Module, getattr(model, name)) for name in ENCODER_MODULES]
    return modules


def prepare_for_finetuning(model: nn.Module, n_blocks: int, encoders: bool = False) -> list[nn.Parameter]:
    """Freeze everything but `trainable_modules` (and, with `encoders`, the CLS tokens); return their parameters."""
    for p in model.parameters():
        p.requires_grad_(False)
    params = [p for module in trainable_modules(model, n_blocks, encoders) for p in module.parameters()]
    if encoders:
        params.append(cast(nn.Parameter, cast(Any, model).cls_tokens))
    for p in params:
        p.requires_grad_(True)
    # Nothing here uses dropout or batch statistics, so eval mode is also the training mode.
    model.eval()
    return params


def freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU copy of the trainable weights, for keeping the best step and saving a checkpoint."""
    return {name: p.detach().to("cpu", copy=True) for name, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)
