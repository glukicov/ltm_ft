"""Fast checks on a tiny, randomly initialised TabFM (no checkpoint download, CPU only)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
import torch
from tabfm import TabFMClassifier
from tabfm.src.pytorch.model import TabFM
from torch import nn

from ltm_ft.data import TRUE_PROB, Split, features, labels, make_split
from ltm_ft.evaluate import paired_bootstrap, predict_positive, score
from ltm_ft.finetune import FinetuneConfig, Float32Master, episode_loss, finetune, lr_multiplier, make_episode
from ltm_ft.model import load_trainable_state, prepare_for_finetuning, trainable_modules, trainable_state


def tiny_tabfm(dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    torch.manual_seed(0)
    model = TabFM(icl_num_blocks=3)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.2)
        for buffer in model.buffers():
            buffer.normal_(0, 1)
    return cast(nn.Module, model.to(dtype).eval())


@pytest.fixture(scope="module")
def split() -> Split:
    return make_split(n_train=120, n_val=30, n_test=40, seed=1)


def test_split_is_disjoint_deterministic_and_has_truth(split: Split) -> None:
    assert (len(split.train), len(split.val), len(split.test)) == (120, 30, 40)
    assert not set(split.train.index) & set(split.test.index)
    assert features(split.train).shape[1] == 21
    assert TRUE_PROB not in features(split.train).columns
    again = make_split(n_train=120, n_val=30, n_test=40, seed=1)
    assert again.test.equals(split.test)


def test_prepared_model_is_the_stock_model_and_masters_round_trip(split: Split) -> None:
    stock = predict_positive(tiny_tabfm(), split.train, split.test, n_estimators=2, seed=0)
    model = tiny_tabfm()
    params = prepare_for_finetuning(model, n_blocks=2, encoders=True)
    np.testing.assert_array_equal(predict_positive(model, split.train, split.test, 2, 0), stock)

    master = Float32Master(params)
    assert all(m.dtype == torch.float32 for m in master.masters)
    X, y = features(split.train), labels(split.train)
    episode = make_episode(model, X.iloc[:90], y[:90], X.iloc[90:], y[90:], seed=0, device="cpu")
    episode_loss(model, episode).backward()
    master.gradients_to_masters()
    assert all(p.grad is None for p in params) and all(m.grad is not None for m in master.masters)
    torch.optim.SGD(master.masters, lr=0.1).step()
    master.masters_to_params()
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(torch.equal(p, m.to(torch.bfloat16)) for p, m in zip(params, master.masters, strict=True))
    assert not np.array_equal(predict_positive(model, split.train, split.test, 2, 0), stock)


def test_prepare_freezes_all_but_the_trainable_modules() -> None:
    model = tiny_tabfm()
    params = prepare_for_finetuning(model, n_blocks=1)
    trainable = {id(p) for m in trainable_modules(model, 1) for p in m.parameters()}
    assert {id(p) for p in params} == trainable
    for p in model.parameters():
        assert p.requires_grad == (id(p) in trainable)
        assert p.dtype == torch.bfloat16


def test_episode_logits_match_the_stock_classifier(split: Split) -> None:
    """The training tensors are the ones inference builds: same logits, once the class shift is undone."""
    model = tiny_tabfm(torch.float32)
    X, y = features(split.train), labels(split.train)
    seed = 3
    episode = make_episode(model, X.iloc[:80], y[:80], X.iloc[80:], y[80:], seed=seed, device="cpu")

    clf = TabFMClassifier(model=model, n_estimators=2, random_state=seed).fit(X.iloc[:80], y[:80])
    stock = clf._predict_proba_internal(X.iloc[80:])  # [views, query, classes], shift already undone

    with torch.no_grad():
        out = model(episode.x, episode.y, episode.train_size, cat_mask=episode.cat_mask, d=episode.d)
    ours = np.roll(out[0, 80:, : episode.n_classes].numpy(), -episode.class_shift, axis=-1)
    np.testing.assert_allclose(ours, stock[episode.view], atol=1e-5)
    encoded = clf.y_encoder_.transform(y[80:].reshape(-1, 1)).ravel()
    np.testing.assert_array_equal((episode.target.numpy() - episode.class_shift) % episode.n_classes, encoded)


def test_one_step_changes_only_trainable_weights(split: Split) -> None:
    model = tiny_tabfm()
    params = prepare_for_finetuning(model, n_blocks=1)
    frozen_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    state_before = trainable_state(model)

    X, y = features(split.train), labels(split.train)
    episode = make_episode(model, X.iloc[:90], y[:90], X.iloc[90:], y[90:], seed=0, device="cpu")
    loss = episode_loss(model, episode)
    loss.backward()
    assert all(p.grad is not None for p in params)
    torch.optim.SGD(params, lr=1.0).step()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            assert torch.equal(p, frozen_before[name])
    assert any(not torch.equal(v, trainable_state(model)[k]) for k, v in state_before.items())
    load_trainable_state(model, state_before)
    assert all(torch.equal(v, trainable_state(model)[k]) for k, v in state_before.items())


def test_lr_schedule_warms_up_then_decays_to_zero() -> None:
    assert lr_multiplier(0, warmup_steps=5, max_steps=50) == pytest.approx(0.2)
    assert lr_multiplier(4, warmup_steps=5, max_steps=50) == pytest.approx(1.0)
    assert lr_multiplier(50, warmup_steps=5, max_steps=50) == pytest.approx(0.0)


def test_bootstrap_of_identical_predictions_is_zero(split: Split) -> None:
    y = labels(split.test)
    p = split.test[TRUE_PROB].to_numpy()
    for d in paired_bootstrap(y, p, p, n_resamples=50).values():
        assert d["delta"] == d["ci_low"] == d["ci_high"] == 0.0
    assert 0 < score(y, p).log_loss < 1


def test_noise_columns_leave_runners_and_labels_unchanged() -> None:
    plain = make_split(n_train=60, n_val=10, n_test=10, seed=2)
    noisy = make_split(n_train=60, n_val=10, n_test=10, seed=2, noise_columns=5)
    assert features(noisy.train).shape[1] == features(plain.train).shape[1] + 5
    assert noisy.train[plain.train.columns].equals(plain.train)


def test_encoder_variant_trains_encoders_and_still_predicts(split: Split) -> None:
    model = tiny_tabfm()
    params = prepare_for_finetuning(model, n_blocks=1, encoders=True)
    X, y = features(split.train), labels(split.train)
    episode = make_episode(model, X.iloc[:90], y[:90], X.iloc[90:], y[90:], seed=0, device="cpu")
    episode_loss(model, episode).backward()
    encoder = cast(nn.Module, cast(Any, model).cell_embedder)
    assert all(p.grad is not None for p in encoder.parameters() if p.requires_grad)
    assert len(params) > len(prepare_for_finetuning(tiny_tabfm(), n_blocks=1))
    p = predict_positive(model, split.train, split.test, n_estimators=2, seed=0)
    assert np.all((p > 0) & (p < 1))


def test_finetune_loop_records_history_and_efficiency(split: Split) -> None:
    model = tiny_tabfm()
    config = FinetuneConfig(
        n_trainable_blocks=1, max_steps=3, warmup_steps=1, context_size=60, query_size=20, eval_every=2, patience=5
    )
    result = finetune(model, split, config, device="cpu", log=lambda _: None)
    assert [h["step"] for h in result.history] == [0, 2, 3]
    assert len(result.step_seconds) == 3 and len(result.validation_seconds) == 3
    efficiency = result.efficiency()
    assert efficiency["steps"] == 3 and efficiency["peak_memory_gb"] is None  # CPU reports no accelerator memory
    assert all(not p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters())
