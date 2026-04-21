"""Tests for the ``normalize_by_query_length`` option on ColBERT contrastive losses."""

from __future__ import annotations

import pytest
import torch

from pylate import losses, models
from pylate.losses.contrastive import compute_query_length_denominator


def _build_model() -> models.ColBERT:
    return models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )


def _sentence_features(model: models.ColBERT):
    anchors = model.tokenize(
        ["fruits are healthy.", "chips are unhealthy."], is_query=True
    )
    positives = model.tokenize(
        ["fruits are good for health.", "chips are bad for health."], is_query=False
    )
    negatives = model.tokenize(
        ["fruits are bad for health.", "chips are good for health."], is_query=False
    )
    return [anchors, positives, negatives]


def test_compute_query_length_denominator_with_expansion() -> None:
    embeddings = torch.zeros(3, 7, 4)
    mask = torch.zeros(3, 7)
    denom = compute_query_length_denominator(
        query_embeddings=embeddings, query_mask=mask, do_query_expansion=True
    )
    assert denom == 7.0


def test_compute_query_length_denominator_without_expansion() -> None:
    embeddings = torch.zeros(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    denom = compute_query_length_denominator(
        query_embeddings=embeddings, query_mask=mask, do_query_expansion=False
    )
    assert denom.shape == (2, 1)
    assert torch.allclose(denom.squeeze(-1), torch.tensor([3.0, 5.0]))


def test_compute_query_length_denominator_clamps_to_one() -> None:
    embeddings = torch.zeros(1, 4, 2)
    mask = torch.zeros(1, 4)
    denom = compute_query_length_denominator(
        query_embeddings=embeddings, query_mask=mask, do_query_expansion=False
    )
    assert torch.allclose(denom, torch.tensor([[1.0]]))


@pytest.mark.skipif(
    torch.backends.mps.is_available(), reason="MPS is not supported by this test"
)
def test_contrastive_normalize_by_query_length_matches_manual_division() -> None:
    """With query expansion enabled (the default), the denominator is the padded
    query width. Enabling ``normalize_by_query_length`` must be equivalent to
    scaling the temperature by that width."""
    model = _build_model()
    sentence_features = _sentence_features(model)

    # Query expansion is on by default, so the query width is constant.
    query_width = sentence_features[0]["input_ids"].shape[1]

    torch.manual_seed(0)
    loss_default = losses.Contrastive(model=model, temperature=query_width)
    torch.manual_seed(0)
    reference = loss_default(sentence_features=sentence_features)

    torch.manual_seed(0)
    loss_norm = losses.Contrastive(
        model=model, temperature=1.0, normalize_by_query_length=True
    )
    torch.manual_seed(0)
    normalized = loss_norm(sentence_features=sentence_features)

    assert torch.allclose(reference, normalized, atol=1e-6)


@pytest.mark.skipif(
    torch.backends.mps.is_available(), reason="MPS is not supported by this test"
)
def test_contrastive_normalize_is_opt_in() -> None:
    """The flag must default to False and leave the loss untouched."""
    model = _build_model()
    sentence_features = _sentence_features(model)

    torch.manual_seed(0)
    baseline = losses.Contrastive(model=model)(sentence_features=sentence_features)
    torch.manual_seed(0)
    opt_in_off = losses.Contrastive(model=model, normalize_by_query_length=False)(
        sentence_features=sentence_features
    )
    assert torch.allclose(baseline, opt_in_off)
