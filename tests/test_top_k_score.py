"""Tests for the ``top_k_score`` parameter.

``top_k_score`` changes the MaxSim aggregation so that only the highest ``k``
per-query-token MaxSim values are summed instead of all of them. This module
checks:

* The scoring functions in ``pylate.scores`` behave correctly.
* ``rank.rerank`` and ``retrieve.ColBERT.retrieve`` propagate the parameter.
* ``models.ColBERT(top_k_score=...)`` patches the internal similarity
  functions and round-trips through save / load.
"""

from __future__ import annotations

import os
import shutil
import uuid

import torch

from pylate import indexes, models, rank, retrieve, scores


def _fixed_embeddings():
    queries = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[2.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    documents = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 2.0], [3.0, 0.0]],
        ]
    )
    return queries, documents


def test_colbert_scores_top_k_score_matches_manual_topk() -> None:
    q, d = _fixed_embeddings()

    full = scores.colbert_scores(q, d)
    tk1 = scores.colbert_scores(q, d, top_k_score=1)
    tk2 = scores.colbert_scores(q, d, top_k_score=2)

    # MaxSim values per (query, doc, query_token)
    maxsim = torch.einsum("ash,bth->abst", q, d).max(dim=-1).values

    expected_full = maxsim.sum(dim=-1)
    expected_top1 = maxsim.topk(1, dim=-1).values.sum(dim=-1)
    expected_top2 = maxsim.topk(2, dim=-1).values.sum(dim=-1)

    torch.testing.assert_close(full, expected_full)
    torch.testing.assert_close(tk1, expected_top1)
    torch.testing.assert_close(tk2, expected_top2)


def test_colbert_scores_top_k_score_clamps_to_num_tokens() -> None:
    q, d = _fixed_embeddings()
    full = scores.colbert_scores(q, d)
    # top_k_score larger than num query tokens must degrade to the full sum.
    huge = scores.colbert_scores(q, d, top_k_score=999)
    torch.testing.assert_close(full, huge)


def test_colbert_scores_pairwise_top_k_score() -> None:
    q, d = _fixed_embeddings()

    full = scores.colbert_scores_pairwise(q, d)
    tk1 = scores.colbert_scores_pairwise(q, d, top_k_score=1)

    # Pairwise: only diagonal (query i vs doc i).
    expected_full = torch.einsum("ash,bth->abst", q, d).max(dim=-1).values
    expected_full = torch.stack(
        [expected_full[i, i].sum() for i in range(q.shape[0])]
    )
    expected_top1 = torch.einsum("ash,bth->abst", q, d).max(dim=-1).values
    expected_top1 = torch.stack(
        [expected_top1[i, i].topk(1).values.sum() for i in range(q.shape[0])]
    )

    torch.testing.assert_close(full, expected_full)
    torch.testing.assert_close(tk1, expected_top1)


def test_colbert_kd_scores_top_k_score() -> None:
    torch.manual_seed(0)
    q = torch.randn(2, 4, 8)
    d = torch.randn(2, 3, 5, 8)  # (batch, n_ways, doc_tokens, dim)

    full = scores.colbert_kd_scores(q, d)
    tk2 = scores.colbert_kd_scores(q, d, top_k_score=2)

    # All values with top_k_score=k must be <= full sum.
    assert (tk2 <= full + 1e-5).all()
    # And strictly smaller for at least one entry given random inputs.
    assert (tk2 < full - 1e-5).any()


def test_rerank_top_k_score() -> None:
    q, d = _fixed_embeddings()
    documents_ids = [["a", "b"], ["c", "d"]]

    full = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=q,
        documents_embeddings=d.unsqueeze(0).expand(2, -1, -1, -1).clone(),
    )
    tk1 = rank.rerank(
        documents_ids=documents_ids,
        queries_embeddings=q,
        documents_embeddings=d.unsqueeze(0).expand(2, -1, -1, -1).clone(),
        top_k_score=1,
    )

    # Same document ids returned (order may change, but sets match).
    for full_q, tk_q in zip(full, tk1):
        assert {r["id"] for r in full_q} == {r["id"] for r in tk_q}

    # Scores must differ between the two modes for at least one query.
    any_diff = False
    for full_q, tk_q in zip(full, tk1):
        full_scores = {r["id"]: r["score"] for r in full_q}
        tk_scores = {r["id"]: r["score"] for r in tk_q}
        for doc_id in full_scores:
            if abs(full_scores[doc_id] - tk_scores[doc_id]) > 1e-5:
                any_diff = True
    assert any_diff, "top_k_score must produce different scores from the full sum"


def test_colbert_model_top_k_score_patches_similarity(tmp_path) -> None:
    """The model must swap its similarity functions when top_k_score is set,
    and persist the value across save / load."""
    model = models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        top_k_score=2,
    )

    assert model.top_k_score == 2

    # _similarity should be a functools.partial wrapping colbert_scores with
    # top_k_score bound to 2.
    assert getattr(model._similarity, "keywords", {}).get("top_k_score") == 2
    assert (
        getattr(model._similarity_pairwise, "keywords", {}).get("top_k_score") == 2
    )

    save_dir = os.path.join(tmp_path, "colbert_top_k")
    model.save(save_dir)

    reloaded = models.ColBERT(model_name_or_path=save_dir, device="cpu")
    assert reloaded.top_k_score == 2
    assert getattr(reloaded._similarity, "keywords", {}).get("top_k_score") == 2


def test_retrieve_top_k_score_end_to_end() -> None:
    """End-to-end smoke test: rerank via retrieve.ColBERT with top_k_score."""
    random_hash = uuid.uuid4().hex
    index_folder = f"test_indexes_{random_hash}"

    model = models.ColBERT(
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )

    documents = [
        "Apples are a common fruit.",
        "Neural networks learn from data.",
        "Bananas are yellow fruits.",
    ]
    documents_embeddings = model.encode(
        documents, is_query=False, convert_to_tensor=True
    )
    queries_embeddings = model.encode(
        ["fruit", "machine learning"], is_query=True, convert_to_tensor=True
    )

    try:
        index = indexes.Voyager(
            index_folder=index_folder,
            index_name="top_k_score",
            override=True,
            embedding_size=128,
        )
        index.add_documents(
            documents_ids=["a", "b", "c"],
            documents_embeddings=documents_embeddings,
        )
        retriever = retrieve.ColBERT(index=index)

        results_full = retriever.retrieve(queries_embeddings=queries_embeddings, k=3)
        results_tk = retriever.retrieve(
            queries_embeddings=queries_embeddings, k=3, top_k_score=2
        )

        assert len(results_full) == len(results_tk) == 2

        # Both modes return the same set of document ids (we index 3 docs, k=3).
        for full_q, tk_q in zip(results_full, results_tk):
            assert {r["id"] for r in full_q} == {r["id"] for r in tk_q}

        # At least one score must differ between the two modes.
        any_diff = False
        for full_q, tk_q in zip(results_full, results_tk):
            full_scores = {r["id"]: r["score"] for r in full_q}
            tk_scores = {r["id"]: r["score"] for r in tk_q}
            for doc_id in full_scores:
                if abs(full_scores[doc_id] - tk_scores[doc_id]) > 1e-5:
                    any_diff = True
        assert any_diff
    finally:
        shutil.rmtree(index_folder, ignore_errors=True)
