from __future__ import annotations

import unittest

import numpy as np

from peer_baselines import (
    Candidate,
    PreparedSample,
    mark_peer_keep_fields,
    select_adagres,
    select_gold_support_oracle,
    select_semantic_mmr,
    select_submodular,
)


def candidate(
    key: str,
    order: int,
    tokens: int,
    embedding: list[float],
    relevance: float,
    oracle: str | None = None,
) -> Candidate:
    vector = np.asarray(embedding, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return Candidate(
        key=key,
        chunk={"id": key, "title": key, "text": f"evidence {key}"},
        order=order,
        text=f"{key} evidence",
        token_count=tokens,
        terms={key, "question"},
        source=key,
        embedding=vector,
        query_similarity=relevance,
        lexical_relevance=relevance,
        covered_oracles={oracle} if oracle else set(),
    )


class PeerBaselineSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        candidates = [
            candidate("a", 0, 4, [1.0, 0.0], 1.0, "a"),
            candidate("b", 1, 4, [0.0, 1.0], 0.8, "b"),
            candidate("c", 2, 7, [0.9, 0.1], 0.7),
        ]
        self.prepared = PreparedSample(
            row_index=0,
            sample={"query": "question", "answer": "a", "oracle": ["a", "b"]},
            question="question",
            question_terms={"question"},
            candidates=candidates,
            query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
            token_budget=8,
            chunk_cap=2,
            budget_keys=["a", "b"],
        )

    def assert_budget(self, indices: list[int]) -> None:
        self.assertLessEqual(len(indices), self.prepared.chunk_cap)
        self.assertLessEqual(
            sum(self.prepared.candidates[idx].token_count for idx in indices),
            self.prepared.token_budget,
        )

    def test_all_selectors_respect_budgets(self) -> None:
        selectors = (
            lambda: select_semantic_mmr(self.prepared, lam=0.7),
            lambda: select_adagres(self.prepared),
            lambda: select_submodular(self.prepared),
            lambda: select_gold_support_oracle(self.prepared),
        )
        for selector in selectors:
            with self.subTest(selector=selector):
                indices, _, _ = selector()
                self.assert_budget(indices)

    def test_gold_oracle_maximizes_support_recall(self) -> None:
        indices, _, details = select_gold_support_oracle(self.prepared)
        self.assertEqual({self.prepared.candidates[idx].key for idx in indices}, {"a", "b"})
        self.assertEqual(details["oracle_support_recall"], 1.0)

    def test_mmr_prefers_complementary_second_chunk(self) -> None:
        indices, _, _ = select_semantic_mmr(self.prepared, lam=0.5)
        self.assertEqual(indices, [0, 1])

    def test_marking_skips_terminate_occurrence_of_selected_id(self) -> None:
        sample = {
            "0": {
                "docs": [
                    {"id": "x", "label": "<TERMINATE>", "info": ""},
                    {"id": "x", "label": "<CONTINUE>", "info": "next"},
                ]
            }
        }
        replay = mark_peer_keep_fields(sample, {"x"}, {"x": 0.8})
        self.assertFalse(replay["0"]["docs"][0]["keep_for_qa"])
        self.assertTrue(replay["0"]["docs"][1]["keep_for_qa"])


if __name__ == "__main__":
    unittest.main()
