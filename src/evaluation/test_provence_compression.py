from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from provence_compression import (  # noqa: E402
    apply_nonempty_fallback,
    answer_in_text,
    choose_threshold,
    load_rows,
    normalize_answer_text,
)


class ProvenceCompressionTest(unittest.TestCase):
    def test_answer_normalization_matches_span_semantics(self) -> None:
        self.assertEqual(normalize_answer_text("The Savannah River Site"), "savannah river site")
        self.assertEqual(answer_in_text("The Savannah River Site", "At Savannah River Site."), 1.0)

    def test_calibration_boundary_is_enforced_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "crosses locked boundary"):
            load_rows(
                Path("does-not-need-to-exist.jsonl"),
                offset=999,
                limit=2,
                split="calibration",
                calibration_rows=1000,
            )

    def test_test_boundary_is_enforced_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "test run must start"):
            load_rows(
                Path("does-not-need-to-exist.jsonl"),
                offset=999,
                limit=1,
                split="test",
                calibration_rows=1000,
            )

    def test_threshold_selection_prefers_shortest_feasible_context(self) -> None:
        sweep = {
            "0": {
                "original_answer_in_context_span": 0.8,
                "compressed_answer_in_context_span": 0.8,
                "original_empty_contexts": 0,
                "compressed_empty_contexts": 0,
                "aggregate_word_ratio": 1.0,
            },
            "0.1": {
                "original_answer_in_context_span": 0.8,
                "compressed_answer_in_context_span": 0.795,
                "original_empty_contexts": 0,
                "compressed_empty_contexts": 0,
                "aggregate_word_ratio": 0.5,
            },
            "0.5": {
                "original_answer_in_context_span": 0.8,
                "compressed_answer_in_context_span": 0.75,
                "original_empty_contexts": 0,
                "compressed_empty_contexts": 0,
                "aggregate_word_ratio": 0.2,
            },
        }
        threshold, selection = choose_threshold(sweep, max_aic_drop_pp=1.0)
        self.assertEqual(threshold, 0.1)
        self.assertFalse(selection["used_fallback"])

    def test_nonempty_fallback_restores_only_first_original_chunk(self) -> None:
        output, count = apply_nonempty_fallback(
            [["first", "second"], ["kept"]],
            [["", ""], ["compressed"]],
        )
        self.assertEqual(output, [["first", ""], ["compressed"]])
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
