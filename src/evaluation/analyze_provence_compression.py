#!/usr/bin/env python3
"""Aggregate Provence calibration and test results across the three datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

DATASETS = (
    ("hotpot_heldout_k3", "HotpotQA", 3811),
    ("2wiki_k3", "2WikiMultiHopQA", 950),
    ("musique_k3", "MuSiQue", 252),
)


def paired_aic_stats(result: dict[str, Any], seed: int = 42) -> dict[str, Any]:
    per_row = result["per_row"]
    positions = [
        index for index, is_yes_no in enumerate(per_row["is_yes_no"]) if not is_yes_no
    ]
    original = np.asarray(
        [per_row["original_answer_in_context"][index] for index in positions],
        dtype=float,
    )
    compressed = np.asarray(
        [per_row["compressed_answer_in_context"][index] for index in positions],
        dtype=float,
    )
    diff = compressed - original
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, len(bootstrap), 500):
        stop = min(start + 500, len(bootstrap))
        indices = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        bootstrap[start:stop] = diff[indices].mean(axis=1)
    improved = int(np.count_nonzero(diff > 0))
    degraded = int(np.count_nonzero(diff < 0))
    discordant = improved + degraded
    p_value = (
        float(stats.binomtest(min(improved, degraded), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "n_span": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "ci95": [
            float(np.percentile(bootstrap, 2.5)),
            float(np.percentile(bootstrap, 97.5)),
        ],
        "mcnemar_exact_p": p_value,
        "improved_rows": improved,
        "degraded_rows": degraded,
        "unchanged_rows": int(len(diff) - discordant),
    }


def holm_adjust(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "per_row"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "protocol": {
            "composition": "Provence sentence pruning after frozen CEBRA chunk selection",
            "calibration_objective": (
                "minimum aggregate word ratio subject to span-only AiC drop <= 1 pp "
                "and no increase in empty contexts"
            ),
            "nonempty_fallback": (
                "restore the first original CEBRA-retained chunk only when sentence "
                "pruning empties an originally nonempty question context"
            ),
            "statistics": (
                "paired span-only AiC; 10,000 bootstrap resamples; exact McNemar; "
                "Holm correction across datasets"
            ),
        },
        "datasets": {},
    }
    raw_p: list[float] = []
    for tag, display_name, expected_n in DATASETS:
        calibration = json.loads(
            (args.root / "calibration_dense" / tag / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        test = json.loads(
            (args.root / "test" / tag / "metrics.json").read_text(encoding="utf-8")
        )
        if calibration["protocol"]["split"] != "calibration":
            raise ValueError(f"invalid calibration split for {tag}")
        if calibration["protocol"]["limit"] != 1000:
            raise ValueError(f"invalid calibration size for {tag}")
        if test["protocol"]["split"] != "test" or test["protocol"]["offset"] != 1000:
            raise ValueError(f"invalid test window for {tag}")
        if test["protocol"]["limit"] != expected_n:
            raise ValueError(f"unexpected test size for {tag}")
        if test["protocol"]["threshold_grid"]:
            raise ValueError(f"test threshold grid must be empty for {tag}")
        locked = float(calibration["protocol"]["threshold"])
        if float(test["protocol"]["threshold"]) != locked:
            raise ValueError(f"test threshold was not locked from calibration for {tag}")

        test_result = test["selected_result"]
        paired = paired_aic_stats(test_result)
        raw_p.append(paired["mcnemar_exact_p"])
        report["datasets"][tag] = {
            "display_name": display_name,
            "locked_threshold": locked,
            "calibration": compact(calibration["selected_result"]),
            "test": compact(test_result),
            "paired_aic": paired,
            "timing_seconds": test["timing_seconds"],
            "model_blocks": test["model_blocks"],
        }

    for (tag, _, _), adjusted in zip(DATASETS, holm_adjust(raw_p)):
        report["datasets"][tag]["paired_aic"]["holm_p_across_datasets"] = adjusted

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Provence after CEBRA",
        "",
        "Each threshold is selected on the first 1,000 calibration rows and then locked for the disjoint test segment.",
        "",
        "| Dataset | Locked threshold | Original words | Compressed words | Compression (%) | Original AiC (%) | Compressed AiC (%) | Delta (pp) | Fallback rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag, display_name, _ in DATASETS:
        entry = report["datasets"][tag]
        result = entry["test"]
        lines.append(
            f"| {display_name} | {entry['locked_threshold']:.6g} | "
            f"{result['mean_original_words']:.1f} | {result['mean_compressed_words']:.1f} | "
            f"{result['compression_rate'] * 100:.2f} | "
            f"{result['original_answer_in_context_span'] * 100:.2f} | "
            f"{result['compressed_answer_in_context_span'] * 100:.2f} | "
            f"{result['answer_in_context_delta_span'] * 100:+.2f} | "
            f"{result['nonempty_fallback_contexts']} |"
        )

    lines.extend(
        [
            "",
            "## Paired span-only answer survival",
            "",
            "| Dataset | Delta (pp) | 95% bootstrap CI (pp) | Improved | Degraded | Holm-adjusted p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for tag, display_name, _ in DATASETS:
        paired = report["datasets"][tag]["paired_aic"]
        lines.append(
            f"| {display_name} | {paired['mean_difference'] * 100:+.2f} | "
            f"[{paired['ci95'][0] * 100:+.2f}, {paired['ci95'][1] * 100:+.2f}] | "
            f"{paired['improved_rows']} | {paired['degraded_rows']} | "
            f"{paired['holm_p_across_datasets']:.3g} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
