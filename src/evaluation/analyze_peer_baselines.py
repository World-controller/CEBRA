#!/usr/bin/env python3
"""Aggregate and statistically compare frozen-trajectory peer baselines."""

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
METHOD_NAMES = {
    "budget_only": "Budget-only",
    "reference_policy": "CEBRA",
    "semantic_mmr": "Semantic MMR",
    "adagres": "AdaGReS",
    "submodular": "Budgeted Submodular Packing",
    "gold_support_oracle": "Gold-support Oracle",
}


def paired_stats(treatment: list[float], control: list[float], seed: int = 42) -> dict[str, Any]:
    left = np.asarray(treatment, dtype=float)
    right = np.asarray(control, dtype=float)
    if left.shape != right.shape:
        raise ValueError(f"paired arrays differ: {left.shape} versus {right.shape}")
    diff = left - right
    if np.any(diff != 0):
        p_value = max(
            float(stats.wilcoxon(left, right, zero_method="wilcox").pvalue),
            float(np.finfo(float).tiny),
        )
    else:
        p_value = 1.0
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(10_000, dtype=float)
    for start in range(0, len(bootstrap), 500):
        stop = min(start + 500, len(bootstrap))
        indices = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        bootstrap[start:stop] = diff[indices].mean(axis=1)
    return {
        "n": int(len(diff)),
        "mean_difference": float(diff.mean()),
        "ci95": [float(np.percentile(bootstrap, 2.5)), float(np.percentile(bootstrap, 97.5))],
        "wilcoxon_p": p_value,
        "changed_rows": int(np.count_nonzero(diff)),
        "improved_rows": int(np.count_nonzero(diff > 0)),
        "degraded_rows": int(np.count_nonzero(diff < 0)),
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


def compact_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "per_row"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {
        "protocol": {
            "primary_comparison": (
                "each peer method receives the per-question token budget and chunk cap materialized by CEBRA"
            ),
            "secondary_comparison": (
                "each peer method receives the per-question token budget and chunk cap materialized by Budget-only B=3"
            ),
            "statistics": "paired per-question support recall; 10,000 bootstrap resamples; Wilcoxon; Holm across datasets per comparison",
        },
        "datasets": {},
    }
    competitors = ("semantic_mmr", "adagres", "submodular")
    raw_p: dict[str, list[float]] = {method: [] for method in competitors}

    for tag, display_name, expected_n in DATASETS:
        primary_path = args.root / "test_cebra_budget" / tag / "metrics.json"
        secondary_path = args.root / "test" / tag / "metrics.json"
        calibration_path = args.root / "calibration_cebra_budget" / tag / "metrics.json"
        primary = json.loads(primary_path.read_text(encoding="utf-8"))
        secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

        if primary["protocol"]["split"] != "test" or primary["protocol"]["offset"] != 1000:
            raise ValueError(f"invalid primary test window for {tag}")
        if primary["protocol"]["limit"] != expected_n:
            raise ValueError(f"unexpected test size for {tag}")
        if primary["protocol"]["mmr_lambda"] != calibration["protocol"]["mmr_lambda"]:
            raise ValueError(f"MMR lambda was not locked from calibration for {tag}")

        entry: dict[str, Any] = {
            "display_name": display_name,
            "locked_mmr_lambda": primary["protocol"]["mmr_lambda"],
            "cebra_budget": {
                method: compact_metrics(result) for method, result in primary["results"].items()
            },
            "budget_only_budget": {
                method: compact_metrics(result) for method, result in secondary["results"].items()
            },
            "paired_cebra_minus_peer": {},
        }
        cebra_recall = primary["results"]["reference_policy"]["per_row"]["support_recall"]
        for method in competitors:
            comparison = paired_stats(
                cebra_recall,
                primary["results"][method]["per_row"]["support_recall"],
            )
            entry["paired_cebra_minus_peer"][method] = comparison
            raw_p[method].append(comparison["wilcoxon_p"])
        report["datasets"][tag] = entry

    for method in competitors:
        adjusted = holm_adjust(raw_p[method])
        for (tag, _, _), value in zip(DATASETS, adjusted):
            report["datasets"][tag]["paired_cebra_minus_peer"][method][
                "holm_p_across_datasets"
            ] = value

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Peer-baseline comparison",
        "",
        "Primary protocol: every peer selector receives the exact per-question chunk cap and MiniLM-token upper bound materialized by CEBRA. Realized cost may be lower.",
        "",
        "| Dataset | Method | Support recall (%) | Mean chunks | Mean words | AiC (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    ordered_methods = (
        "reference_policy",
        "semantic_mmr",
        "adagres",
        "submodular",
        "gold_support_oracle",
    )
    for tag, display_name, _ in DATASETS:
        results = report["datasets"][tag]["cebra_budget"]
        for method in ordered_methods:
            value = results[method]
            lines.append(
                f"| {display_name} | {METHOD_NAMES[method]} | "
                f"{value['support_recall'] * 100:.2f} | {value['mean_chunks']:.2f} | "
                f"{value['mean_words']:.1f} | {value['answer_in_context'] * 100:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Paired support-recall differences",
            "",
            "| Dataset | Comparison | Difference (pp) | 95% CI (pp) | Holm-adjusted p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for tag, display_name, _ in DATASETS:
        comparisons = report["datasets"][tag]["paired_cebra_minus_peer"]
        for method in competitors:
            value = comparisons[method]
            lines.append(
                f"| {display_name} | CEBRA - {METHOD_NAMES[method]} | "
                f"{value['mean_difference'] * 100:+.2f} | "
                f"[{value['ci95'][0] * 100:+.2f}, {value['ci95'][1] * 100:+.2f}] | "
                f"{value['holm_p_across_datasets']:.3g} |"
            )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
