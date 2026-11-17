#!/usr/bin/env python3
"""Stream-audit peer replay integrity without loading large JSONL files."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from cmrg_calibrated import recall, sample_oracle_ids  # noqa: E402
from cmrg_controls import extract_final_chunks, iter_chunks, stable_key  # noqa: E402

DATASETS = (
    ("hotpot_heldout_k3", 3811),
    ("2wiki_k3", 950),
    ("musique_k3", 252),
)
METHODS_BY_REGIME = {
    "test": ("budget_only", "semantic_mmr", "adagres", "submodular", "gold_support_oracle"),
    "test_cebra_budget": (
        "reference_policy",
        "semantic_mmr",
        "adagres",
        "submodular",
        "gold_support_oracle",
    ),
}


def skip_rows(handle: TextIO, count: int) -> None:
    for _ in range(count):
        if not handle.readline():
            raise ValueError("reference replay ended before requested offset")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--regime", choices=tuple(METHODS_BY_REGIME), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods = METHODS_BY_REGIME[args.regime]
    report: dict[str, Any] = {"regime": args.regime, "datasets": {}}
    for tag, expected_rows in DATASETS:
        result_dir = args.root / args.regime / tag
        handles = {
            method: (result_dir / f"{method}_replay.jsonl").open("r", encoding="utf-8")
            for method in methods
        }
        reference_name = (
            f"calibrated_full_{tag}_replay.jsonl"
            if args.regime == "test_cebra_budget"
            else f"budget_only_{tag}_replay.jsonl"
        )
        reference_handle = (
            REPO_ROOT / "refine-logs" / "four_configs" / tag / reference_name
        ).open("r", encoding="utf-8")
        skip_rows(reference_handle, 1000)
        checked = 0
        try:
            iterators = [handles[method] for method in methods] + [reference_handle]
            for position, lines in enumerate(itertools.zip_longest(*iterators)):
                if position >= expected_rows:
                    if any(line not in (None, "") for line in lines[:-1]):
                        raise ValueError(f"generated replay has extra rows for {tag}")
                    break
                if any(line is None for line in lines):
                    raise ValueError(f"replay length mismatch at {tag}:{position}")
                method_rows = {
                    method: json.loads(lines[idx]) for idx, method in enumerate(methods)
                }
                reference = json.loads(lines[-1])
                source_index = 1000 + position
                baseline_updates = None
                for method, row in method_rows.items():
                    meta = row["peer_baseline"]
                    if meta["source_row_index"] != source_index:
                        raise ValueError(f"source index mismatch at {tag}:{position}:{method}")
                    if meta["used_tokens"] > meta["token_budget"]:
                        raise ValueError(f"token budget exceeded at {tag}:{position}:{method}")
                    if len(meta["selected_keys"]) > meta["chunk_cap"]:
                        raise ValueError(f"chunk cap exceeded at {tag}:{position}:{method}")
                    if len(meta["selected_keys"]) != len(set(meta["selected_keys"])):
                        raise ValueError(f"duplicate selection at {tag}:{position}:{method}")
                    final_keys = [stable_key(chunk) for chunk in extract_final_chunks(row)]
                    if final_keys != meta["selected_keys"]:
                        raise ValueError(f"final order mismatch at {tag}:{position}:{method}")
                    updates = []
                    for _, _, chunk in iter_chunks(row):
                        if chunk.get("label") == "<TERMINATE>" and chunk.get("keep_for_qa"):
                            raise ValueError(f"TERMINATE retained at {tag}:{position}:{method}")
                        updates.append(bool(chunk.get("use_for_next_query")))
                    if baseline_updates is None:
                        baseline_updates = updates
                    elif updates != baseline_updates:
                        raise ValueError(f"query update drift at {tag}:{position}:{method}")

                reference_method = methods[0]
                selected_reference = {
                    stable_key(chunk) for chunk in extract_final_chunks(method_rows[reference_method])
                }
                canonical_reference = {
                    stable_key(chunk) for chunk in extract_final_chunks(reference)
                }
                if selected_reference != canonical_reference:
                    raise ValueError(f"reference selected-set mismatch at {tag}:{position}")

                oracle_row = method_rows["gold_support_oracle"]
                oracle_score = recall(sample_oracle_ids(oracle_row), extract_final_chunks(oracle_row))
                for method in methods[:-1]:
                    row = method_rows[method]
                    score = recall(sample_oracle_ids(row), extract_final_chunks(row))
                    if score > oracle_score + 1e-12:
                        raise ValueError(f"oracle bound violated at {tag}:{position}:{method}")
                checked += 1
        finally:
            for handle in handles.values():
                handle.close()
            reference_handle.close()
        if checked != expected_rows:
            raise ValueError(f"checked {checked}, expected {expected_rows} for {tag}")
        report["datasets"][tag] = {
            "rows": checked,
            "source_window": [1000, 1000 + checked],
            "budget_caps": "pass",
            "terminate_exclusion": "pass",
            "stable_deduplication": "pass",
            "query_update_independence": "pass",
            "reference_selected_set": "pass",
            "gold_support_upper_bound": "pass",
        }

    report["verdict"] = "PASS"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
