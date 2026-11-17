#!/usr/bin/env python3
"""Audit Provence replays for split locking and frozen-trajectory invariants."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from cmrg_controls import extract_final_chunks, iter_chunks, stable_key  # noqa: E402

DATASETS = (
    ("hotpot_heldout_k3", 3811),
    ("2wiki_k3", 950),
    ("musique_k3", 252),
)


def source_slice(path: Path, offset: int, limit: int) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        emitted = 0
        for index, line in enumerate(handle):
            if index < offset:
                continue
            if emitted >= limit:
                break
            if line.strip():
                yield json.loads(line)
                emitted += 1
        if emitted != limit:
            raise ValueError(f"requested {limit} rows but read {emitted} from {path}")


def output_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def restore_row(row: dict[str, Any]) -> dict[str, Any]:
    restored = copy.deepcopy(row)
    restored.pop("provence", None)
    for _, _, chunk in iter_chunks(restored):
        if "provence_original_text" in chunk:
            chunk["text"] = chunk.pop("provence_original_text")
            chunk.pop("provence_compression", None)
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--four-config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict[str, Any] = {"status": "PASS", "datasets": {}}
    total_violations = 0
    for tag, expected_n in DATASETS:
        calibration = json.loads(
            (args.root / "calibration_dense" / tag / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        test = json.loads(
            (args.root / "test" / tag / "metrics.json").read_text(encoding="utf-8")
        )
        threshold = float(calibration["protocol"]["threshold"])
        violations: list[dict[str, Any]] = []
        if test["protocol"]["threshold_grid"]:
            violations.append({"field": "test_threshold_grid", "value": test["protocol"]["threshold_grid"]})
        if float(test["protocol"]["threshold"]) != threshold:
            violations.append(
                {
                    "field": "locked_threshold",
                    "calibration": threshold,
                    "test": test["protocol"]["threshold"],
                }
            )

        source_path = (
            args.four_config_root / tag / f"calibrated_full_{tag}_replay.jsonl"
        )
        replay_path = args.root / "test" / tag / "provence_replay.jsonl"
        count = 0
        for position, (source, replay) in enumerate(
            zip(source_slice(source_path, 1000, expected_n), output_rows(replay_path))
        ):
            count += 1
            source_chunks = extract_final_chunks(source)
            replay_chunks = extract_final_chunks(replay)
            source_keys = [stable_key(chunk) for chunk in source_chunks]
            replay_keys = [stable_key(chunk) for chunk in replay_chunks]
            provenance = replay.get("provence", {})
            if source_keys != replay_keys:
                violations.append({"row": position, "field": "selected_chunk_keys"})
            if provenance.get("source_row_index") != 1000 + position:
                violations.append({"row": position, "field": "source_row_index"})
            if float(provenance.get("threshold", -1.0)) != threshold:
                violations.append({"row": position, "field": "row_threshold"})
            if provenance.get("selected_chunk_keys") != source_keys:
                violations.append({"row": position, "field": "provenance_keys"})
            marked = sum(
                "provence_original_text" in chunk for _, _, chunk in iter_chunks(replay)
            )
            if marked != len(source_chunks):
                violations.append(
                    {
                        "row": position,
                        "field": "marked_chunk_count",
                        "expected": len(source_chunks),
                        "actual": marked,
                    }
                )
            if restore_row(replay) != source:
                violations.append({"row": position, "field": "frozen_row_restore"})
            if len(violations) >= 25:
                break

        if count != expected_n:
            violations.append({"field": "row_count", "expected": expected_n, "actual": count})
        result = test["selected_result"]
        if result["compressed_empty_contexts"] > result["original_empty_contexts"]:
            violations.append(
                {
                    "field": "empty_contexts",
                    "original": result["original_empty_contexts"],
                    "compressed": result["compressed_empty_contexts"],
                }
            )
        total_violations += len(violations)
        report["datasets"][tag] = {
            "rows_audited": count,
            "locked_threshold": threshold,
            "violations": violations,
            "status": "PASS" if not violations else "FAIL",
        }

    if total_violations:
        report["status"] = "FAIL"
    report["total_violations"] = total_violations
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if total_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
