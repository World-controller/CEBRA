import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efficient_rag.cmrg import CMRGConfig
from evaluation.cmrg_replay import load_nlp, replay_sample
from evaluation.cmrg_retrieve_metrics import (
    extract_final_chunks,
    oracle_ids,
    recall,
)
from evaluation.cmrg_qa_reuse_pilot import prompt_signature


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def retrieval_metrics(rows: list[dict]) -> dict:
    per_sample = []
    raw_total = 0
    duplicate_total = 0
    empty_count = 0
    for sample in rows:
        chunks, raw_kept_count, duplicate_count = extract_final_chunks(sample)
        raw_total += raw_kept_count
        duplicate_total += duplicate_count
        if not chunks:
            empty_count += 1
        per_sample.append(
            {
                "recall": recall(oracle_ids(sample), chunks),
                "final_context_k": len(chunks),
                "raw_kept_count": raw_kept_count,
            }
        )
    return {
        "samples": len(rows),
        "recall": float(np.mean([row["recall"] for row in per_sample]))
        if per_sample
        else 0.0,
        "final_context_k": float(
            np.mean([row["final_context_k"] for row in per_sample])
        )
        if per_sample
        else 0.0,
        "raw_kept_count": float(
            np.mean([row["raw_kept_count"] for row in per_sample])
        )
        if per_sample
        else 0.0,
        "duplicate_ratio": float(duplicate_total) / max(float(raw_total), 1.0),
        "empty_final_context_count": empty_count,
    }


def context_changed_count(baseline_rows: list[dict], candidate_rows: list[dict]) -> int:
    changed = 0
    for baseline, candidate in zip(baseline_rows, candidate_rows):
        if prompt_signature(baseline) != prompt_signature(candidate):
            changed += 1
    return changed


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument(
        "--thresholds",
        default="0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25",
    )
    parser.add_argument("--min_keep_values", default="0,1")
    parser.add_argument("--max_recall_drop", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_rows = load_jsonl(args.fpath, args.limit)
    baseline_metrics = retrieval_metrics(baseline_rows)
    nlp = load_nlp()

    candidates = []
    for min_keep in parse_int_list(args.min_keep_values):
        for threshold in parse_float_list(args.thresholds):
            config = CMRGConfig(
                keep_threshold=threshold,
                min_keep_per_iter=min_keep,
                control_next_query=False,
                debug_features=True,
            )
            candidate_rows = [
                replay_sample(copy.deepcopy(sample), nlp, config)
                for sample in baseline_rows
            ]
            metrics = retrieval_metrics(candidate_rows)
            changed_count = context_changed_count(baseline_rows, candidate_rows)
            item = {
                "keep_threshold": threshold,
                "min_keep_per_iter": min_keep,
                "context_changed_count": changed_count,
                "context_changed_ratio": changed_count / max(len(baseline_rows), 1),
                "metrics": metrics,
                "delta": {
                    "recall": metrics["recall"] - baseline_metrics["recall"],
                    "final_context_k": metrics["final_context_k"]
                    - baseline_metrics["final_context_k"],
                    "raw_kept_count": metrics["raw_kept_count"]
                    - baseline_metrics["raw_kept_count"],
                    "duplicate_ratio": metrics["duplicate_ratio"]
                    - baseline_metrics["duplicate_ratio"],
                },
                "passes_recall_gate": metrics["recall"]
                >= baseline_metrics["recall"] - args.max_recall_drop,
                "passes_empty_gate": metrics["empty_final_context_count"]
                <= baseline_metrics["empty_final_context_count"],
            }
            candidates.append(item)

    viable = [
        item
        for item in candidates
        if item["passes_recall_gate"]
        and item["passes_empty_gate"]
        and item["context_changed_count"] > 0
    ]
    selected = None
    if viable:
        selected = sorted(
            viable,
            key=lambda item: (
                item["metrics"]["final_context_k"],
                item["metrics"]["duplicate_ratio"],
                -item["context_changed_count"],
                -item["metrics"]["recall"],
            ),
        )[0]

    result = {
        "baseline_metrics": baseline_metrics,
        "max_recall_drop": args.max_recall_drop,
        "selected": selected,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
