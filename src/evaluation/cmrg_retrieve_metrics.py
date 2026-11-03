from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall(oracle: list[str], chunks: list[str]) -> float:
    oracle_set = {str(item) for item in oracle}
    if not oracle_set:
        return 0.0
    covered_oracles = set()
    for chunk in chunks:
        covered_oracles |= set(str(chunk).split("//")).intersection(oracle_set)
    return len(covered_oracles) / len(oracle_set)


def extract_final_chunks(sample: dict) -> tuple[list[str], int, int]:
    has_retention_keep = any(
        "keep_for_qa" in chunk
        for iter_idx in range(4)
        if str(iter_idx) in sample
        for chunk in sample[str(iter_idx)].get("docs", [])
    )
    chunk_ids: list[str] = []
    seen = set()
    raw_kept_count = 0
    duplicate_count = 0
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for chunk in sample[iter_key].get("docs", []):
            if has_retention_keep and "keep_for_qa" in chunk:
                should_use = bool(chunk["keep_for_qa"])
            else:
                should_use = chunk.get("label") != "<TERMINATE>"
            if not should_use:
                continue
            raw_kept_count += 1
            key = chunk.get("id") or chunk.get("text")
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            chunk_ids.append(str(key))
    return chunk_ids, raw_kept_count, duplicate_count


def schema_stats(sample: dict) -> tuple[int, int, int]:
    required_core_fields = {"keep_for_qa", "use_for_next_query"}
    score_fields = {"retention_score", "chain_marginal_score"}
    total_docs = 0
    docs_with_any_retention = 0
    docs_with_required_retention = 0
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for chunk in sample[iter_key].get("docs", []):
            total_docs += 1
            if required_core_fields.intersection(chunk) or score_fields.intersection(chunk):
                docs_with_any_retention += 1
            if required_core_fields.issubset(chunk) and score_fields.intersection(chunk):
                docs_with_required_retention += 1
    return total_docs, docs_with_any_retention, docs_with_required_retention


def oracle_ids(sample: dict) -> list[str]:
    if "oracle_ids" in sample:
        return [str(item) for item in sample["oracle_ids"]]
    if "oracle" in sample:
        return [str(item) for item in sample["oracle"]]
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_jsonl(args.fpath)
    if args.offset or args.limit is not None:
        data = data[
            args.offset : None if args.limit is None else args.offset + args.limit
        ]
    rows = []
    total_docs = 0
    docs_with_any_retention = 0
    docs_with_required_retention = 0
    empty_final_context_count = 0
    for sample in data:
        chunks, raw_kept_count, duplicate_count = extract_final_chunks(sample)
        sample_total_docs, sample_any_retention, sample_required_retention = schema_stats(sample)
        total_docs += sample_total_docs
        docs_with_any_retention += sample_any_retention
        docs_with_required_retention += sample_required_retention
        if not chunks:
            empty_final_context_count += 1
        rows.append(
            {
                "recall": recall(oracle_ids(sample), chunks),
                "final_context_k": len(chunks),
                "raw_kept_count": raw_kept_count,
                "duplicate_count": duplicate_count,
            }
        )
    metrics = {
        "samples": len(rows),
        "recall": float(np.mean([row["recall"] for row in rows])) if rows else 0.0,
        "final_context_k": float(np.mean([row["final_context_k"] for row in rows]))
        if rows
        else 0.0,
        "raw_kept_count": float(np.mean([row["raw_kept_count"] for row in rows]))
        if rows
        else 0.0,
        "duplicate_ratio": (
            float(sum(row["duplicate_count"] for row in rows))
            / max(float(sum(row["raw_kept_count"] for row in rows)), 1.0)
        ),
        "recall_per_chunk": (
            float(np.mean([row["recall"] for row in rows]))
            / max(float(np.mean([row["final_context_k"] for row in rows])), 1e-12)
            if rows
            else 0.0
        ),
        "empty_final_context_count": empty_final_context_count,
        "total_docs": total_docs,
        "retention_docs_with_any_required_field": docs_with_any_retention,
        "retention_docs_with_all_required_fields": docs_with_required_retention,
        "retention_required_field_coverage": (
            float(docs_with_required_retention) / docs_with_any_retention
            if docs_with_any_retention
            else None
        ),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
