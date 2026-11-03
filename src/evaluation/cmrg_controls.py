from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    from conf import CONTINUE_TAG, FINISH_TAG, TERMINATE_TAG
except Exception:  # pragma: no cover - keeps the utility usable in isolated smoke tests.
    CONTINUE_TAG = "<CONTINUE>"
    FINISH_TAG = "<FINISH>"
    TERMINATE_TAG = "<TERMINATE>"


def load_jsonl_slice(path: str, offset: int = 0, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(chunk: dict) -> str:
    return str(chunk.get("id") or chunk.get("text") or "")


def iter_chunks(sample: dict):
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for rank, chunk in enumerate(sample[iter_key].get("docs", []), start=1):
            yield iter_idx, rank, chunk


def label_score(label: str) -> float:
    if label == CONTINUE_TAG:
        return 1.0
    if label == FINISH_TAG:
        return 0.8
    if label == TERMINATE_TAG:
        return -2.0
    return 0.0


def normalize_term(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


def extract_terms(text: str) -> set[str]:
    if not text:
        return set()
    words = re.findall(r"[A-Za-z][A-Za-z0-9.-]*", text.lower())
    return {normalize_term(word) for word in words if len(word) > 2}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def chunk_text(chunk: dict) -> str:
    return " ".join([str(chunk.get("title", "")), str(chunk.get("text", ""))])


def retention_features(
    *,
    question_terms: set[str],
    query_terms: set[str],
    kept_terms: set[str],
    kept_ids: set[str],
    chunk: dict,
    label: str,
    rank: int,
    order: int,
) -> tuple[dict[str, float], set[str]]:
    candidate_terms = extract_terms(chunk_text(chunk))
    novelty_terms = candidate_terms - kept_terms
    novelty = len(novelty_terms) / max(len(candidate_terms), 1)
    bridge_overlap = jaccard(candidate_terms, kept_terms)
    duplicate_id = 1.0 if stable_key(chunk) in kept_ids else 0.0
    redundancy = max(duplicate_id, bridge_overlap * (1.0 - min(novelty, 1.0)))
    values = {
        "label_score": label_score(label),
        "question_overlap": jaccard(candidate_terms, question_terms | query_terms),
        "bridge_overlap": bridge_overlap,
        "novelty": novelty,
        "redundancy": redundancy,
        "rank_prior": 1.0 / max(float(rank), 1.0),
        "order_prior": 1.0 / max(float(order), 1.0),
    }
    return values, candidate_terms


def extract_final_chunks(sample: dict) -> list[dict]:
    has_explicit_keep = any(
        "keep_for_qa" in chunk
        for _, _, chunk in iter_chunks(sample)
    )
    chunks: list[dict] = []
    seen: set[str] = set()
    for _, _, chunk in iter_chunks(sample):
        should_use = bool(chunk.get("keep_for_qa")) if has_explicit_keep else chunk.get("label") != TERMINATE_TAG
        if not should_use:
            continue
        key = stable_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        chunks.append(chunk)
    return chunks


def mark_keep_fields(
    sample: dict,
    keep_keys: set[str],
    score_fn: Callable[[dict], float],
    *,
    feature_map: dict[str, dict[str, float]] | None = None,
    debug_features: bool = False,
) -> dict:
    seen: set[str] = set()
    for _, _, chunk in iter_chunks(sample):
        key = stable_key(chunk)
        keep = bool(key and key in keep_keys and key not in seen)
        if keep:
            seen.add(key)
        chunk["keep_for_qa"] = keep
        chunk["use_for_next_query"] = bool(
            chunk.get("label") == CONTINUE_TAG and str(chunk.get("info", "")).strip()
        )
        chunk["retention_score"] = round(float(score_fn(chunk)), 6)
        if debug_features and feature_map and key in feature_map:
            chunk["retention_features"] = dict(feature_map[key])
    return sample


def label_only_replay(sample: dict, _: int, *, debug_features: bool = False) -> dict:
    keep_keys = {
        stable_key(chunk)
        for _, _, chunk in iter_chunks(sample)
        if stable_key(chunk) and chunk.get("label") != TERMINATE_TAG
    }
    return mark_keep_fields(
        sample,
        keep_keys,
        lambda chunk: label_score(str(chunk.get("label", ""))),
        debug_features=debug_features,
    )


def rank_only_replay(sample: dict, budget: int, *, debug_features: bool = False) -> dict:
    keep_keys: list[str] = []
    seen: set[str] = set()
    for _, _, chunk in iter_chunks(sample):
        key = stable_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        keep_keys.append(key)
        if len(keep_keys) >= budget:
            break
    return mark_keep_fields(
        sample,
        set(keep_keys),
        lambda chunk: 1.0,
        debug_features=debug_features,
    )


def budget_only_replay(sample: dict, budget: int, *, debug_features: bool = False) -> dict:
    keep_keys: list[str] = []
    seen: set[str] = set()
    for _, _, chunk in iter_chunks(sample):
        if chunk.get("label") == TERMINATE_TAG:
            continue
        key = stable_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        keep_keys.append(key)
        if len(keep_keys) >= budget:
            break
    return mark_keep_fields(
        sample,
        set(keep_keys),
        lambda chunk: 1.0,
        debug_features=debug_features,
    )


def budget_redundancy_hybrid_replay(
    sample: dict,
    budget: int,
    *,
    pool_extra: int,
    redundancy_threshold: float,
    redundancy_weight: float,
    debug_features: bool = False,
) -> dict:
    question = sample.get("query") or sample.get("question") or ""
    question_terms = extract_terms(question)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    pool_size = max(budget, budget + pool_extra)

    for order, (iter_idx, rank, chunk) in enumerate(iter_chunks(sample), start=1):
        if chunk.get("label") == TERMINATE_TAG:
            continue
        key = stable_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "chunk": chunk,
                "key": key,
                "iter_idx": iter_idx,
                "rank": rank,
                "order": order,
                "label": str(chunk.get("label", "")),
            }
        )
        if len(candidates) >= pool_size:
            break

    kept_terms: set[str] = set()
    kept_ids: set[str] = set()
    keep_keys: list[str] = []
    scores: dict[str, float] = {}
    feature_map: dict[str, dict[str, float]] = {}
    remaining = list(candidates)

    while remaining and len(keep_keys) < budget:
        scored: list[dict[str, Any]] = []
        for item in remaining:
            iter_key = str(item["iter_idx"])
            query = sample.get(iter_key, {}).get("query", question)
            values, terms = retention_features(
                question_terms=question_terms,
                query_terms=extract_terms(query),
                kept_terms=kept_terms,
                kept_ids=kept_ids,
                chunk=item["chunk"],
                label=item["label"],
                rank=item["rank"],
                order=item["order"],
            )
            score = (
                2.0 * values["order_prior"]
                + 0.30 * values["label_score"]
                + 0.20 * values["question_overlap"]
                + 0.15 * values["novelty"]
                - redundancy_weight * values["redundancy"]
            )
            scored.append(
                {
                    "item": item,
                    "features": values,
                    "terms": terms,
                    "score": score,
                }
            )

        baseline_next = scored[0]
        if baseline_next["features"]["redundancy"] <= redundancy_threshold:
            chosen = baseline_next
        else:
            non_redundant = [
                row
                for row in scored
                if row["features"]["redundancy"] <= redundancy_threshold
            ]
            chosen = (
                max(non_redundant, key=lambda row: (row["score"], -row["item"]["order"]))
                if non_redundant
                else baseline_next
            )

        item = chosen["item"]
        keep_keys.append(item["key"])
        scores[item["key"]] = chosen["score"]
        feature_map[item["key"]] = chosen["features"]
        kept_terms |= chosen["terms"]
        kept_ids.add(item["key"])
        remaining = [row for row in remaining if row["key"] != item["key"]]

    return mark_keep_fields(
        sample,
        set(keep_keys),
        lambda chunk: scores.get(stable_key(chunk), 0.0),
        feature_map=feature_map,
        debug_features=debug_features,
    )


def resolve_budgets(args: argparse.Namespace, rows: list[dict]) -> tuple[list[int], dict[str, Any]]:
    if args.budget_source:
        source_rows = load_jsonl_slice(args.budget_source, args.budget_source_offset, args.budget_source_limit)
        if len(source_rows) != len(rows):
            raise SystemExit(
                f"Budget source row count mismatch: source={len(source_rows)}, baseline={len(rows)}"
            )
        budgets = [len(extract_final_chunks(row)) for row in source_rows]
        return budgets, {
            "budget_mode": "budget_source",
            "budget_source": args.budget_source,
            "budget_source_offset": args.budget_source_offset,
            "budget_source_limit": args.budget_source_limit,
        }

    if args.target_final_k is None:
        raise SystemExit("Either --target_final_k or --budget_source is required.")
    if args.target_final_k < 0:
        raise SystemExit("--target_final_k must be non-negative.")
    target_budget = int(math.floor(args.target_final_k))
    return [target_budget for _ in rows], {
        "budget_mode": "fixed_target_final_k",
        "target_final_k": target_budget,
    }


def build_controls(args: argparse.Namespace) -> None:
    baseline_rows = load_jsonl_slice(args.baseline, args.baseline_offset, args.baseline_limit)
    budgets, budget_manifest = resolve_budgets(args, baseline_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    run_ids = {
        "label_only": args.run_id_label_only,
        "rank_only": args.run_id_rank_only,
        "budget_only": args.run_id_budget_only,
        "budget_redundancy_hybrid": args.run_id_budget_redundancy_hybrid,
    }
    replay_fns: dict[str, Callable[[dict, int], dict]] = {
        "label_only": lambda sample, budget: label_only_replay(sample, budget, debug_features=args.debug_features),
        "rank_only": lambda sample, budget: rank_only_replay(sample, budget, debug_features=args.debug_features),
        "budget_only": lambda sample, budget: budget_only_replay(sample, budget, debug_features=args.debug_features),
        "budget_redundancy_hybrid": lambda sample, budget: budget_redundancy_hybrid_replay(
            sample,
            budget,
            pool_extra=args.hybrid_pool_extra,
            redundancy_threshold=args.hybrid_redundancy_threshold,
            redundancy_weight=args.hybrid_redundancy_weight,
            debug_features=args.debug_features,
        ),
    }

    unknown = selected_variants - set(replay_fns)
    if unknown:
        raise SystemExit(f"Unknown variants: {sorted(unknown)}")

    manifest: dict[str, Any] = {
        "baseline": args.baseline,
        "baseline_offset": args.baseline_offset,
        "baseline_limit": args.baseline_limit,
        "samples": len(baseline_rows),
        "file_tag": args.file_tag,
        "score_field": "retention_score",
        "variants": {},
        **budget_manifest,
    }

    for variant in sorted(selected_variants):
        name = run_ids[variant]
        rows = [
            replay_fns[variant](json.loads(json.dumps(sample)), budget)
            for sample, budget in zip(baseline_rows, budgets)
        ]
        path = output_dir / f"{name}_{args.file_tag}_replay.jsonl"
        write_jsonl(rows, path)
        manifest["variants"][name] = {"variant": variant, "path": str(path)}
        if variant == "budget_redundancy_hybrid":
            manifest["variants"][name].update(
                {
                    "pool_extra": args.hybrid_pool_extra,
                    "redundancy_threshold": args.hybrid_redundancy_threshold,
                    "redundancy_weight": args.hybrid_redundancy_weight,
                }
            )

    manifest_name = args.manifest_name or f"retention_{args.file_tag}_control_manifest.json"
    manifest_path = output_dir / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build budget-controlled and redundancy-aware retention replays."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output_dir", default="refine-logs/retention_controls")
    parser.add_argument("--baseline_offset", type=int, default=0)
    parser.add_argument("--baseline_limit", type=int, default=None)
    parser.add_argument("--target_final_k", type=int, default=None)
    parser.add_argument("--budget_source", default=None)
    parser.add_argument("--budget_source_offset", type=int, default=0)
    parser.add_argument("--budget_source_limit", type=int, default=None)
    parser.add_argument("--hybrid_pool_extra", type=int, default=3)
    parser.add_argument("--hybrid_redundancy_threshold", type=float, default=0.45)
    parser.add_argument("--hybrid_redundancy_weight", type=float, default=1.2)
    parser.add_argument("--file_tag", default="controls")
    parser.add_argument("--manifest_name", default=None)
    parser.add_argument("--debug_features", action="store_true")
    parser.add_argument(
        "--variants",
        default="budget_only,budget_redundancy_hybrid",
        help="Comma-separated: label_only,rank_only,budget_only,budget_redundancy_hybrid",
    )
    parser.add_argument("--run_id_label_only", default="label_only")
    parser.add_argument("--run_id_rank_only", default="rank_only")
    parser.add_argument("--run_id_budget_only", default="budget_only")
    parser.add_argument("--run_id_budget_redundancy_hybrid", default="budget_redundancy_hybrid")
    return parser.parse_args()


if __name__ == "__main__":
    build_controls(parse_args())
