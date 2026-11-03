import argparse
import json
import os
import pickle
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conf import CONTINUE_TAG, FINISH_TAG, TERMINATE_TAG

FEATURE_NAMES = [
    "label_score",
    "question_overlap",
    "bridge_overlap",
    "novelty",
    "redundancy",
    "rank_prior",
    "iter_prior",
    "has_next_info",
    "title_question_overlap",
]
TERM_CACHE_MAXSIZE = 200_000
TERM_CACHE: OrderedDict[tuple[int, str], set[str]] = OrderedDict()


def clear_term_cache() -> None:
    TERM_CACHE.clear()


def get_cached_terms(cache_key: tuple[int, str]) -> Optional[set[str]]:
    if cache_key not in TERM_CACHE:
        return None
    terms = TERM_CACHE.pop(cache_key)
    TERM_CACHE[cache_key] = terms
    return set(terms)


def cache_terms(cache_key: tuple[int, str], terms: set[str]) -> None:
    if cache_key in TERM_CACHE:
        TERM_CACHE.pop(cache_key)
    TERM_CACHE[cache_key] = set(terms)
    while len(TERM_CACHE) > TERM_CACHE_MAXSIZE:
        TERM_CACHE.popitem(last=False)


def feature_names_from_drop(drop_features: str) -> list[str]:
    dropped = {item.strip() for item in drop_features.split(",") if item.strip()}
    unknown = sorted(dropped - set(FEATURE_NAMES))
    if unknown:
        raise SystemExit(f"Unknown drop_features: {unknown}; valid={FEATURE_NAMES}")
    names = [name for name in FEATURE_NAMES if name not in dropped]
    if not names:
        raise SystemExit("--drop_features cannot remove every feature.")
    return names


def validate_feature_names(feature_names: Any) -> list[str]:
    if not isinstance(feature_names, list) or not all(
        isinstance(name, str) for name in feature_names
    ):
        raise SystemExit(f"Invalid feature_names in artifact: {feature_names}")
    expected_order = [name for name in FEATURE_NAMES if name in feature_names]
    if (
        not feature_names
        or len(set(feature_names)) != len(feature_names)
        or feature_names != expected_order
    ):
        raise SystemExit(
            f"Feature mismatch: artifact feature_names must be a non-empty "
            f"order-preserving subset of {FEATURE_NAMES}; got {feature_names}"
        )
    return list(feature_names)


def load_jsonl_slice(path: str, offset: int = 0, limit: Optional[int] = None) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def iter_jsonl_slice(
    path: str, offset: int = 0, limit: Optional[int] = None
) -> Iterable[dict]:
    emitted = 0
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < offset:
                continue
            yield json.loads(line)
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def load_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def normalize_term(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


def fallback_terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9.-]*", text.lower())
    return {word for word in words if len(word) > 2}


def extract_terms(text: str, nlp: Any = None) -> set[str]:
    if not text:
        return set()
    cache_key = (id(nlp), text[:4000])
    cached_terms = get_cached_terms(cache_key)
    if cached_terms is not None:
        return cached_terms
    if nlp is None:
        terms = fallback_terms(text)
        cache_terms(cache_key, terms)
        return terms
    doc = nlp(text[:4000])
    terms: set[str] = set()
    for ent in doc.ents:
        normalized = normalize_term(ent.text)
        if len(normalized) > 1:
            terms.add(normalized)
    for chunk in doc.noun_chunks:
        normalized = normalize_term(chunk.text)
        if len(normalized) > 2:
            terms.add(normalized)
    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.text) > 2:
            terms.add(normalize_term(token.lemma_ or token.text))
    terms = {term for term in terms if term}
    cache_terms(cache_key, terms)
    return terms


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def label_score(label: str) -> float:
    if label == CONTINUE_TAG:
        return 1.0
    if label == FINISH_TAG:
        return 0.8
    if label == TERMINATE_TAG:
        return -2.0
    return 0.0


def sample_oracle_ids(sample: dict) -> set[str]:
    return {str(item) for item in sample.get("oracle", [])}


def chunk_text(chunk: dict) -> str:
    return " ".join([str(chunk.get("title", "")), str(chunk.get("text", ""))])


def feature_dict(
    *,
    question_terms: set[str],
    query_terms: set[str],
    kept_terms: set[str],
    kept_ids: set[str],
    chunk: dict,
    label: str,
    info: str,
    rank: int,
    iter_idx: int,
    nlp: Any,
) -> tuple[dict[str, float], set[str]]:
    title_terms = extract_terms(str(chunk.get("title", "")), nlp)
    candidate_terms = extract_terms(chunk_text(chunk), nlp)
    novelty_terms = candidate_terms - kept_terms
    novelty = len(novelty_terms) / max(len(candidate_terms), 1)
    bridge_overlap = jaccard(candidate_terms, kept_terms)
    duplicate_id = 1.0 if str(chunk.get("id", "")) in kept_ids else 0.0
    redundancy = max(duplicate_id, bridge_overlap * (1.0 - min(novelty, 1.0)))
    values = {
        "label_score": label_score(label),
        "question_overlap": jaccard(candidate_terms, question_terms | query_terms),
        "bridge_overlap": bridge_overlap,
        "novelty": novelty,
        "redundancy": redundancy,
        "rank_prior": 1.0 / max(rank, 1),
        "iter_prior": 1.0 / float(iter_idx + 1),
        "has_next_info": 1.0 if str(info).strip() else 0.0,
        "title_question_overlap": jaccard(title_terms, question_terms | query_terms),
    }
    return values, candidate_terms


def vectorize(features: dict[str, float], names: Optional[list[str]] = None) -> list[float]:
    if names is None:
        names = FEATURE_NAMES
    return [float(features[name]) for name in names]


def training_examples(
    sample: dict, nlp: Any, feature_names: Optional[list[str]] = None
) -> tuple[list[list[float]], list[int]]:
    if feature_names is None:
        feature_names = FEATURE_NAMES
    question = sample.get("query") or sample.get("question") or ""
    question_terms = extract_terms(question, nlp)
    oracle_ids = sample_oracle_ids(sample)
    covered_oracles: set[str] = set()
    kept_terms: set[str] = set()
    kept_ids: set[str] = set()
    features: list[list[float]] = []
    labels: list[int] = []

    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        docs = sample[iter_key].get("docs", [])
        query_terms = extract_terms(sample[iter_key].get("query", question), nlp)
        for rank, chunk in enumerate(docs, start=1):
            doc_label = str(chunk.get("label", ""))
            info = str(chunk.get("info", ""))
            values, candidate_terms = feature_dict(
                question_terms=question_terms,
                query_terms=query_terms,
                kept_terms=kept_terms,
                kept_ids=kept_ids,
                chunk=chunk,
                label=doc_label,
                info=info,
                rank=rank,
                iter_idx=iter_idx,
                nlp=nlp,
            )
            doc_id = str(chunk.get("id", ""))
            doc_oracle_parts = set(doc_id.split("//")).intersection(oracle_ids)
            target = int(bool(doc_oracle_parts - covered_oracles))
            features.append(vectorize(values, feature_names))
            labels.append(target)

            # Teacher-forced chain state: oracle-positive chunks define covered hops
            # during training, while inference uses the model's own keep decisions.
            if target:
                covered_oracles |= doc_oracle_parts
                kept_terms |= candidate_terms
                if doc_id:
                    kept_ids.add(doc_id)
    return features, labels


def train_model(args: argparse.Namespace) -> None:
    nlp = load_nlp()
    feature_names = feature_names_from_drop(getattr(args, "drop_features", ""))
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for sample in iter_jsonl_slice(args.fpath, args.train_offset, args.train_limit):
        features, labels = training_examples(sample, nlp, feature_names)
        x_rows.extend(features)
        y_rows.extend(labels)

    if not x_rows or len(set(y_rows)) < 2:
        raise SystemExit("Training data must contain both positive and negative labels.")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    random_state=args.seed,
                ),
            ),
        ]
    )
    pipeline.fit(np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows, dtype=np.int32))
    model = pipeline.named_steps["model"]
    dropped_names = [name for name in FEATURE_NAMES if name not in set(feature_names)]
    summary = {
        "feature_names": feature_names,
        "train_offset": args.train_offset,
        "train_limit": args.train_limit,
        "examples": len(y_rows),
        "positives": int(sum(y_rows)),
        "positive_rate": float(sum(y_rows) / len(y_rows)),
        "intercept": [float(v) for v in model.intercept_],
        "coef": {name: float(value) for name, value in zip(feature_names, model.coef_[0])},
    }
    if dropped_names:
        summary["drop_features"] = dropped_names
    artifact = {"pipeline": pipeline, "feature_names": feature_names, "summary": summary}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(artifact, f)
    if args.summary_output:
        Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_output).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_model(path: str) -> tuple[Pipeline, list[str]]:
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    feature_names = validate_feature_names(artifact.get("feature_names"))
    return artifact["pipeline"], feature_names


def replay_sample_calibrated(
    sample: dict,
    *,
    nlp: Any,
    model: Pipeline,
    threshold: float,
    min_keep_per_iter: int,
    debug_features: bool,
    feature_names: Optional[list[str]] = None,
) -> dict:
    if feature_names is None:
        feature_names = FEATURE_NAMES
    question = sample.get("query") or sample.get("question") or ""
    question_terms = extract_terms(question, nlp)
    kept_terms: set[str] = set()
    kept_ids: set[str] = set()

    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        docs = sample[iter_key].get("docs", [])
        query_terms = extract_terms(sample[iter_key].get("query", question), nlp)
        iter_outputs = []
        for rank, chunk in enumerate(docs, start=1):
            doc_label = str(chunk.get("label", ""))
            info = str(chunk.get("info", ""))
            values, candidate_terms = feature_dict(
                question_terms=question_terms,
                query_terms=query_terms,
                kept_terms=kept_terms,
                kept_ids=kept_ids,
                chunk=chunk,
                label=doc_label,
                info=info,
                rank=rank,
                iter_idx=iter_idx,
                nlp=nlp,
            )
            probability = float(model.predict_proba([vectorize(values, feature_names)])[0][1])
            keep = doc_label != TERMINATE_TAG and probability >= threshold
            iter_outputs.append(
                {
                    "chunk": chunk,
                    "features": values,
                    "terms": candidate_terms,
                    "probability": probability,
                    "keep": keep,
                    "label": doc_label,
                    "info": info,
                }
            )

        kept_count = sum(1 for item in iter_outputs if item["keep"])
        if min_keep_per_iter > 0 and kept_count < min_keep_per_iter:
            candidates = [
                item
                for item in iter_outputs
                if item["label"] != TERMINATE_TAG and not item["keep"]
            ]
            for item in sorted(candidates, key=lambda row: row["probability"], reverse=True)[
                : min_keep_per_iter - kept_count
            ]:
                item["keep"] = True

        for item in iter_outputs:
            chunk = item["chunk"]
            chunk["keep_for_qa"] = bool(item["keep"])
            chunk["use_for_next_query"] = bool(
                item["label"] == CONTINUE_TAG and str(item["info"]).strip()
            )
            chunk["chain_marginal_score"] = round(float(item["probability"]), 6)
            if debug_features:
                chunk["cmrg_features"] = dict(item["features"])
            if item["keep"]:
                kept_terms |= item["terms"]
                doc_id = str(chunk.get("id", ""))
                if doc_id:
                    kept_ids.add(doc_id)
    return sample


def replay_model(args: argparse.Namespace) -> None:
    nlp = load_nlp()
    model, feature_names = load_model(args.model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in iter_jsonl_slice(args.fpath, args.offset, args.limit):
            row = replay_sample_calibrated(
                sample,
                nlp=nlp,
                model=model,
                feature_names=feature_names,
                threshold=args.threshold,
                min_keep_per_iter=args.min_keep_per_iter,
                debug_features=args.debug_features,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_final_chunks(sample: dict) -> tuple[list[dict], int, int]:
    has_cmrg_keep = any(
        "keep_for_qa" in chunk
        for iter_idx in range(4)
        if str(iter_idx) in sample
        for chunk in sample[str(iter_idx)].get("docs", [])
    )
    chunks = []
    seen = set()
    raw_kept_count = 0
    duplicate_count = 0
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for chunk in sample[iter_key].get("docs", []):
            if has_cmrg_keep and "keep_for_qa" in chunk:
                should_use = bool(chunk["keep_for_qa"])
            else:
                should_use = chunk.get("label") != TERMINATE_TAG
            if not should_use:
                continue
            raw_kept_count += 1
            key = chunk.get("id") or chunk.get("text")
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            chunks.append(chunk)
    return chunks, raw_kept_count, duplicate_count


def recall(oracle_ids: set[str], chunks: list[dict]) -> float:
    if not oracle_ids:
        return 0.0
    covered: set[str] = set()
    for chunk in chunks:
        # EfficientRAG sometimes merges chunk ids with "//"; match the existing
        # retrieval metric so calibrated sweeps are comparable to R004-R009.
        key = str(chunk.get("id") or chunk.get("text") or "")
        covered |= set(key.split("//")).intersection(oracle_ids)
    return len(covered) / len(oracle_ids)


def prompt_signature(sample: dict) -> dict:
    chunks, _, _ = extract_final_chunks(sample)
    return {
        "query": str(sample.get("query", "")),
        "context": [
            {
                "id": str(chunk.get("id", "")),
                "title": str(chunk.get("title", "")),
                "text": str(chunk.get("text", "")),
            }
            for chunk in chunks
        ],
    }


def retrieval_metrics(rows: list[dict]) -> dict:
    raw_total = 0
    duplicate_total = 0
    empty_count = 0
    per_sample = []
    for sample in rows:
        chunks, raw_kept_count, duplicate_count = extract_final_chunks(sample)
        raw_total += raw_kept_count
        duplicate_total += duplicate_count
        if not chunks:
            empty_count += 1
        per_sample.append(
            {
                "recall": recall(sample_oracle_ids(sample), chunks),
                "final_context_k": len(chunks),
                "raw_kept_count": raw_kept_count,
            }
        )
    return {
        "samples": len(rows),
        "recall": float(np.mean([row["recall"] for row in per_sample])) if rows else 0.0,
        "final_context_k": float(np.mean([row["final_context_k"] for row in per_sample]))
        if rows
        else 0.0,
        "raw_kept_count": float(np.mean([row["raw_kept_count"] for row in per_sample]))
        if rows
        else 0.0,
        "duplicate_ratio": float(duplicate_total) / max(float(raw_total), 1.0),
        "empty_final_context_count": empty_count,
    }


def sweep_model(args: argparse.Namespace) -> None:
    nlp = load_nlp()
    model, feature_names = load_model(args.model)
    baseline_rows = load_jsonl_slice(args.fpath, args.offset, args.limit)
    baseline_metrics = retrieval_metrics(baseline_rows)
    candidates = []
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    min_keep_values = [int(item) for item in args.min_keep_values.split(",") if item.strip()]
    for min_keep in min_keep_values:
        for threshold in thresholds:
            candidate_rows = [
                replay_sample_calibrated(
                    json.loads(json.dumps(sample)),
                    nlp=nlp,
                    model=model,
                    feature_names=feature_names,
                    threshold=threshold,
                    min_keep_per_iter=min_keep,
                    debug_features=False,
                )
                for sample in baseline_rows
            ]
            metrics = retrieval_metrics(candidate_rows)
            changed_count = sum(
                1
                for baseline, candidate in zip(baseline_rows, candidate_rows)
                if prompt_signature(baseline) != prompt_signature(candidate)
            )
            item = {
                "threshold": threshold,
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
        "train_independent_eval_offset": args.offset,
        "max_recall_drop": args.max_recall_drop,
        "selected": selected,
        "candidates": candidates,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("--fpath", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--summary_output", default=None)
    train.add_argument("--train_offset", type=int, default=500)
    train.add_argument("--train_limit", type=int, default=2000)
    train.add_argument("--max_iter", type=int, default=1000)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--drop_features", default="")
    train.set_defaults(func=train_model)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--fpath", required=True)
    replay.add_argument("--model", required=True)
    replay.add_argument("--output", required=True)
    replay.add_argument("--threshold", type=float, default=0.5)
    replay.add_argument("--min_keep_per_iter", type=int, default=1)
    replay.add_argument("--offset", type=int, default=0)
    replay.add_argument("--limit", type=int, default=None)
    replay.add_argument("--debug_features", action="store_true")
    replay.set_defaults(func=replay_model)

    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--fpath", required=True)
    sweep.add_argument("--model", required=True)
    sweep.add_argument("--output", required=True)
    sweep.add_argument("--offset", type=int, default=0)
    sweep.add_argument("--limit", type=int, default=500)
    sweep.add_argument(
        "--thresholds",
        default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90",
    )
    sweep.add_argument("--min_keep_values", default="0,1")
    sweep.add_argument("--max_recall_drop", type=float, default=0.01)
    sweep.set_defaults(func=sweep_model)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
