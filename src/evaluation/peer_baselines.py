#!/usr/bin/env python3
"""Peer-baseline replay on frozen EfficientRAG trajectories.

The script is deliberately split-disciplined: a calibration run cannot read
rows at or beyond ``--calibration-rows``.  Every policy sees the same union of
eligible, de-duplicated, non-TERMINATE chunks from the frozen trajectory.  The
Budget-only replay defines a per-question chunk cap and encoder-token budget;
the semantic policies must stay within both.

Implemented policies:

* Budget-only: original trajectory order, at most B chunks.
* Semantic MMR: query relevance minus maximum selected-set redundancy.
* AdaGReS: the paper's adaptive-beta set objective and positive-gain greedy.
* Submodular packing: relevance, query coverage, representativeness, and
  source diversity with gain-per-token greedy and singleton fallback.
* Gold-support Oracle: support-recall upper bound under the same budgets.

AdaGReS and the submodular packer are equation-level reproductions because the
released AdaGReS service is tied to a private Milvus schema and does not expose
a frozen-candidate runner.  All semantic policies use the same encoder.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import os
import platform
import re
import string
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from cmrg_calibrated import fallback_terms, recall, sample_oracle_ids  # noqa: E402
from cmrg_controls import (  # noqa: E402
    CONTINUE_TAG,
    TERMINATE_TAG,
    budget_only_replay,
    extract_final_chunks,
    iter_chunks,
    stable_key,
)

METHODS = (
    "budget_only",
    "semantic_mmr",
    "adagres",
    "submodular",
    "gold_support_oracle",
)
VALID_METHODS = METHODS + ("reference_policy",)
EPS = 1e-12


@dataclass
class Candidate:
    key: str
    chunk: dict[str, Any]
    order: int
    text: str
    token_count: int
    terms: set[str]
    source: str
    embedding: np.ndarray | None = None
    query_similarity: float = 0.0
    lexical_relevance: float = 0.0
    covered_oracles: set[str] = field(default_factory=set)


@dataclass
class PreparedSample:
    row_index: int
    sample: dict[str, Any]
    question: str
    question_terms: set[str]
    candidates: list[Candidate]
    query_embedding: np.ndarray
    token_budget: int
    chunk_cap: int
    budget_keys: list[str]


def chunk_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        part for part in (str(chunk.get("title", "")), str(chunk.get("text", ""))) if part
    )


def normalize_answer_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_in_context(sample: dict[str, Any], chunks: Sequence[dict[str, Any]]) -> float:
    answer = normalize_answer_text(str(sample.get("answer", "")))
    if not answer:
        return 0.0
    context = normalize_answer_text(" ".join(chunk_text(chunk) for chunk in chunks))
    return float(answer in context)


def nonnegative_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return max(0.0, float(np.dot(left, right)))


def collect_candidates(sample: dict[str, Any], tokenizer: Any) -> list[Candidate]:
    """Collect eligible chunks in stable trajectory order."""
    seen: set[str] = set()
    candidates: list[Candidate] = []
    oracle_ids = sample_oracle_ids(sample)
    for order, (_, _, chunk) in enumerate(iter_chunks(sample)):
        if str(chunk.get("label", "")) == TERMINATE_TAG:
            continue
        key = stable_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        text = chunk_text(chunk)
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        token_count = max(1, len(encoded["input_ids"]))
        key_parts = set(key.split("//"))
        candidates.append(
            Candidate(
                key=key,
                chunk=chunk,
                order=order,
                text=text,
                token_count=token_count,
                terms=fallback_terms(text),
                source=str(chunk.get("title") or key),
                covered_oracles=key_parts & oracle_ids,
            )
        )
    return candidates


def mark_peer_keep_fields(
    sample: dict[str, Any],
    keep_keys: set[str],
    score_by_key: dict[str, float],
) -> dict[str, Any]:
    """Mark only the first eligible occurrence of each selected key.

    A chunk id can first occur with a TERMINATE label and reappear later as an
    eligible chunk.  Selection is over eligible candidates, so marking must not
    attach the later decision to the earlier TERMINATE occurrence.
    """
    seen_kept: set[str] = set()
    for _, _, chunk in iter_chunks(sample):
        key = stable_key(chunk)
        eligible = str(chunk.get("label", "")) != TERMINATE_TAG
        keep = bool(eligible and key and key in keep_keys and key not in seen_kept)
        if keep:
            seen_kept.add(key)
        chunk["keep_for_qa"] = keep
        chunk["use_for_next_query"] = bool(
            chunk.get("label") == CONTINUE_TAG and str(chunk.get("info", "")).strip()
        )
        chunk["retention_score"] = round(float(score_by_key.get(key, 0.0)), 6)
    return sample


def load_rows(
    input_path: Path,
    *,
    offset: int,
    limit: int,
    split: str,
    calibration_rows: int,
) -> list[tuple[int, dict[str, Any]]]:
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be non-negative and limit must be positive")
    if split == "calibration" and offset + limit > calibration_rows:
        raise ValueError(
            f"calibration run [{offset}, {offset + limit}) crosses locked boundary "
            f"{calibration_rows}"
        )
    if split == "test" and offset < calibration_rows:
        raise ValueError(
            f"test run must start at or after calibration boundary {calibration_rows}"
        )

    rows: list[tuple[int, dict[str, Any]]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < offset:
                continue
            if row_index >= offset + limit:
                break
            if line.strip():
                rows.append((row_index, json.loads(line)))
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows but read {len(rows)} from {input_path}")
    return rows


def _encode_texts(model: Any, texts: Sequence[str], batch_size: int) -> dict[str, np.ndarray]:
    unique_texts = list(dict.fromkeys(texts))
    vectors = model.encode(
        unique_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return {text: np.asarray(vector, dtype=np.float32) for text, vector in zip(unique_texts, vectors)}


def prepare_samples(
    indexed_rows: Sequence[tuple[int, dict[str, Any]]],
    model: Any,
    *,
    budget: int,
    batch_size: int,
    indexed_reference_rows: Sequence[tuple[int, dict[str, Any]]] | None = None,
) -> list[PreparedSample]:
    tokenizer = model.tokenizer
    staged: list[tuple[int, dict[str, Any], str, list[Candidate], list[str]]] = []
    all_texts: list[str] = []

    if indexed_reference_rows is not None and len(indexed_reference_rows) != len(indexed_rows):
        raise ValueError("budget reference and input must have the same number of rows")

    for position, (row_index, sample) in enumerate(indexed_rows):
        question = str(sample.get("query") or sample.get("question") or "")
        candidates = collect_candidates(sample, tokenizer)
        if indexed_reference_rows is None:
            budget_row = budget_only_replay(copy.deepcopy(sample), budget)
        else:
            reference_index, budget_row = indexed_reference_rows[position]
            reference_question = str(
                budget_row.get("query") or budget_row.get("question") or ""
            )
            if reference_index != row_index or reference_question != question:
                raise ValueError(
                    f"budget reference is not aligned at source row {row_index}"
                )
        budget_chunks = extract_final_chunks(budget_row)
        budget_keys = [stable_key(chunk) for chunk in budget_chunks]
        staged.append((row_index, sample, question, candidates, budget_keys))
        all_texts.append(question)
        all_texts.extend(candidate.text for candidate in candidates)

    vector_by_text = _encode_texts(model, all_texts, batch_size)
    prepared: list[PreparedSample] = []
    for row_index, sample, question, candidates, budget_keys in staged:
        query_embedding = vector_by_text[question]
        by_key = {candidate.key: candidate for candidate in candidates}
        question_terms = fallback_terms(question)
        for candidate in candidates:
            candidate.embedding = vector_by_text[candidate.text]
            candidate.query_similarity = nonnegative_cosine(query_embedding, candidate.embedding)
            union = candidate.terms | question_terms
            candidate.lexical_relevance = (
                len(candidate.terms & question_terms) / len(union) if union else 0.0
            )
        token_budget = sum(by_key[key].token_count for key in budget_keys if key in by_key)
        prepared.append(
            PreparedSample(
                row_index=row_index,
                sample=sample,
                question=question,
                question_terms=question_terms,
                candidates=candidates,
                query_embedding=query_embedding,
                token_budget=token_budget,
                chunk_cap=len(budget_keys),
                budget_keys=budget_keys,
            )
        )
    return prepared


def _similarity_matrix(candidates: Sequence[Candidate]) -> np.ndarray:
    if not candidates:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings = np.stack([candidate.embedding for candidate in candidates])
    return np.maximum(embeddings @ embeddings.T, 0.0)


def select_semantic_mmr(
    prepared: PreparedSample, *, lam: float
) -> tuple[list[int], dict[str, float], dict[str, Any]]:
    candidates = prepared.candidates
    sims = _similarity_matrix(candidates)
    selected: list[int] = []
    used_tokens = 0
    selection_scores: dict[str, float] = {}
    while len(selected) < prepared.chunk_cap:
        feasible = [
            idx
            for idx, candidate in enumerate(candidates)
            if idx not in selected and used_tokens + candidate.token_count <= prepared.token_budget
        ]
        if not feasible:
            break
        scored = []
        for idx in feasible:
            redundancy = max((float(sims[idx, prev]) for prev in selected), default=0.0)
            score = lam * candidates[idx].query_similarity - (1.0 - lam) * redundancy
            scored.append((score, -candidates[idx].order, idx))
        score, _, best = max(scored)
        selected.append(best)
        used_tokens += candidates[best].token_count
        selection_scores[candidates[best].key] = float(score)
    return selected, selection_scores, {"lambda": lam, "used_tokens": used_tokens}


def select_adagres(
    prepared: PreparedSample,
) -> tuple[list[int], dict[str, float], dict[str, Any]]:
    """Adaptive-beta greedy selection following AdaGReS Eq. (1)--(7)."""
    candidates = prepared.candidates
    if not candidates or prepared.chunk_cap == 0 or prepared.token_budget == 0:
        return [], {}, {"beta": 0.0, "estimated_k": 0, "used_tokens": 0}
    sims = _similarity_matrix(candidates)
    mean_length = float(np.mean([candidate.token_count for candidate in candidates]))
    estimated_k = max(
        1,
        min(prepared.chunk_cap, int(prepared.token_budget // max(mean_length, 1.0))),
    )
    mean_relevance = float(np.mean([candidate.query_similarity for candidate in candidates]))
    if len(candidates) > 1:
        pair_values = sims[np.triu_indices(len(candidates), k=1)]
        mean_pair_similarity = float(pair_values.mean()) if len(pair_values) else 0.0
    else:
        mean_pair_similarity = 0.0
    if estimated_k <= 1 or mean_pair_similarity <= EPS:
        beta = 0.01
    else:
        beta = mean_relevance / (((estimated_k - 1) / 2.0) * mean_pair_similarity + EPS)

    selected: list[int] = []
    used_tokens = 0
    selection_scores: dict[str, float] = {}
    while len(selected) < prepared.chunk_cap:
        feasible = [
            idx
            for idx, candidate in enumerate(candidates)
            if idx not in selected and used_tokens + candidate.token_count <= prepared.token_budget
        ]
        if not feasible:
            break
        scored = []
        for idx in feasible:
            redundancy = sum(float(sims[idx, prev]) for prev in selected)
            gain = candidates[idx].query_similarity - beta * redundancy
            scored.append((gain, -candidates[idx].order, idx))
        gain, _, best = max(scored)
        if gain <= 0.0:
            break
        selected.append(best)
        used_tokens += candidates[best].token_count
        selection_scores[candidates[best].key] = float(gain)
    return selected, selection_scores, {
        "beta": float(beta),
        "estimated_k": estimated_k,
        "mean_query_similarity": mean_relevance,
        "mean_pair_similarity": mean_pair_similarity,
        "used_tokens": used_tokens,
    }


class SubmodularObjective:
    def __init__(
        self,
        prepared: PreparedSample,
        *,
        w_rel: float = 1.0,
        w_query: float = 0.5,
        w_repr: float = 0.4,
        w_div: float = 0.3,
        saturation: float = 0.3,
    ) -> None:
        self.prepared = prepared
        self.candidates = prepared.candidates
        self.sims = _similarity_matrix(self.candidates)
        self.weights = (w_rel, w_query, w_repr, w_div)
        self.saturation = saturation
        self.rel = np.asarray(
            [candidate.lexical_relevance for candidate in self.candidates], dtype=np.float64
        )
        self.rel_denom = max(float(self.rel.sum()), EPS)
        self.query_denom = max(len(prepared.question_terms), 1)
        self.degrees = self.sims.sum(axis=1).astype(np.float64)
        self.repr_caps = saturation * self.degrees
        self.repr_denom = max(float(self.repr_caps.sum()), EPS)
        source_mass: dict[str, float] = {}
        for candidate, relevance in zip(self.candidates, self.rel):
            source_mass[candidate.source] = source_mass.get(candidate.source, 0.0) + float(relevance)
        self.div_denom = max(sum(math.sqrt(value) for value in source_mass.values()), EPS)

    def __call__(self, selected: Iterable[int]) -> float:
        indices = tuple(sorted(selected))
        if not indices:
            return 0.0
        rel_value = float(self.rel[list(indices)].sum()) / self.rel_denom
        covered_terms: set[str] = set()
        for idx in indices:
            covered_terms |= self.candidates[idx].terms & self.prepared.question_terms
        query_value = len(covered_terms) / self.query_denom

        representation_mass = self.sims[:, list(indices)].sum(axis=1)
        repr_value = float(np.minimum(representation_mass, self.repr_caps).sum()) / self.repr_denom

        selected_source_mass: dict[str, float] = {}
        for idx in indices:
            source = self.candidates[idx].source
            selected_source_mass[source] = selected_source_mass.get(source, 0.0) + float(self.rel[idx])
        div_value = sum(math.sqrt(value) for value in selected_source_mass.values()) / self.div_denom
        w_rel, w_query, w_repr, w_div = self.weights
        return (
            w_rel * rel_value
            + w_query * query_value
            + w_repr * repr_value
            + w_div * div_value
        )


def select_submodular(
    prepared: PreparedSample,
) -> tuple[list[int], dict[str, float], dict[str, Any]]:
    objective = SubmodularObjective(prepared)
    candidates = prepared.candidates
    selected: list[int] = []
    used_tokens = 0
    current_value = 0.0
    selection_scores: dict[str, float] = {}
    while len(selected) < prepared.chunk_cap:
        feasible = [
            idx
            for idx, candidate in enumerate(candidates)
            if idx not in selected and used_tokens + candidate.token_count <= prepared.token_budget
        ]
        if not feasible:
            break
        scored = []
        for idx in feasible:
            value = objective((*selected, idx))
            gain = value - current_value
            density = gain / candidates[idx].token_count
            scored.append((density, gain, -candidates[idx].order, idx, value))
        _, gain, _, best, best_value = max(scored)
        if gain <= 0.0:
            break
        selected.append(best)
        used_tokens += candidates[best].token_count
        current_value = best_value
        selection_scores[candidates[best].key] = float(gain)

    singleton_options = [
        (objective((idx,)), -candidate.token_count, -candidate.order, idx)
        for idx, candidate in enumerate(candidates)
        if candidate.token_count <= prepared.token_budget and prepared.chunk_cap >= 1
    ]
    singleton_fallback = False
    if singleton_options:
        singleton_value, _, _, singleton_idx = max(singleton_options)
        if singleton_value > current_value + EPS:
            selected = [singleton_idx]
            used_tokens = candidates[singleton_idx].token_count
            current_value = singleton_value
            selection_scores = {candidates[singleton_idx].key: float(singleton_value)}
            singleton_fallback = True
    return selected, selection_scores, {
        "objective": current_value,
        "used_tokens": used_tokens,
        "singleton_fallback": singleton_fallback,
        "weights": {"rel": 1.0, "query": 0.5, "repr": 0.4, "div": 0.3},
        "saturation": 0.3,
    }


def select_gold_support_oracle(
    prepared: PreparedSample,
) -> tuple[list[int], dict[str, float], dict[str, Any]]:
    candidates = prepared.candidates
    oracle_ids = sample_oracle_ids(prepared.sample)
    relevant = [idx for idx, candidate in enumerate(candidates) if candidate.covered_oracles]
    best_indices: tuple[int, ...] = ()
    best_key = (-1.0, 0, 0)
    max_size = min(prepared.chunk_cap, len(relevant))
    for size in range(1, max_size + 1):
        for indices in itertools.combinations(relevant, size):
            token_count = sum(candidates[idx].token_count for idx in indices)
            if token_count > prepared.token_budget:
                continue
            covered = set().union(*(candidates[idx].covered_oracles for idx in indices))
            support_recall = len(covered) / len(oracle_ids) if oracle_ids else 0.0
            candidate_key = (support_recall, -token_count, -sum(candidates[idx].order for idx in indices))
            if candidate_key > best_key:
                best_key = candidate_key
                best_indices = indices
    score_map = {
        candidates[idx].key: len(candidates[idx].covered_oracles) / max(len(oracle_ids), 1)
        for idx in best_indices
    }
    return list(best_indices), score_map, {
        "used_tokens": sum(candidates[idx].token_count for idx in best_indices),
        "oracle_support_recall": max(best_key[0], 0.0),
    }


def apply_policy(
    prepared: PreparedSample,
    method: str,
    *,
    mmr_lambda: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method in {"budget_only", "reference_policy"}:
        selected_indices = [
            idx for idx, candidate in enumerate(prepared.candidates) if candidate.key in prepared.budget_keys
        ]
        score_map = {
            prepared.candidates[idx].key: 1.0 - prepared.candidates[idx].order * 1e-6
            for idx in selected_indices
        }
        details: dict[str, Any] = {
            "used_tokens": sum(prepared.candidates[idx].token_count for idx in selected_indices)
        }
    elif method == "semantic_mmr":
        selected_indices, score_map, details = select_semantic_mmr(prepared, lam=mmr_lambda)
    elif method == "adagres":
        selected_indices, score_map, details = select_adagres(prepared)
    elif method == "submodular":
        selected_indices, score_map, details = select_submodular(prepared)
    elif method == "gold_support_oracle":
        selected_indices, score_map, details = select_gold_support_oracle(prepared)
    else:
        raise ValueError(f"unknown method: {method}")

    selection_order = [prepared.candidates[idx].key for idx in selected_indices]
    selected_keys = set(selection_order)
    trajectory_order = [
        candidate.key for candidate in prepared.candidates if candidate.key in selected_keys
    ]
    base_scores = {candidate.key: candidate.query_similarity for candidate in prepared.candidates}
    base_scores.update(score_map)
    replay = mark_peer_keep_fields(copy.deepcopy(prepared.sample), selected_keys, base_scores)
    replay["peer_baseline"] = {
        "method": method,
        "source_row_index": prepared.row_index,
        "token_budget": prepared.token_budget,
        "chunk_cap": prepared.chunk_cap,
        "selected_keys": trajectory_order,
        "selection_order": selection_order,
        **details,
    }
    return replay, details


def summarize_method(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    recalls: list[float] = []
    ks: list[int] = []
    tokens: list[int] = []
    words: list[int] = []
    aic: list[float] = []
    yesno: list[bool] = []
    empty = 0
    for row in rows:
        chunks = extract_final_chunks(row)
        recalls.append(recall(sample_oracle_ids(row), chunks))
        ks.append(len(chunks))
        tokens.append(int(row["peer_baseline"]["used_tokens"]))
        words.append(sum(len(re.findall(r"\w+", chunk_text(chunk))) for chunk in chunks))
        aic.append(answer_in_context(row, chunks))
        normalized_answer = normalize_answer_text(str(row.get("answer", "")))
        yesno.append(normalized_answer in {"yes", "no"})
        empty += int(not chunks)
    span_aic = [value for value, is_yesno in zip(aic, yesno) if not is_yesno]
    return {
        "samples": len(recalls),
        "support_recall": float(np.mean(recalls)),
        "mean_chunks": float(np.mean(ks)),
        "mean_budget_tokens": float(np.mean(tokens)),
        "mean_words": float(np.mean(words)),
        "answer_in_context": float(np.mean(span_aic)) if span_aic else 0.0,
        "answer_in_context_all": float(np.mean(aic)),
        "answer_in_context_span": float(np.mean(span_aic)) if span_aic else 0.0,
        "yes_no_questions": int(sum(yesno)),
        "empty_contexts": empty,
        "per_row": {
            "support_recall": recalls,
            "chunks": ks,
            "budget_tokens": tokens,
            "words": words,
            "answer_in_context": aic,
            "is_yes_no": yesno,
        },
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "test"), required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--calibration-rows", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=3)
    parser.add_argument(
        "--budget-reference-replay",
        type=Path,
        default=None,
        help="optional aligned replay whose retained chunks define per-question token and chunk budgets",
    )
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument(
        "--mmr-lambda-grid",
        default="",
        help="calibration-only comma-separated sweep; winner is locked by recall, empty count, and budget distance",
    )
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.mmr_lambda <= 1.0:
        raise ValueError("--mmr-lambda must be in [0, 1]")
    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    unknown = sorted(set(methods) - set(VALID_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    mmr_grid = tuple(
        float(part.strip()) for part in args.mmr_lambda_grid.split(",") if part.strip()
    )
    if any(not 0.0 <= value <= 1.0 for value in mmr_grid):
        raise ValueError("all --mmr-lambda-grid values must be in [0, 1]")
    if mmr_grid and args.split != "calibration":
        raise ValueError("--mmr-lambda-grid is calibration-only; pass the locked --mmr-lambda on test")

    indexed_rows = load_rows(
        args.input,
        offset=args.offset,
        limit=args.limit,
        split=args.split,
        calibration_rows=args.calibration_rows,
    )
    indexed_reference_rows = None
    if args.budget_reference_replay is not None:
        indexed_reference_rows = load_rows(
            args.budget_reference_replay,
            offset=args.offset,
            limit=args.limit,
            split=args.split,
            calibration_rows=args.calibration_rows,
        )

    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    torch.manual_seed(42)
    np.random.seed(42)

    started = time.perf_counter()
    model = SentenceTransformer(
        args.encoder,
        device=device,
        local_files_only=not args.allow_model_download,
    )
    prepared = prepare_samples(
        indexed_rows,
        model,
        budget=args.budget,
        batch_size=args.batch_size,
        indexed_reference_rows=indexed_reference_rows,
    )
    preparation_seconds = time.perf_counter() - started

    mmr_sweep: dict[str, Any] = {}
    if mmr_grid and "semantic_mmr" in methods:
        target_mean_chunks = float(np.mean([item.chunk_cap for item in prepared]))
        for lam in mmr_grid:
            sweep_rows = [
                apply_policy(item, "semantic_mmr", mmr_lambda=lam)[0] for item in prepared
            ]
            sweep_summary = summarize_method(sweep_rows)
            mmr_sweep[f"{lam:.6g}"] = {
                key: value for key, value in sweep_summary.items() if key != "per_row"
            }
        selected_key = max(
            mmr_sweep,
            key=lambda key: (
                mmr_sweep[key]["support_recall"],
                -mmr_sweep[key]["empty_contexts"],
                -abs(mmr_sweep[key]["mean_chunks"] - target_mean_chunks),
                -float(key),
            ),
        )
        args.mmr_lambda = float(selected_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    method_seconds: dict[str, float] = {}
    for method in methods:
        method_started = time.perf_counter()
        rows = [apply_policy(item, method, mmr_lambda=args.mmr_lambda)[0] for item in prepared]
        method_seconds[method] = time.perf_counter() - method_started
        write_jsonl(rows, args.output_dir / f"{method}_replay.jsonl")
        summaries[method] = summarize_method(rows)

    manifest = {
        "protocol": {
            "input": str(args.input),
            "split": args.split,
            "offset": args.offset,
            "limit": args.limit,
            "calibration_rows": args.calibration_rows,
            "budget_only_cap": args.budget,
            "budget_reference_replay": (
                str(args.budget_reference_replay)
                if args.budget_reference_replay is not None
                else None
            ),
            "budget_matching": (
                "per-question encoder-token budget and chunk cap measured from the reference replay"
                if args.budget_reference_replay is not None
                else "per-question encoder-token budget and chunk cap measured from Budget-only"
            ),
            "candidate_pool": (
                "union of frozen trajectory chunks; non-TERMINATE; stable id/text de-duplication"
            ),
            "query_updates": "frozen; use_for_next_query remains label/info-derived and independent",
            "output_order": "selected chunks retain their original frozen-trajectory order",
            "encoder": args.encoder,
            "budget_tokenization": "full untruncated encoder WordPiece sequence",
            "embedding_truncation_tokens": int(model.max_seq_length),
            "device": device,
            "mmr_lambda": args.mmr_lambda,
            "mmr_lambda_grid": list(mmr_grid),
            "mmr_tie_break": (
                "higher support recall, fewer empty contexts, closer mean chunks to Budget-only, lower lambda"
            ),
            "methods": list(methods),
            "seed": 42,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": git_revision(),
            "torch": torch.__version__,
        },
        "timing_seconds": {
            "prepare_and_encode": preparation_seconds,
            "methods": method_seconds,
            "total": time.perf_counter() - started,
        },
        "mmr_calibration_sweep": mmr_sweep,
        "results": summaries,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    compact_results = {
        method: {key: value for key, value in summary.items() if key != "per_row"}
        for method, summary in summaries.items()
    }
    print(
        json.dumps(
            {"timing_seconds": manifest["timing_seconds"], "results": compact_results},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
