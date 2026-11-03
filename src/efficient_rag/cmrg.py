import re
from dataclasses import dataclass
from typing import Any

from conf import CONTINUE_TAG, FINISH_TAG, TERMINATE_TAG


@dataclass
class CMRGConfig:
    keep_threshold: float = 0.35
    min_keep_per_iter: int = 1
    control_next_query: bool = False
    debug_features: bool = False


def _normalize_term(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text


def _fallback_terms(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9.-]*", text.lower())
    return {word for word in words if len(word) > 2}


def extract_terms(text: str, nlp: Any = None) -> set[str]:
    """Extract lightweight lexical anchors without introducing a graph/KG dependency."""
    if not text:
        return set()
    if nlp is None:
        return _fallback_terms(text)
    doc = nlp(text[:4000])
    terms: set[str] = set()
    for ent in doc.ents:
        normalized = _normalize_term(ent.text)
        if len(normalized) > 1:
            terms.add(normalized)
    for chunk in doc.noun_chunks:
        normalized = _normalize_term(chunk.text)
        if len(normalized) > 2:
            terms.add(normalized)
    for token in doc:
        if token.is_alpha and not token.is_stop and len(token.text) > 2:
            terms.add(_normalize_term(token.lemma_ or token.text))
    return {term for term in terms if term}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _label_score(label: str) -> float:
    if label == CONTINUE_TAG:
        return 1.0
    if label == FINISH_TAG:
        return 0.8
    if label == TERMINATE_TAG:
        return -2.0
    return 0.0


def score_candidates(
    *,
    question: str,
    query: str,
    chunks: list[dict[str, Any]],
    labels: list[str],
    infos: list[str],
    kept_context: list[dict[str, Any]],
    nlp: Any,
    config: CMRGConfig,
) -> list[dict[str, Any]]:
    question_terms = extract_terms(question, nlp)
    query_terms = extract_terms(query, nlp)
    kept_terms: set[str] = set()
    kept_ids: set[str] = set()
    for kept in kept_context:
        kept_terms |= set(kept.get("_cmrg_terms", set()))
        if kept.get("id"):
            kept_ids.add(str(kept["id"]))

    scored: list[dict[str, Any]] = []
    for rank, (chunk, label, info) in enumerate(zip(chunks, labels, infos), start=1):
        chunk_terms = extract_terms(
            " ".join([str(chunk.get("title", "")), str(chunk.get("text", ""))]),
            nlp,
        )
        question_overlap = _jaccard(chunk_terms, question_terms | query_terms)
        bridge_overlap = _jaccard(chunk_terms, kept_terms)
        new_terms = chunk_terms - kept_terms
        novelty = len(new_terms) / max(len(chunk_terms), 1)
        duplicate_id = 1.0 if str(chunk.get("id", "")) in kept_ids else 0.0
        redundancy = max(duplicate_id, bridge_overlap * (1.0 - min(novelty, 1.0)))
        rank_prior = 1.0 / rank

        score = (
            0.65 * _label_score(label)
            + 0.55 * question_overlap
            + 0.45 * bridge_overlap
            + 0.35 * novelty
            + 0.20 * rank_prior
            - 0.85 * redundancy
        )

        keep_for_qa = label != TERMINATE_TAG and score >= config.keep_threshold
        use_for_next_query = (
            label == CONTINUE_TAG
            and bool(str(info).strip())
            and (keep_for_qa if config.control_next_query else True)
        )

        features = {
            "label_score": _label_score(label),
            "question_overlap": question_overlap,
            "bridge_overlap": bridge_overlap,
            "novelty": novelty,
            "redundancy": redundancy,
            "rank_prior": rank_prior,
        }
        scored_item = {
            "keep_for_qa": keep_for_qa,
            "use_for_next_query": use_for_next_query,
            "chain_marginal_score": round(score, 6),
            "_cmrg_terms": chunk_terms,
            "_cmrg_features": features,
        }
        scored.append(scored_item)

        if keep_for_qa:
            kept_terms |= chunk_terms
            if chunk.get("id"):
                kept_ids.add(str(chunk["id"]))

    _ensure_minimum_keep(scored, labels, config)
    return scored


def _ensure_minimum_keep(
    scored: list[dict[str, Any]], labels: list[str], config: CMRGConfig
) -> None:
    if config.min_keep_per_iter <= 0:
        return
    kept = sum(1 for item in scored if item["keep_for_qa"])
    if kept >= config.min_keep_per_iter:
        return

    candidates = [
        (idx, item["chain_marginal_score"])
        for idx, item in enumerate(scored)
        if labels[idx] != TERMINATE_TAG
    ]
    for idx, _ in sorted(candidates, key=lambda pair: pair[1], reverse=True)[
        : config.min_keep_per_iter - kept
    ]:
        scored[idx]["keep_for_qa"] = True


def public_cmrg_fields(item: dict[str, Any], *, debug_features: bool) -> dict[str, Any]:
    fields = {
        "keep_for_qa": bool(item["keep_for_qa"]),
        "use_for_next_query": bool(item["use_for_next_query"]),
        "chain_marginal_score": float(item["chain_marginal_score"]),
    }
    if debug_features:
        fields["cmrg_features"] = dict(item.get("_cmrg_features", {}))
    return fields
