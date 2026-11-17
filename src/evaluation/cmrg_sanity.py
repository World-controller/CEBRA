import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from conf import CONTINUE_TAG, TERMINATE_TAG
from efficient_rag.cmrg import CMRGConfig, public_cmrg_fields, score_candidates
from efficientrag_qa import EfficientRAG_QA


FORBIDDEN_FEATURE_KEYS = {
    "oracle",
    "oracle_id",
    "oracle_ids",
    "gold",
    "gold_answer",
    "answer",
    "positive_paragraph_idx",
    "metric",
    "em",
    "f1",
    "correctness",
}


def build_sample():
    return {
        "query": "Which city is the birthplace of the author of The Hobbit?",
        "answer": "Bloemfontein",
        "oracle": ["sample-00", "sample-01"],
        "0": {
            "query": "Which city is the birthplace of the author of The Hobbit?",
            "filter_input": "",
            "docs": [
                {
                    "id": "sample-00",
                    "title": "The Hobbit",
                    "text": "The Hobbit is a fantasy novel by J. R. R. Tolkien.",
                    "label": CONTINUE_TAG,
                    "info": "J. R. R. Tolkien",
                },
                {
                    "id": "sample-02",
                    "title": "Fantasy",
                    "text": "Fantasy is a genre of speculative fiction.",
                    "label": CONTINUE_TAG,
                    "info": "fantasy fiction",
                },
                {
                    "id": "sample-03",
                    "title": "Noise",
                    "text": "This unrelated passage discusses marine biology.",
                    "label": TERMINATE_TAG,
                    "info": "",
                },
            ],
        },
        "1": {
            "query": "J. R. R. Tolkien birthplace",
            "filter_input": "J. R. R. Tolkien",
            "docs": [
                {
                    "id": "sample-01",
                    "title": "J. R. R. Tolkien",
                    "text": "J. R. R. Tolkien was born in Bloemfontein.",
                    "label": CONTINUE_TAG,
                    "info": "Bloemfontein",
                },
                {
                    "id": "sample-00",
                    "title": "The Hobbit",
                    "text": "The Hobbit is a fantasy novel by J. R. R. Tolkien.",
                    "label": CONTINUE_TAG,
                    "info": "J. R. R. Tolkien",
                },
            ],
        },
    }


def run_cmrg_on_sample(sample):
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
    config = CMRGConfig(keep_threshold=0.35, debug_features=True)
    kept_context = []
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        docs = sample[iter_key]["docs"]
        labels = [doc["label"] for doc in docs]
        infos = [doc["info"] for doc in docs]
        cmrg_outputs = score_candidates(
            question=sample["query"],
            query=sample[iter_key]["query"],
            chunks=docs,
            labels=labels,
            infos=infos,
            kept_context=kept_context,
            nlp=nlp,
            config=config,
        )
        for doc, cmrg_output in zip(docs, cmrg_outputs):
            doc.update(public_cmrg_fields(cmrg_output, debug_features=True))
            if cmrg_output["keep_for_qa"]:
                kept_context.append(
                    {
                        "id": doc["id"],
                        "title": doc["title"],
                        "text": doc["text"],
                        "_cmrg_terms": cmrg_output["_cmrg_terms"],
                    }
                )
    return sample


def extract_chunk_ids(sample):
    qa = EfficientRAG_QA(model=None, data=[], dataset="hotpotQA", num_workers=1)
    return [chunk["id"] for chunk in qa.extract_chunks(sample)]


def audit_forbidden_features(sample):
    violations = []
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for doc in sample[iter_key]["docs"]:
            features = doc.get("cmrg_features", {})
            for key in features:
                normalized = key.lower()
                if normalized in FORBIDDEN_FEATURE_KEYS:
                    violations.append({"doc_id": doc.get("id"), "feature": key})
    return violations


def main():
    old_sample = build_sample()
    new_sample = run_cmrg_on_sample(build_sample())
    mixed_sample = build_sample()
    mixed_sample["0"]["docs"][0]["keep_for_qa"] = True
    all_terminate_sample = {
        "query": "No answer",
        "answer": "",
        "oracle": [],
        "0": {
            "query": "No answer",
            "filter_input": "",
            "docs": [
                {
                    "id": "noise-00",
                    "title": "Noise",
                    "text": "",
                    "label": TERMINATE_TAG,
                    "info": "",
                    "keep_for_qa": False,
                }
            ],
        },
    }
    old_chunk_ids = extract_chunk_ids(old_sample)
    new_chunk_ids = extract_chunk_ids(new_sample)
    mixed_chunk_ids = extract_chunk_ids(mixed_sample)
    all_terminate_chunk_ids = extract_chunk_ids(all_terminate_sample)
    violations = audit_forbidden_features(new_sample)
    result = {
        "r001_schema_fields_present": all(
            {"keep_for_qa", "use_for_next_query", "chain_marginal_score"}.issubset(
                set(doc)
            )
            for iter_key in ("0", "1")
            for doc in new_sample[iter_key]["docs"]
        ),
        "r002_forbidden_feature_violations": violations,
        "r003_old_fallback_chunk_ids": old_chunk_ids,
        "r003_cmrg_chunk_ids": new_chunk_ids,
        "r003_old_fallback_ok": old_chunk_ids
        == ["sample-00", "sample-02", "sample-01"],
        "r003_cmrg_path_ok": "sample-01" in new_chunk_ids
        and "sample-03" not in new_chunk_ids,
        "mixed_schema_chunk_ids": mixed_chunk_ids,
        "mixed_schema_ok": mixed_chunk_ids
        == ["sample-00", "sample-02", "sample-01"],
        "all_terminate_chunk_ids": all_terminate_chunk_ids,
        "all_terminate_ok": all_terminate_chunk_ids == [],
    }
    output_path = Path("refine-logs") / "cmrg_sanity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["r001_schema_fields_present"]:
        raise SystemExit("CMRG schema fields missing")
    if violations:
        raise SystemExit("Forbidden CMRG feature detected")
    if not result["r003_old_fallback_ok"]:
        raise SystemExit("Old QA fallback behavior changed")
    if not result["r003_cmrg_path_ok"]:
        raise SystemExit("CMRG QA extraction path failed")
    if not result["mixed_schema_ok"]:
        raise SystemExit("Mixed schema fallback failed")
    if not result["all_terminate_ok"]:
        raise SystemExit("All-TERMINATE fallback failed")


if __name__ == "__main__":
    main()
