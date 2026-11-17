import argparse
import inspect
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efficient_rag import cmrg


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

ALLOWED_CMRG_FEATURE_KEYS = {
    "label_score",
    "question_overlap",
    "bridge_overlap",
    "novelty",
    "redundancy",
    "rank_prior",
}


def load_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def iter_docs(sample: dict):
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for doc in sample[iter_key].get("docs", []):
            yield iter_key, doc


def audit_sample(sample: dict, sample_index: int) -> tuple[int, list[dict]]:
    feature_doc_count = 0
    violations = []
    for iter_key, doc in iter_docs(sample):
        features = doc.get("cmrg_features")
        if not isinstance(features, dict):
            continue
        feature_doc_count += 1
        # R006 只允许 inference features；oracle/answer/metric 类字段必须为 0。
        for key in features:
            if key.lower() in FORBIDDEN_FEATURE_KEYS:
                violations.append(
                    {
                        "sample_index": sample_index,
                        "iter": iter_key,
                        "doc_id": doc.get("id"),
                        "feature": key,
                    }
                )
            if key not in ALLOWED_CMRG_FEATURE_KEYS:
                violations.append(
                    {
                        "sample_index": sample_index,
                        "iter": iter_key,
                        "doc_id": doc.get("id"),
                        "feature": key,
                        "reason": "unexpected_feature_key",
                    }
                )
    return feature_doc_count, violations


def audit_cmrg_source() -> list[dict]:
    source = "\n".join(
        [
            inspect.getsource(cmrg.score_candidates),
            inspect.getsource(cmrg.public_cmrg_fields),
            inspect.getsource(cmrg.extract_terms),
        ]
    ).lower()
    violations = []
    for forbidden in sorted(FORBIDDEN_FEATURE_KEYS):
        if re.search(rf"\b{re.escape(forbidden)}\b", source):
            violations.append(
                {
                    "scope": "cmrg_source",
                    "feature": forbidden,
                    "reason": "forbidden_token_in_scoring_source",
                }
            )
    return violations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require_features", action="store_true")
    parser.add_argument("--skip_source_audit", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_jsonl(args.fpath)
    if args.offset or args.limit is not None:
        data = data[
            args.offset : None if args.limit is None else args.offset + args.limit
        ]

    feature_doc_count = 0
    violations = []
    for sample_index, sample in enumerate(data, start=args.offset):
        sample_feature_docs, sample_violations = audit_sample(sample, sample_index)
        feature_doc_count += sample_feature_docs
        violations.extend(sample_violations)
    source_violations = [] if args.skip_source_audit else audit_cmrg_source()
    violations.extend(source_violations)

    result = {
        "samples": len(data),
        "feature_doc_count": feature_doc_count,
        "allowed_feature_keys": sorted(ALLOWED_CMRG_FEATURE_KEYS),
        "source_audit_enabled": not args.skip_source_audit,
        "source_forbidden_reference_count": len(source_violations),
        "forbidden_feature_count": len(violations),
        "violations": violations,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")

    if violations:
        raise SystemExit("Forbidden CMRG feature detected")
    if args.require_features and feature_doc_count == 0:
        raise SystemExit("No cmrg_features found; rerun replay with --debug_features")


if __name__ == "__main__":
    main()
