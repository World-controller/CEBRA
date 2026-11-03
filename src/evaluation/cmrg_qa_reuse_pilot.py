import argparse
import json
from pathlib import Path


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer_value(answer) -> str:
    if isinstance(answer, list):
        return " ".join(str(item) for item in answer)
    return str(answer)


def extract_context_items(sample: dict) -> list[dict]:
    has_cmrg_keep = any(
        "keep_for_qa" in chunk
        for iter_idx in range(4)
        if str(iter_idx) in sample
        for chunk in sample[str(iter_idx)].get("docs", [])
    )
    context_items = []
    seen = set()
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for chunk in sample[iter_key].get("docs", []):
            if has_cmrg_keep and "keep_for_qa" in chunk:
                should_use = bool(chunk["keep_for_qa"])
            else:
                should_use = chunk.get("label") != "<TERMINATE>"
            if not should_use:
                continue
            key = chunk.get("id") or chunk.get("text")
            if key in seen:
                continue
            seen.add(key)
            context_items.append(
                {
                    "id": str(chunk.get("id", "")),
                    "title": str(chunk.get("title", "")),
                    "text": str(chunk.get("text", "")),
                }
            )
    return context_items


def prompt_signature(sample: dict) -> dict:
    return {
        "query": str(sample.get("query", "")),
        "context": extract_context_items(sample),
    }


def alignment_violations(
    baseline_sample: dict,
    cmrg_sample: dict,
    qa_sample: dict,
    correctness_sample: dict,
    idx: int,
) -> list[dict]:
    violations = []
    baseline_question = str(baseline_sample.get("query", ""))
    cmrg_question = str(cmrg_sample.get("query", ""))
    qa_question = str(qa_sample.get("question", ""))
    correctness_question = str(correctness_sample.get("question", ""))
    baseline_answer = normalize_answer_value(baseline_sample.get("answer", ""))
    cmrg_answer = normalize_answer_value(cmrg_sample.get("answer", ""))
    qa_answer = normalize_answer_value(qa_sample.get("answer", ""))
    correctness_answer = normalize_answer_value(correctness_sample.get("answer", ""))

    if len({baseline_question, cmrg_question, qa_question, correctness_question}) != 1:
        violations.append(
            {
                "sample_index": idx,
                "field": "question",
                "baseline": baseline_question,
                "cmrg": cmrg_question,
                "qa": qa_question,
                "correctness": correctness_question,
            }
        )
    if len({baseline_answer, cmrg_answer, qa_answer, correctness_answer}) != 1:
        violations.append(
            {
                "sample_index": idx,
                "field": "answer",
                "baseline": baseline_answer,
                "cmrg": cmrg_answer,
                "qa": qa_answer,
                "correctness": correctness_answer,
            }
        )
    return violations


def score_correctness(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0, "em": 0.0, "f1": 0.0, "accuracy": 0.0}
    return {
        "samples": len(rows),
        "em": sum(float(row.get("em", 0.0)) for row in rows) / len(rows),
        "f1": sum(float(row.get("f1", 0.0)) for row in rows) / len(rows),
        "accuracy": sum(1.0 if is_correct(row.get("correctness")) else 0.0 for row in rows)
        / len(rows),
    }


def is_correct(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def count_empty_outputs(rows: list[dict]) -> int:
    return sum(1 for row in rows if not str(row.get("model_output", "")).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_retrieval", required=True)
    parser.add_argument("--cmrg_retrieval", required=True)
    parser.add_argument("--baseline_qa", required=True)
    parser.add_argument("--baseline_correctness", required=True)
    parser.add_argument("--output_dir", default="refine-logs/m2")
    parser.add_argument("--limit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_retrieval = load_jsonl(Path(args.baseline_retrieval), args.limit)
    cmrg_retrieval = load_jsonl(Path(args.cmrg_retrieval), args.limit)
    baseline_qa = load_jsonl(Path(args.baseline_qa), args.limit)
    baseline_correctness = load_jsonl(Path(args.baseline_correctness), args.limit)

    if not (
        len(baseline_retrieval)
        == len(cmrg_retrieval)
        == len(baseline_qa)
        == len(baseline_correctness)
    ):
        raise SystemExit("Input lengths differ; cannot safely reuse QA outputs.")

    changed = []
    alignment_errors = []
    for idx, (base_sample, cmrg_sample) in enumerate(
        zip(baseline_retrieval, cmrg_retrieval)
    ):
        qa_sample = baseline_qa[idx]
        correctness_sample = baseline_correctness[idx]
        alignment_errors.extend(
            alignment_violations(
                base_sample, cmrg_sample, qa_sample, correctness_sample, idx
            )
        )
        base_signature = prompt_signature(base_sample)
        cmrg_signature = prompt_signature(cmrg_sample)
        if base_signature != cmrg_signature:
            changed.append(
                {
                    "sample_index": idx,
                    "baseline_k": len(base_signature["context"]),
                    "cmrg_k": len(cmrg_signature["context"]),
                    "baseline_query": base_signature["query"],
                    "cmrg_query": cmrg_signature["query"],
                    "baseline_context_prefix": base_signature["context"][:3],
                    "cmrg_context_prefix": cmrg_signature["context"][:3],
                }
            )

    output_dir = Path(args.output_dir)
    summary = {
        "samples": len(baseline_qa),
        "alignment_error_count": len(alignment_errors),
        "alignment_errors": alignment_errors[:20],
        "context_changed_count": len(changed),
        "context_changes": changed[:20],
        "reuse_allowed": len(changed) == 0 and len(alignment_errors) == 0,
        "baseline_metrics": score_correctness(baseline_correctness),
        "cmrg_zero_metrics": score_correctness(baseline_correctness)
        if len(changed) == 0 and len(alignment_errors) == 0
        else None,
        "baseline_empty_outputs": count_empty_outputs(baseline_qa),
        "cmrg_zero_empty_outputs": count_empty_outputs(baseline_qa)
        if len(changed) == 0 and len(alignment_errors) == 0
        else None,
        "note": (
            "CMRG-zero QA outputs are reused from baseline because deduplicated QA "
            "contexts are identical for every sampled item."
        )
        if len(changed) == 0 and len(alignment_errors) == 0
        else "Alignment or contexts differ; run generator for CMRG-zero instead of reusing outputs.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "R007_hotpot500_baseline_qa_metrics.json").write_text(
        json.dumps(summary["baseline_metrics"], indent=2), encoding="utf-8"
    )
    write_jsonl(baseline_qa, output_dir / "R007_hotpot500_baseline_qa_results.jsonl")
    write_jsonl(
        baseline_correctness,
        output_dir / "R007_hotpot500_baseline_qa_correctness.jsonl",
    )

    if changed or alignment_errors:
        (output_dir / "R008_context_diff_requires_generation.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        raise SystemExit("CMRG alignment/context check failed; generator run required.")

    (output_dir / "R008_hotpot500_cmrg_zero_qa_metrics.json").write_text(
        json.dumps(summary["cmrg_zero_metrics"], indent=2), encoding="utf-8"
    )
    write_jsonl(
        baseline_qa,
        output_dir / "R008_hotpot500_cmrg_zero_qa_results_reused.jsonl",
    )
    write_jsonl(
        baseline_correctness,
        output_dir / "R008_hotpot500_cmrg_zero_qa_correctness_reused.jsonl",
    )
    (output_dir / "R007_R008_hotpot500_qa_reuse_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
