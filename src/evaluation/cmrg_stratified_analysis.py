import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.append(os.path.dirname(__file__))

import cmrg_retrieve_metrics as retrieval_metrics


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def answer_text(row: dict[str, Any]) -> str:
    value = row.get("answer", "")
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


def model_output(row: dict[str, Any]) -> str:
    return str(row.get("model_output") or row.get("prediction") or row.get("origin_prediction") or "")


def final_context_stats(sample: dict[str, Any]) -> dict[str, Any]:
    chunks, raw_kept_count, duplicate_count = retrieval_metrics.extract_final_chunks(sample)
    oracle = retrieval_metrics.oracle_ids(sample)
    return {
        "chunk_ids": chunks,
        "k": len(chunks),
        "raw_kept_count": raw_kept_count,
        "duplicate_count": duplicate_count,
        "duplicate_ratio": duplicate_count / raw_kept_count if raw_kept_count else 0.0,
        "recall": retrieval_metrics.recall(oracle, chunks),
    }


def compare_float(left: float, right: float, eps: float = 1e-12) -> str:
    if left > right + eps:
        return "full_win"
    if right > left + eps:
        return "budget_win"
    return "tie"


def safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "mean_delta_em_full_minus_budget": safe_mean([row["delta_em"] for row in rows]),
        "mean_delta_f1_full_minus_budget": safe_mean([row["delta_f1"] for row in rows]),
        "mean_delta_accuracy_full_minus_budget": safe_mean(
            [row["delta_accuracy"] for row in rows]
        ),
        "mean_full_k": safe_mean([row["full"]["retrieval"]["k"] for row in rows]),
        "mean_budget_k": safe_mean([row["budget"]["retrieval"]["k"] for row in rows]),
        "mean_full_recall": safe_mean([row["full"]["retrieval"]["recall"] for row in rows]),
        "mean_budget_recall": safe_mean(
            [row["budget"]["retrieval"]["recall"] for row in rows]
        ),
        "mean_full_duplicate_ratio": safe_mean(
            [row["full"]["retrieval"]["duplicate_ratio"] for row in rows]
        ),
        "mean_budget_duplicate_ratio": safe_mean(
            [row["budget"]["retrieval"]["duplicate_ratio"] for row in rows]
        ),
        "mean_context_jaccard": safe_mean([row["context_jaccard"] for row in rows]),
        "mean_answer_length": safe_mean([row["answer_length"] for row in rows]),
        "mean_question_length": safe_mean([row["question_length"] for row in rows]),
    }


def validate_alignment(
    name: str,
    retrieval_rows: list[dict[str, Any]],
    qa_rows: list[dict[str, Any]],
    correctness_rows: list[dict[str, Any]],
) -> None:
    if not (len(retrieval_rows) == len(qa_rows) == len(correctness_rows)):
        raise SystemExit(
            f"{name} row count mismatch: retrieval={len(retrieval_rows)}, "
            f"qa={len(qa_rows)}, correctness={len(correctness_rows)}"
        )
    for idx, (retrieval_row, qa_row, correctness_row) in enumerate(
        zip(retrieval_rows, qa_rows, correctness_rows)
    ):
        retrieval_question = str(retrieval_row.get("query") or retrieval_row.get("question") or "")
        qa_question = str(qa_row.get("question") or qa_row.get("query") or "")
        correctness_question = str(
            correctness_row.get("question") or correctness_row.get("query") or ""
        )
        if len({retrieval_question, qa_question, correctness_question}) != 1:
            raise SystemExit(
                f"{name} question mismatch at row {idx}: "
                f"{retrieval_question[:80]!r} / {qa_question[:80]!r} / "
                f"{correctness_question[:80]!r}"
            )


def build_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    full_retrieval = load_jsonl(args.full_retrieval)
    full_qa = load_jsonl(args.full_qa)
    full_correctness = load_jsonl(args.full_correctness)
    budget_retrieval = load_jsonl(args.budget_retrieval)
    budget_qa = load_jsonl(args.budget_qa)
    budget_correctness = load_jsonl(args.budget_correctness)

    validate_alignment("full", full_retrieval, full_qa, full_correctness)
    validate_alignment("budget", budget_retrieval, budget_qa, budget_correctness)
    if len(full_retrieval) != len(budget_retrieval):
        raise SystemExit("Full and budget rows must have identical length.")

    rows = []
    for idx, (
        full_ret,
        full_qa_row,
        full_corr,
        budget_ret,
        budget_qa_row,
        budget_corr,
    ) in enumerate(
        zip(
            full_retrieval,
            full_qa,
            full_correctness,
            budget_retrieval,
            budget_qa,
            budget_correctness,
        )
    ):
        question = str(full_qa_row.get("question") or full_ret.get("query") or "")
        budget_question = str(budget_qa_row.get("question") or budget_ret.get("query") or "")
        if question != budget_question:
            raise SystemExit(f"Full/budget question mismatch at row {idx}.")

        full_stats = final_context_stats(full_ret)
        budget_stats = final_context_stats(budget_ret)
        full_chunks = set(full_stats["chunk_ids"])
        budget_chunks = set(budget_stats["chunk_ids"])
        union = full_chunks | budget_chunks
        context_jaccard = len(full_chunks & budget_chunks) / len(union) if union else 1.0

        full_em = float(full_corr.get("em", 0.0))
        budget_em = float(budget_corr.get("em", 0.0))
        full_f1 = float(full_corr.get("f1", 0.0))
        budget_f1 = float(budget_corr.get("f1", 0.0))
        full_acc = 1.0 if is_correct(full_corr.get("correctness")) else 0.0
        budget_acc = 1.0 if is_correct(budget_corr.get("correctness")) else 0.0

        rows.append(
            {
                "index": idx,
                "question": question,
                "answer": answer_text(full_corr),
                "answer_length": len(answer_text(full_corr).split()),
                "question_length": len(question.split()),
                "full": {
                    "prediction": model_output(full_qa_row),
                    "em": full_em,
                    "f1": full_f1,
                    "accuracy": full_acc,
                    "retrieval": full_stats,
                },
                "budget": {
                    "prediction": model_output(budget_qa_row),
                    "em": budget_em,
                    "f1": budget_f1,
                    "accuracy": budget_acc,
                    "retrieval": budget_stats,
                },
                "delta_em": full_em - budget_em,
                "delta_f1": full_f1 - budget_f1,
                "delta_accuracy": full_acc - budget_acc,
                "em_outcome": compare_float(full_em, budget_em),
                "f1_outcome": compare_float(full_f1, budget_f1),
                "accuracy_outcome": compare_float(full_acc, budget_acc),
                "context_jaccard": context_jaccard,
            }
        )
    return rows


def aggregate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {
        "em_full_win": [row for row in rows if row["em_outcome"] == "full_win"],
        "em_budget_win": [row for row in rows if row["em_outcome"] == "budget_win"],
        "em_tie": [row for row in rows if row["em_outcome"] == "tie"],
        "f1_full_win": [row for row in rows if row["f1_outcome"] == "full_win"],
        "f1_budget_win": [row for row in rows if row["f1_outcome"] == "budget_win"],
        "f1_tie": [row for row in rows if row["f1_outcome"] == "tie"],
        "accuracy_full_win": [
            row for row in rows if row["accuracy_outcome"] == "full_win"
        ],
        "accuracy_budget_win": [
            row for row in rows if row["accuracy_outcome"] == "budget_win"
        ],
        "accuracy_tie": [row for row in rows if row["accuracy_outcome"] == "tie"],
    }
    summary = {
        "dataset": args.dataset,
        "rows": len(rows),
        "overall": summarize_group(rows),
        "win_loss_tie": {name: len(group_rows) for name, group_rows in groups.items()},
        "groups": {name: summarize_group(group_rows) for name, group_rows in groups.items()},
        "interpretation_flags": {
            "full_beats_budget_on_mean_em": safe_mean([row["delta_em"] for row in rows])
            > 0,
            "full_beats_budget_on_mean_f1": safe_mean([row["delta_f1"] for row in rows])
            > 0,
            "full_beats_budget_on_accuracy": safe_mean(
                [row["delta_accuracy"] for row in rows]
            )
            > 0,
            "budget_is_strong_control": (
                abs(safe_mean([row["delta_f1"] for row in rows])) < args.small_delta
                or safe_mean([row["delta_f1"] for row in rows]) < 0
            ),
        },
    }
    return summary


def selected_cases(rows: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    def trim(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "index": row["index"],
            "question": row["question"],
            "answer": row["answer"],
            "full_prediction": row["full"]["prediction"],
            "budget_prediction": row["budget"]["prediction"],
            "full_em": row["full"]["em"],
            "budget_em": row["budget"]["em"],
            "full_f1": row["full"]["f1"],
            "budget_f1": row["budget"]["f1"],
            "delta_em": row["delta_em"],
            "delta_f1": row["delta_f1"],
            "full_recall": row["full"]["retrieval"]["recall"],
            "budget_recall": row["budget"]["retrieval"]["recall"],
            "full_k": row["full"]["retrieval"]["k"],
            "budget_k": row["budget"]["retrieval"]["k"],
            "context_jaccard": row["context_jaccard"],
        }

    return {
        "full_largest_f1_wins": [
            trim(row) for row in sorted(rows, key=lambda item: item["delta_f1"], reverse=True)[:limit]
        ],
        "budget_largest_f1_wins": [
            trim(row) for row in sorted(rows, key=lambda item: item["delta_f1"])[:limit]
        ],
        "changed_em_cases": [
            trim(row) for row in rows if row["em_outcome"] != "tie"
        ][: limit * 2],
    }


def write_markdown(
    output_path: Path,
    summary: dict[str, Any],
    cases: dict[str, list[dict[str, Any]]],
) -> None:
    lines = [
        f"# CMRG Stratified Analysis: {summary['dataset']}",
        "",
        "## Overall",
        "",
        f"- Rows: `{summary['rows']}`",
        f"- Mean ΔEM full-budget: `{summary['overall']['mean_delta_em_full_minus_budget']:.6f}`",
        f"- Mean ΔF1 full-budget: `{summary['overall']['mean_delta_f1_full_minus_budget']:.6f}`",
        f"- Mean ΔAccuracy full-budget: `{summary['overall']['mean_delta_accuracy_full_minus_budget']:.6f}`",
        f"- Mean context Jaccard: `{summary['overall']['mean_context_jaccard']:.4f}`",
        "",
        "## Win / Loss / Tie",
        "",
        "| Metric | Full wins | Budget wins | Ties |",
        "|---|---:|---:|---:|",
        (
            f"| EM | {summary['win_loss_tie']['em_full_win']} | "
            f"{summary['win_loss_tie']['em_budget_win']} | {summary['win_loss_tie']['em_tie']} |"
        ),
        (
            f"| F1 | {summary['win_loss_tie']['f1_full_win']} | "
            f"{summary['win_loss_tie']['f1_budget_win']} | {summary['win_loss_tie']['f1_tie']} |"
        ),
        (
            f"| Accuracy | {summary['win_loss_tie']['accuracy_full_win']} | "
            f"{summary['win_loss_tie']['accuracy_budget_win']} | "
            f"{summary['win_loss_tie']['accuracy_tie']} |"
        ),
        "",
        "## Group Summaries",
        "",
        "| Group | Count | ΔEM | ΔF1 | Full K | Budget K | Context Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name in [
        "f1_full_win",
        "f1_budget_win",
        "f1_tie",
        "em_full_win",
        "em_budget_win",
        "em_tie",
    ]:
        group = summary["groups"][group_name]
        lines.append(
            f"| {group_name} | {group['count']} | "
            f"{group['mean_delta_em_full_minus_budget']:.6f} | "
            f"{group['mean_delta_f1_full_minus_budget']:.6f} | "
            f"{group['mean_full_k']:.3f} | {group['mean_budget_k']:.3f} | "
            f"{group['mean_context_jaccard']:.4f} |"
        )
    lines.extend(["", "## Representative Cases", ""])
    for case_group, case_rows in cases.items():
        lines.extend([f"### {case_group}", ""])
        for row in case_rows[:5]:
            lines.extend(
                [
                    f"- Index `{row['index']}` ΔF1 `{row['delta_f1']:.4f}` ΔEM `{row['delta_em']:.1f}`",
                    f"  - Q: {row['question']}",
                    f"  - Gold: {row['answer']}",
                    f"  - Full: {row['full_prediction']}",
                    f"  - Budget: {row['budget_prediction']}",
                    f"  - Recall full/budget: `{row['full_recall']:.3f}` / `{row['budget_recall']:.3f}`, context Jaccard `{row['context_jaccard']:.3f}`",
                ]
            )
        lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full_retrieval", required=True)
    parser.add_argument("--full_qa", required=True)
    parser.add_argument("--full_correctness", required=True)
    parser.add_argument("--budget_retrieval", required=True)
    parser.add_argument("--budget_qa", required=True)
    parser.add_argument("--budget_correctness", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--cases_json", required=True)
    parser.add_argument("--case_limit", type=int, default=10)
    parser.add_argument("--small_delta", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_rows(args)
    summary = aggregate(args, rows)
    cases = selected_cases(rows, args.case_limit)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    cases_json = Path(args.cases_json)
    cases_json.parent.mkdir(parents=True, exist_ok=True)
    cases_json.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    write_markdown(Path(args.output_md), summary, cases)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
