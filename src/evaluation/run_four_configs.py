from __future__ import annotations

import argparse
import gc
import json
import pickle
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cmrg_calibrated as calibrated


DEFAULT_THRESHOLDS = "0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"
REPLAY_FIELDS = {
    "keep_for_qa",
    "use_for_next_query",
    "chain_marginal_score",
    "retention_score",
    "retention_features",
    "cmrg_features",
}


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def iter_docs(sample: dict):
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        for chunk in sample[iter_key].get("docs", []):
            yield chunk


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def path_ready(path: Path, skip_existing: bool) -> bool:
    return skip_existing and path.exists() and path.stat().st_size > 0


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, check=True)


def clean_sample(sample: dict, removed_counts: dict[str, int]) -> dict:
    row = deepcopy(sample)
    for chunk in iter_docs(row):
        for field in REPLAY_FIELDS:
            if field in chunk:
                removed_counts[field] += 1
                del chunk[field]
    return row


def write_clean_substrate(
    source: str,
    output: Path,
    *,
    offset: int,
    limit: int | None,
    skip_existing: bool,
) -> dict[str, Any]:
    if path_ready(output, skip_existing):
        return {
            "source": source,
            "path": str(output),
            "offset": offset,
            "limit": limit,
            "samples": count_jsonl_rows(output),
            "removed_replay_fields": None,
            "skipped_existing": True,
        }

    removed_counts = {name: 0 for name in sorted(REPLAY_FIELDS)}
    samples = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for sample in calibrated.iter_jsonl_slice(source, offset, limit):
            cleaned = clean_sample(sample, removed_counts)
            f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            samples += 1
            del cleaned, sample
    gc.collect()
    return {
        "source": source,
        "path": str(output),
        "offset": offset,
        "limit": limit,
        "samples": samples,
        "removed_replay_fields": removed_counts,
        "skipped_existing": False,
    }


def controls_paths(output_dir: Path, dataset_tag: str) -> dict[str, Path]:
    return {
        "budget_only": output_dir / f"budget_only_{dataset_tag}_replay.jsonl",
        "budget_redundancy_hybrid": output_dir
        / f"budget_redundancy_hybrid_{dataset_tag}_replay.jsonl",
    }


def run_controls(
    *,
    python_bin: str,
    clean_eval_path: Path,
    output_dir: Path,
    dataset_tag: str,
    target_final_k: int,
    skip_existing: bool,
) -> dict[str, str]:
    paths = controls_paths(output_dir, dataset_tag)
    if all(path_ready(path, skip_existing) for path in paths.values()):
        return {name: str(path) for name, path in paths.items()}

    script_path = Path(__file__).with_name("cmrg_controls.py")
    command = [
        python_bin,
        str(script_path),
        "--baseline",
        str(clean_eval_path),
        "--output_dir",
        str(output_dir),
        "--target_final_k",
        str(target_final_k),
        "--file_tag",
        dataset_tag,
        "--manifest_name",
        f"{dataset_tag}_control_manifest.json",
        "--variants",
        "budget_only,budget_redundancy_hybrid",
    ]
    run_subprocess(command)
    return {name: str(path) for name, path in paths.items()}


def train_model(
    *,
    python_bin: str,
    clean_train_path: Path,
    output_path: Path,
    summary_path: Path,
    train_limit: int | None,
    seed: int,
    skip_existing: bool,
    drop_features: str = "",
) -> None:
    if path_ready(output_path, skip_existing):
        return

    script_path = Path(__file__).with_name("cmrg_calibrated.py")
    command = [
        python_bin,
        str(script_path),
        "train",
        "--fpath",
        str(clean_train_path),
        "--output",
        str(output_path),
        "--summary_output",
        str(summary_path),
        "--train_offset",
        "0",
        "--max_iter",
        "1000",
        "--seed",
        str(seed),
    ]
    if train_limit is not None:
        command.extend(["--train_limit", str(train_limit)])
    if drop_features:
        command.extend(["--drop_features", drop_features])
    run_subprocess(command)


def load_pickle_feature_names(path: Path) -> list[str]:
    with path.open("rb") as f:
        artifact = pickle.load(f)
    return list(artifact["feature_names"])


def changed_rows(reference_rows: list[dict], candidate_rows: list[dict]) -> int:
    return sum(
        1
        for reference, candidate in zip(reference_rows, candidate_rows)
        if calibrated.prompt_signature(reference) != calibrated.prompt_signature(candidate)
    )


def changed_rows_from_paths(reference_path: Path, candidate_path: Path) -> int:
    changed = 0
    with reference_path.open("r", encoding="utf-8") as left, candidate_path.open(
        "r", encoding="utf-8"
    ) as right:
        for left_line, right_line in zip(left, right):
            reference = json.loads(left_line)
            candidate = json.loads(right_line)
            if calibrated.prompt_signature(reference) != calibrated.prompt_signature(candidate):
                changed += 1
            del reference, candidate
    return changed


def controls_rewrites_all_keep_flags(path: Path) -> bool:
    for sample in calibrated.iter_jsonl_slice(str(path)):
        for chunk in iter_docs(sample):
            if "keep_for_qa" not in chunk:
                return False
        del sample
    return True


def choose_candidate(
    candidates: list[dict[str, Any]],
    *,
    budget_metrics: dict[str, Any],
    max_recall_drop: float,
) -> dict[str, Any]:
    recall_floor = budget_metrics["recall"] - max_recall_drop
    empty_cap = budget_metrics["empty_final_context_count"]

    # min_keep 是兜底机制而非主操作模式：优先只在 min_keep=0 的候选里完成选择，
    # 否则 empty gate 与 K 匹配的交互会把选择挤到强制保留的高重复退化点。
    primary = [item for item in candidates if item["min_keep_per_iter"] == 0]
    pool: list[dict[str, Any]] | None = None
    relaxed_empty_gate = True
    for cand_pool in ([primary] if primary else []) + [candidates]:
        strict = [
            item
            for item in cand_pool
            if item["metrics"]["recall"] >= recall_floor
            and item["metrics"]["empty_final_context_count"] <= empty_cap
        ]
        relaxed = [
            item for item in cand_pool if item["metrics"]["recall"] >= recall_floor
        ]
        if strict:
            pool = strict
            relaxed_empty_gate = False
            break
        if relaxed:
            pool = relaxed
            relaxed_empty_gate = True
            break
    if pool is None:
        pool = candidates
        relaxed_empty_gate = True

    target_k = budget_metrics["final_context_k"]
    selected = sorted(
        pool,
        key=lambda item: (
            abs(item["metrics"]["final_context_k"] - target_k),
            -item["metrics"]["recall"],
            item["metrics"]["empty_final_context_count"],
            item["threshold"],
            item["min_keep_per_iter"],
        ),
    )[0]
    return {**selected, "relaxed_empty_gate": relaxed_empty_gate}


def sweep_for_budget_match(
    clean_eval_path: Path,
    budget_path: Path,
    *,
    nlp: Any,
    model: Any,
    feature_names: list[str],
    thresholds: list[float],
    min_keep_values: list[int],
    max_recall_drop: float,
    sweep_limit: int | None,
) -> dict[str, Any]:
    rows = calibrated.load_jsonl_slice(str(clean_eval_path), 0, sweep_limit)
    budget_rows = calibrated.load_jsonl_slice(str(budget_path), 0, sweep_limit)
    budget_metrics = calibrated.retrieval_metrics(budget_rows)
    candidates: list[dict[str, Any]] = []

    for min_keep in min_keep_values:
        for threshold in thresholds:
            replayed = [
                calibrated.replay_sample_calibrated(
                    deepcopy(sample),
                    nlp=nlp,
                    model=model,
                    feature_names=feature_names,
                    threshold=threshold,
                    min_keep_per_iter=min_keep,
                    debug_features=False,
                )
                for sample in rows
            ]
            metrics = calibrated.retrieval_metrics(replayed)
            candidates.append(
                {
                    "threshold": threshold,
                    "min_keep_per_iter": min_keep,
                    "metrics": metrics,
                    "changed_rows_vs_budget_only": changed_rows(budget_rows, replayed),
                }
            )
            del replayed
            gc.collect()

    selected = choose_candidate(
        candidates,
        budget_metrics=budget_metrics,
        max_recall_drop=max_recall_drop,
    )
    result = {
        "sweep_limit": sweep_limit,
        "sweep_samples": len(rows),
        "budget_only_metrics": budget_metrics,
        "selected": selected,
        "candidates": candidates,
    }
    del rows, budget_rows
    gc.collect()
    return result


def replay_to_file_streaming(
    clean_eval_path: Path,
    *,
    nlp: Any,
    model: Any,
    feature_names: list[str],
    threshold: float,
    min_keep_per_iter: int,
    output_path: Path,
    skip_existing: bool,
) -> None:
    if path_ready(output_path, skip_existing):
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in calibrated.iter_jsonl_slice(str(clean_eval_path)):
            row = calibrated.replay_sample_calibrated(
                sample,
                nlp=nlp,
                model=model,
                feature_names=feature_names,
                threshold=threshold,
                min_keep_per_iter=min_keep_per_iter,
                debug_features=False,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            del row, sample
    gc.collect()


class MetricsAccumulator:
    def __init__(self) -> None:
        self.samples = 0
        self.recall_sum = 0.0
        self.final_context_k_sum = 0.0
        self.raw_kept_count_sum = 0.0
        self.raw_total = 0
        self.duplicate_total = 0
        self.empty_final_context_count = 0

    def add(self, sample: dict) -> None:
        chunks, raw_kept_count, duplicate_count = calibrated.extract_final_chunks(sample)
        if not chunks:
            self.empty_final_context_count += 1
        self.samples += 1
        self.recall_sum += calibrated.recall(calibrated.sample_oracle_ids(sample), chunks)
        self.final_context_k_sum += len(chunks)
        self.raw_kept_count_sum += raw_kept_count
        self.raw_total += raw_kept_count
        self.duplicate_total += duplicate_count

    def result(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "recall": self.recall_sum / self.samples if self.samples else 0.0,
            "final_context_k": self.final_context_k_sum / self.samples
            if self.samples
            else 0.0,
            "raw_kept_count": self.raw_kept_count_sum / self.samples
            if self.samples
            else 0.0,
            "duplicate_ratio": float(self.duplicate_total) / max(float(self.raw_total), 1.0),
            "empty_final_context_count": self.empty_final_context_count,
        }


def metrics_for_path(path: Path) -> dict[str, Any]:
    accumulator = MetricsAccumulator()
    for sample in calibrated.iter_jsonl_slice(str(path)):
        accumulator.add(sample)
        del sample
    return accumulator.result()


def summarize_path(
    label: str,
    path: Path,
    *,
    budget_path: Path,
    selected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = metrics_for_path(path)
    item: dict[str, Any] = {
        "config": label,
        "recall": metrics["recall"],
        "final_context_k": metrics["final_context_k"],
        "duplicate_ratio": metrics["duplicate_ratio"],
        "raw_kept_count": metrics["raw_kept_count"],
        "empty_final_context_count": metrics["empty_final_context_count"],
        "changed_rows_vs_budget_only": changed_rows_from_paths(budget_path, path),
        "threshold": None,
        "min_keep_per_iter": None,
        "selection_note": "",
    }
    if selected is not None:
        item["threshold"] = selected["threshold"]
        item["min_keep_per_iter"] = selected["min_keep_per_iter"]
        if selected.get("relaxed_empty_gate"):
            item["selection_note"] = "未找到同时满足 recall 与 empty gate 的候选，已放宽 empty gate"
    return item


def markdown_table(summary_rows: list[dict[str, Any]]) -> str:
    headers = [
        "config",
        "recall",
        "final_context_k",
        "duplicate_ratio",
        "raw_kept_count",
        "empty_final_context_count",
        "changed_rows_vs_budget_only",
        "threshold",
        "min_keep_per_iter",
        "selection_note",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in summary_rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def output_paths(output_dir: Path, dataset_tag: str) -> dict[str, Path]:
    models_dir = output_dir / "models"
    return {
        "clean_eval_substrate": output_dir / f"{dataset_tag}_clean_substrate.jsonl",
        "clean_train_substrate": output_dir / f"{dataset_tag}_clean_train_substrate.jsonl",
        "budget_only_replay": output_dir / f"budget_only_{dataset_tag}_replay.jsonl",
        "budget_redundancy_hybrid_replay": output_dir
        / f"budget_redundancy_hybrid_{dataset_tag}_replay.jsonl",
        "calibrated_no_redundancy_replay": output_dir
        / f"calibrated_no_redundancy_{dataset_tag}_replay.jsonl",
        "calibrated_full_replay": output_dir
        / f"calibrated_full_{dataset_tag}_replay.jsonl",
        "calibrated_no_redundancy_model": models_dir
        / f"{dataset_tag}_calibrated_no_redundancy.pkl",
        "calibrated_full_model": models_dir / f"{dataset_tag}_calibrated_full.pkl",
        "calibrated_no_redundancy_summary": models_dir
        / f"{dataset_tag}_calibrated_no_redundancy_summary.json",
        "calibrated_full_summary": models_dir
        / f"{dataset_tag}_calibrated_full_summary.json",
        "summary_json": output_dir / f"{dataset_tag}_four_configs_summary.json",
        "summary_md": output_dir / f"{dataset_tag}_four_configs_summary.md",
    }


def run_four_configs(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = output_paths(output_dir, args.dataset_tag)
    python_bin = sys.executable

    # 净化历史回放字段；skip_existing 时直接复用磁盘文件，避免重读全量底料。
    eval_clean_manifest = write_clean_substrate(
        args.eval_substrate,
        paths["clean_eval_substrate"],
        offset=args.offset,
        limit=args.limit,
        skip_existing=args.skip_existing,
    )
    train_clean_manifest = write_clean_substrate(
        args.train_substrate,
        paths["clean_train_substrate"],
        offset=args.train_offset,
        limit=args.train_limit,
        skip_existing=args.skip_existing,
    )
    train_samples = train_clean_manifest["samples"]
    gc.collect()

    control_paths = run_controls(
        python_bin=python_bin,
        clean_eval_path=paths["clean_eval_substrate"],
        output_dir=output_dir,
        dataset_tag=args.dataset_tag,
        target_final_k=args.target_final_k,
        skip_existing=args.skip_existing,
    )
    budget_path = Path(control_paths["budget_only"])
    hybrid_path = Path(control_paths["budget_redundancy_hybrid"])

    train_model(
        python_bin=python_bin,
        clean_train_path=paths["clean_train_substrate"],
        output_path=paths["calibrated_full_model"],
        summary_path=paths["calibrated_full_summary"],
        train_limit=train_samples,
        seed=args.seed,
        skip_existing=args.skip_existing,
    )
    train_model(
        python_bin=python_bin,
        clean_train_path=paths["clean_train_substrate"],
        output_path=paths["calibrated_no_redundancy_model"],
        summary_path=paths["calibrated_no_redundancy_summary"],
        train_limit=train_samples,
        seed=args.seed,
        skip_existing=args.skip_existing,
        drop_features="redundancy",
    )
    gc.collect()

    nlp = calibrated.load_nlp()
    thresholds = parse_csv_floats(args.thresholds)
    min_keep_values = parse_csv_ints(args.min_keep_values)

    full_model, full_feature_names = calibrated.load_model(
        str(paths["calibrated_full_model"])
    )
    full_sweep = sweep_for_budget_match(
        paths["clean_eval_substrate"],
        budget_path,
        nlp=nlp,
        model=full_model,
        feature_names=full_feature_names,
        thresholds=thresholds,
        min_keep_values=min_keep_values,
        max_recall_drop=args.max_recall_drop,
        sweep_limit=args.sweep_limit,
    )
    replay_to_file_streaming(
        paths["clean_eval_substrate"],
        nlp=nlp,
        model=full_model,
        feature_names=full_feature_names,
        threshold=full_sweep["selected"]["threshold"],
        min_keep_per_iter=full_sweep["selected"]["min_keep_per_iter"],
        output_path=paths["calibrated_full_replay"],
        skip_existing=args.skip_existing,
    )
    del full_model
    calibrated.clear_term_cache()
    gc.collect()

    no_redundancy_model, no_redundancy_feature_names = calibrated.load_model(
        str(paths["calibrated_no_redundancy_model"])
    )
    no_redundancy_sweep = sweep_for_budget_match(
        paths["clean_eval_substrate"],
        budget_path,
        nlp=nlp,
        model=no_redundancy_model,
        feature_names=no_redundancy_feature_names,
        thresholds=thresholds,
        min_keep_values=min_keep_values,
        max_recall_drop=args.max_recall_drop,
        sweep_limit=args.sweep_limit,
    )
    replay_to_file_streaming(
        paths["clean_eval_substrate"],
        nlp=nlp,
        model=no_redundancy_model,
        feature_names=no_redundancy_feature_names,
        threshold=no_redundancy_sweep["selected"]["threshold"],
        min_keep_per_iter=no_redundancy_sweep["selected"]["min_keep_per_iter"],
        output_path=paths["calibrated_no_redundancy_replay"],
        skip_existing=args.skip_existing,
    )
    del no_redundancy_model
    calibrated.clear_term_cache()
    gc.collect()

    controls_rewrite_keep_for_qa = {
        "budget_only": controls_rewrites_all_keep_flags(budget_path),
        "budget_redundancy_hybrid": controls_rewrites_all_keep_flags(hybrid_path),
    }
    summary_rows = [
        summarize_path("A", paths["clean_eval_substrate"], budget_path=budget_path),
        summarize_path("A+B", budget_path, budget_path=budget_path),
        summarize_path("A+D", hybrid_path, budget_path=budget_path),
        summarize_path(
            "A+C",
            paths["calibrated_no_redundancy_replay"],
            budget_path=budget_path,
            selected=no_redundancy_sweep["selected"],
        ),
        summarize_path(
            "A+C+D",
            paths["calibrated_full_replay"],
            budget_path=budget_path,
            selected=full_sweep["selected"],
        ),
    ]

    output_files = {
        name: str(path)
        for name, path in paths.items()
        if name not in {"summary_json", "summary_md"}
    }
    result = {
        "dataset_tag": args.dataset_tag,
        "args": vars(args),
        "sweep_limit": args.sweep_limit,
        "cleaning": {
            "eval": eval_clean_manifest,
            "train": train_clean_manifest,
        },
        "controls_rewrite_keep_for_qa": controls_rewrite_keep_for_qa,
        "controls_rewrite_conclusion": (
            "cmrg_controls 会遍历每个 doc 写入 keep_for_qa；净化步骤仍然必要，"
            "因为所有 baseline 指标与 calibrated replay 必须先避开历史 keep 标志污染。"
        ),
        "model_feature_names": {
            "calibrated_no_redundancy": load_pickle_feature_names(
                paths["calibrated_no_redundancy_model"]
            ),
            "calibrated_full": load_pickle_feature_names(paths["calibrated_full_model"]),
        },
        "selection": {
            "A+C": no_redundancy_sweep["selected"],
            "A+C+D": full_sweep["selected"],
        },
        "summary_rows": summary_rows,
        "output_files": output_files,
    }
    paths["summary_json"].write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    paths["summary_md"].write_text(markdown_table(summary_rows), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键运行 A+B、A+D、A+C、A+C+D 四组预算控制配置。"
    )
    parser.add_argument("--eval_substrate", required=True)
    parser.add_argument("--train_substrate", required=True)
    parser.add_argument("--dataset_tag", required=True)
    parser.add_argument("--target_final_k", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--min_keep_values", default="0,1")
    parser.add_argument("--max_recall_drop", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--train_offset", type=int, default=0)
    parser.add_argument("--train_limit", type=int, default=None)
    parser.add_argument("--sweep_limit", type=int, default=1000)
    parser.add_argument("--skip_existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_four_configs(parse_args())


if __name__ == "__main__":
    main()
