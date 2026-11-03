#!/usr/bin/env python3
"""Apply Provence sentence pruning to CEBRA's frozen final contexts.

The script treats Provence as a post-selection compressor: retrieval, chunk
selection, chunk order, and query updates remain frozen.  Calibration and test
windows are enforced at load time.  A calibration run performs one model
forward pass and evaluates every threshold from the cached token scores.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import re
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "evaluation"))

from cmrg_controls import extract_final_chunks, stable_key  # noqa: E402

DEFAULT_MODEL = "naver/provence-reranker-debertav3-v1"
DEFAULT_REVISION = "ef49e233e3c6e50efc476c68f1390f8a63add4d4"
WORD_RE = re.compile(r"\w+")


def normalize_answer_text(text: str) -> str:
    text = text.lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_in_text(answer: str, text: str) -> float:
    normalized_answer = normalize_answer_text(answer)
    if not normalized_answer:
        return 0.0
    return float(normalized_answer in normalize_answer_text(text))


def load_rows(
    path: Path,
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
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if row_index < offset:
                continue
            if row_index >= offset + limit:
                break
            if line.strip():
                rows.append((row_index, json.loads(line)))
    if len(rows) != limit:
        raise ValueError(f"requested {limit} rows but read {len(rows)} from {path}")
    return rows


def build_model_inputs(
    indexed_rows: Sequence[tuple[int, dict[str, Any]]],
) -> tuple[list[str], list[list[str]]]:
    questions: list[str] = []
    contexts: list[list[str]] = []
    for _, row in indexed_rows:
        questions.append(str(row.get("query") or row.get("question") or ""))
        contexts.append([str(chunk.get("text", "")) for chunk in extract_final_chunks(row)])
    return questions, contexts


def cache_provence_scores(
    model: Any,
    questions: Sequence[str],
    contexts: Sequence[Sequence[str]],
    *,
    batch_size: int,
    enable_warnings: bool,
) -> tuple[Any, list[dict[str, Any]], float]:
    """Run Provence once and retain token scores needed for threshold sweeps."""
    import torch
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pad_sequence

    remote_module = sys.modules[model.__class__.__module__]
    dataset = remote_module.TestDataset(
        queries=list(questions),
        contexts=[list(items) for items in contexts],
        titles="first_sentence",
        tokenizer=model.tokenizer,
        max_len=model.max_len,
        enable_warnings=enable_warnings,
    )
    blocks: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            ids_list = dataset.ids[start:stop]
            ids = pad_sequence(
                ids_list,
                batch_first=True,
                padding_value=dataset.pad_idx,
            ).to(model.device)
            mask = (ids != dataset.pad_idx).to(model.device)
            outputs = model(ids, mask)
            scores = functional.softmax(outputs["compression_logits"].float(), dim=-1)[
                :, :, 1
            ].cpu()
            for position, ids_item in enumerate(ids_list):
                dataset_index = start + position
                length = len(ids_item)
                blocks.append(
                    {
                        "qi": int(dataset.qis[dataset_index]),
                        "ci": int(dataset.cis[dataset_index]),
                        "si": int(dataset.sis[dataset_index]),
                        "ids": ids_item.clone(),
                        "scores": scores[position, :length].clone(),
                        "sent_coords": np.asarray(
                            dataset.sent_coords[dataset_index], dtype=np.int64
                        ),
                    }
                )
            if stop == len(dataset) or stop % 100 < batch_size:
                print(
                    f"Provence inference: {stop}/{len(dataset)} model blocks",
                    flush=True,
                )
    return remote_module, blocks, time.perf_counter() - started


def decode_at_threshold(
    remote_module: Any,
    tokenizer: Any,
    contexts: Sequence[Sequence[str]],
    blocks: Sequence[dict[str, Any]],
    threshold: float,
) -> list[list[str]]:
    """Reproduce ``Provence.process`` decoding from cached token scores."""
    import torch

    selected: dict[tuple[int, int], dict[int, Any]] = defaultdict(dict)
    for block in blocks:
        token_preds = block["scores"].numpy() > threshold
        selected_mask = remote_module.sentence_rounding(
            token_preds,
            block["sent_coords"],
            threshold=threshold,
            always_select_title=block["si"] == 0,
        )
        selected[(block["qi"], block["ci"])][block["si"]] = block["ids"][
            selected_mask[: len(block["ids"])]
        ]

    output: list[list[str]] = []
    for qi, context_list in enumerate(contexts):
        row_output: list[str] = []
        for ci, original in enumerate(context_list):
            pieces = selected.get((qi, ci))
            if not pieces:
                row_output.append(str(original))
                continue
            token_ids = torch.cat([value for _, value in sorted(pieces.items())])
            row_output.append(
                tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        output.append(row_output)
    return output


def apply_nonempty_fallback(
    original_contexts: Sequence[Sequence[str]],
    compressed_contexts: Sequence[Sequence[str]],
) -> tuple[list[list[str]], int]:
    """Restore the first original chunk only when pruning empties a nonempty row."""
    output = [list(items) for items in compressed_contexts]
    fallback_contexts = 0
    for originals, compressed in zip(original_contexts, output):
        if any(item.strip() for item in compressed) or not any(
            item.strip() for item in originals
        ):
            continue
        restore_index = next(
            index for index, item in enumerate(originals) if item.strip()
        )
        compressed[restore_index] = str(originals[restore_index])
        fallback_contexts += 1
    return output, fallback_contexts


def materialize_rows(
    indexed_rows: Sequence[tuple[int, dict[str, Any]]],
    compressed_contexts: Sequence[Sequence[str]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (row_index, source), row_contexts in zip(indexed_rows, compressed_contexts):
        row = copy.deepcopy(source)
        final_chunks = extract_final_chunks(row)
        if len(final_chunks) != len(row_contexts):
            raise ValueError(f"context alignment failed at source row {row_index}")
        for chunk, compressed_text in zip(final_chunks, row_contexts):
            original_text = str(chunk.get("text", ""))
            chunk["provence_original_text"] = original_text
            chunk["text"] = compressed_text
            chunk["provence_compression"] = {
                "threshold": threshold,
                "original_words": len(WORD_RE.findall(original_text)),
                "compressed_words": len(WORD_RE.findall(compressed_text)),
            }
        row["provence"] = {
            "source_row_index": row_index,
            "threshold": threshold,
            "selected_chunk_keys": [stable_key(chunk) for chunk in final_chunks],
            "scope": "sentence pruning after frozen CEBRA chunk selection",
        }
        output.append(row)
    return output


def summarize(
    indexed_rows: Sequence[tuple[int, dict[str, Any]]],
    compressed_contexts: Sequence[Sequence[str]],
    *,
    nonempty_fallback_contexts: int = 0,
) -> dict[str, Any]:
    original_words: list[int] = []
    compressed_words: list[int] = []
    original_aic: list[float] = []
    compressed_aic: list[float] = []
    is_yes_no: list[bool] = []
    original_empty = 0
    compressed_empty = 0
    compressed_empty_chunks = 0

    for (_, row), compressed in zip(indexed_rows, compressed_contexts):
        original = [str(chunk.get("text", "")) for chunk in extract_final_chunks(row)]
        original_text = "\n".join(original)
        compressed_text = "\n".join(compressed)
        original_words.append(len(WORD_RE.findall(original_text)))
        compressed_words.append(len(WORD_RE.findall(compressed_text)))
        answer = str(row.get("answer", ""))
        original_aic.append(answer_in_text(answer, original_text))
        compressed_aic.append(answer_in_text(answer, compressed_text))
        is_yes_no.append(normalize_answer_text(answer) in {"yes", "no"})
        original_empty += int(not original_text.strip())
        compressed_empty += int(not compressed_text.strip())
        compressed_empty_chunks += sum(not item.strip() for item in compressed)

    span_positions = [index for index, flag in enumerate(is_yes_no) if not flag]
    original_span = [original_aic[index] for index in span_positions]
    compressed_span = [compressed_aic[index] for index in span_positions]
    ratios = [
        compressed / original if original else 1.0
        for original, compressed in zip(original_words, compressed_words)
    ]
    return {
        "samples": len(indexed_rows),
        "span_questions": len(span_positions),
        "yes_no_questions": int(sum(is_yes_no)),
        "mean_original_words": float(np.mean(original_words)),
        "mean_compressed_words": float(np.mean(compressed_words)),
        "aggregate_word_ratio": float(sum(compressed_words) / max(sum(original_words), 1)),
        "mean_per_question_word_ratio": float(np.mean(ratios)),
        "compression_rate": float(1.0 - sum(compressed_words) / max(sum(original_words), 1)),
        "original_answer_in_context_span": float(np.mean(original_span))
        if original_span
        else 0.0,
        "compressed_answer_in_context_span": float(np.mean(compressed_span))
        if compressed_span
        else 0.0,
        "answer_in_context_delta_span": float(np.mean(compressed_span) - np.mean(original_span))
        if original_span
        else 0.0,
        "original_empty_contexts": original_empty,
        "compressed_empty_contexts": compressed_empty,
        "compressed_empty_chunks": int(compressed_empty_chunks),
        "nonempty_fallback_contexts": nonempty_fallback_contexts,
        "per_row": {
            "original_words": original_words,
            "compressed_words": compressed_words,
            "original_answer_in_context": original_aic,
            "compressed_answer_in_context": compressed_aic,
            "is_yes_no": is_yes_no,
        },
    }


def choose_threshold(
    sweep: dict[str, dict[str, Any]],
    *,
    max_aic_drop_pp: float,
) -> tuple[float, dict[str, Any]]:
    tolerance = max_aic_drop_pp / 100.0
    feasible = {
        key: metrics
        for key, metrics in sweep.items()
        if metrics["compressed_answer_in_context_span"]
        >= metrics["original_answer_in_context_span"] - tolerance - 1e-12
        and metrics["compressed_empty_contexts"] <= metrics["original_empty_contexts"]
    }
    pool = feasible or sweep
    if feasible:
        selected_key = min(
            pool,
            key=lambda key: (
                pool[key]["aggregate_word_ratio"],
                -pool[key]["compressed_answer_in_context_span"],
                pool[key]["compressed_empty_contexts"],
                float(key),
            ),
        )
    else:
        selected_key = max(
            pool,
            key=lambda key: (
                pool[key]["compressed_answer_in_context_span"],
                -pool[key]["compressed_empty_contexts"],
                -pool[key]["aggregate_word_ratio"],
                -float(key),
            ),
        )
    return float(selected_key), {
        "feasible_thresholds": [float(key) for key in feasible],
        "used_fallback": not bool(feasible),
        "criterion": (
            "minimum aggregate word ratio subject to span-only AiC drop <= "
            f"{max_aic_drop_pp:g} percentage point(s) and no increase in empty contexts"
        ),
        "tie_break": "higher compressed AiC, fewer empty contexts, lower threshold",
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in summary.items() if key != "per_row"}


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
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
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument(
        "--threshold-grid",
        default="",
        help="calibration-only comma-separated threshold sweep",
    )
    parser.add_argument("--max-aic-drop-pp", type=float, default=1.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--enable-model-warnings", action="store_true")
    parser.add_argument(
        "--nonempty-fallback",
        choices=("first", "none"),
        default="first",
        help="restore the first original retained chunk if pruning empties a nonempty row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    threshold_grid = tuple(
        float(part.strip()) for part in args.threshold_grid.split(",") if part.strip()
    )
    if any(not 0.0 <= value <= 1.0 for value in threshold_grid):
        raise ValueError("all --threshold-grid values must be in [0, 1]")
    if threshold_grid and args.split != "calibration":
        raise ValueError("--threshold-grid is calibration-only")
    if args.max_aic_drop_pp < 0:
        raise ValueError("--max-aic-drop-pp must be non-negative")

    indexed_rows = load_rows(
        args.input,
        offset=args.offset,
        limit=args.limit,
        split=args.split,
        calibration_rows=args.calibration_rows,
    )
    questions, contexts = build_model_inputs(indexed_rows)

    import torch
    from transformers import AutoModel

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

    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        args.model,
        revision=args.model_revision,
        trust_remote_code=True,
        local_files_only=not args.allow_model_download,
    ).to(device)
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    remote_module, blocks, inference_seconds = cache_provence_scores(
        model,
        questions,
        contexts,
        batch_size=args.batch_size,
        enable_warnings=args.enable_model_warnings,
    )

    sweep: dict[str, dict[str, Any]] = {}
    thresholds = threshold_grid or (args.threshold,)
    decoded_by_threshold: dict[float, list[list[str]]] = {}
    decoding_seconds: dict[str, float] = {}
    for threshold in thresholds:
        started = time.perf_counter()
        decoded = decode_at_threshold(
            remote_module,
            model.tokenizer,
            contexts,
            blocks,
            threshold,
        )
        fallback_contexts = 0
        if args.nonempty_fallback == "first":
            decoded, fallback_contexts = apply_nonempty_fallback(contexts, decoded)
        decoding_seconds[f"{threshold:.6g}"] = time.perf_counter() - started
        decoded_by_threshold[threshold] = decoded
        sweep[f"{threshold:.6g}"] = summarize(
            indexed_rows,
            decoded,
            nonempty_fallback_contexts=fallback_contexts,
        )

    selection: dict[str, Any] | None = None
    if threshold_grid:
        args.threshold, selection = choose_threshold(
            sweep,
            max_aic_drop_pp=args.max_aic_drop_pp,
        )
    selected_contexts = decoded_by_threshold[args.threshold]
    selected_summary = sweep[f"{args.threshold:.6g}"]
    output_rows = materialize_rows(
        indexed_rows,
        selected_contexts,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_rows, args.output_dir / "provence_replay.jsonl")
    manifest = {
        "protocol": {
            "input": str(args.input),
            "split": args.split,
            "offset": args.offset,
            "limit": args.limit,
            "calibration_rows": args.calibration_rows,
            "scope": "sentence pruning after frozen CEBRA chunk selection",
            "retrieval": "frozen",
            "chunk_selection": "frozen CEBRA keep_for_qa",
            "query_updates": "frozen",
            "chunk_order": "unchanged",
            "model": args.model,
            "model_revision": args.model_revision,
            "threshold": args.threshold,
            "threshold_grid": list(threshold_grid),
            "max_aic_drop_pp": args.max_aic_drop_pp,
            "always_select_title": True,
            "title_mode": "first_sentence",
            "nonempty_fallback": args.nonempty_fallback,
            "device": device,
            "batch_size": args.batch_size,
            "seed": 42,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git_revision": git_revision(),
            "torch": torch.__version__,
        },
        "timing_seconds": {
            "model_load": model_load_seconds,
            "model_inference": inference_seconds,
            "threshold_decoding": decoding_seconds,
            "inference_per_model_block": inference_seconds / max(len(blocks), 1),
        },
        "model_blocks": len(blocks),
        "selection": selection,
        "threshold_sweep": {
            key: compact_summary(value) for key, value in sweep.items()
        },
        "selected_result": selected_summary,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "threshold": args.threshold,
                "selection": selection,
                "model_blocks": len(blocks),
                "timing_seconds": manifest["timing_seconds"],
                "result": compact_summary(selected_summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
