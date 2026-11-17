#!/usr/bin/env python3
"""Public CEBRA entry point. Algorithms and serialized fields stay in their original modules."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NLP_PROFILE = {"spacy": "3.7.5", "model": "en_core_web_sm", "model_version": "3.7.0"}
SMOKE_PROFILE = {"name": "explicit-regex-smoke-only"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def new_output(path: Path, inputs: list[Path]) -> None:
    if path.resolve() in {p.resolve() for p in inputs}:
        raise ValueError("Output must not overwrite an input.")
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Choose a new output path.")


def engines():
    # Keep --help usable even before scientific dependencies are installed.
    sys.path.insert(0, str(ROOT / "src" / "evaluation"))
    import cmrg_calibrated as caes
    import cmrg_controls as controls
    import run_four_configs as baos
    import cmrg_retrieve_metrics as metrics
    return caes, controls, baos, metrics


def load_nlp(smoke: bool):
    if smoke:
        return None
    try:
        import spacy
        if spacy.__version__ != NLP_PROFILE["spacy"]:
            raise ValueError(f"CEBRA release NLP profile requires spaCy {NLP_PROFILE['spacy']}.")
        nlp = spacy.load(NLP_PROFILE["model"])
        if nlp.meta.get("version") != NLP_PROFILE["model_version"]:
            raise ValueError("Incorrect en_core_web_sm version for the release NLP profile.")
        if not {"parser", "ner", "lemmatizer"}.issubset(nlp.pipe_names):
            raise ValueError("The full parser, NER, and lemmatizer pipeline is required.")
        return nlp
    except (ImportError, OSError) as error:
        raise ValueError("Missing NLP dependency. Install requirements-cebra.txt and the pinned NLP wheel; no fallback is used.") from error


@contextmanager
def nlp_override(caes, nlp):
    previous = caes.load_nlp
    caes.clear_term_cache()
    caes.load_nlp = lambda: nlp
    try:
        yield
    finally:
        caes.load_nlp = previous
        caes.clear_term_cache()


def load_rows(path: Path, *, smoke: bool = False, gold: bool = False) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"Row {index} must be an object.")
        if smoke and row.get("_cebra_example") != "synthetic-smoke":
            raise ValueError("--smoke-example accepts only explicitly marked synthetic smoke rows.")
        if gold and not isinstance(row.get("oracle"), list):
            raise ValueError("Training/calibration requires an explicit oracle list on each row.")
        keys = sorted(int(k) for k in row if k.isdigit())
        if keys != list(range(len(keys))) or len(keys) > 4:
            raise ValueError("Trajectory rounds must be contiguous keys 0..3; longer trajectories are not silently truncated.")
        for key in keys:
            docs = row[str(key)].get("docs")
            if not isinstance(docs, list):
                raise ValueError("Each round must contain a docs list.")
            for doc in docs:
                if not isinstance(doc, dict) or not doc.get("id"):
                    raise ValueError("Each document must have a nonempty id.")
                if doc.get("label") not in {"<CONTINUE>", "<FINISH>", "<TERMINATE>"}:
                    raise ValueError("Each document requires a recognized upstream label.")
        rows.append(row)
    if not rows:
        raise ValueError("Input contains no questions (an individual empty context is supported).")
    return rows


def clean_rows(rows, baos):
    return [baos.clean_sample(row, {name: 0 for name in baos.REPLAY_FIELDS}) for row in rows]


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def model_metadata_path(model: Path) -> Path:
    return Path(str(model) + ".meta.json")


def load_checked_model(path: Path, metadata: dict, caes):
    # Pickle is executable: load only trusted artifacts, after checking bytes and runtime.
    if sha256(path) != metadata.get("model_sha256"):
        raise ValueError("Model SHA-256 does not match the declared artifact.")
    version = importlib.metadata.version("scikit-learn")
    if version != metadata.get("scikit_learn"):
        raise ValueError(f"Model requires scikit-learn {metadata.get('scikit_learn')}; found {version}.")
    model, names = caes.load_model(str(path))
    if names != caes.FEATURE_NAMES or names != metadata.get("feature_names"):
        raise ValueError("CEBRA requires the ordered nine-feature scorer; model feature mismatch.")
    if model.n_features_in_ != len(names) or list(model.classes_) != [0, 1]:
        raise ValueError("Invalid scorer dimensions or class order.")
    return model, names


def resolve_model(args, caes):
    config = read_json(args.config) if getattr(args, "config", None) else None
    if config is not None:
        if config.get("schema_version") != 1:
            raise ValueError("Unsupported CEBRA configuration version.")
        model_path = (args.config.parent / config["model"]).resolve()
        metadata = read_json(model_metadata_path(model_path))
        if metadata["model_sha256"] != config["model_sha256"]:
            raise ValueError("Configuration and model metadata disagree.")
    else:
        model_path = args.model.resolve()
        metadata = read_json(model_metadata_path(model_path))
    expected_nlp = SMOKE_PROFILE if args.smoke_example else NLP_PROFILE
    if not args.smoke_example and metadata.get("nlp_profile") != expected_nlp:
        raise ValueError("A smoke-trained or incompatible NLP-profile model cannot be used for formal runs.")
    model, names = load_checked_model(model_path, metadata, caes)
    return model_path, metadata, model, names, config


def provenance(args, model_path=None):
    return {
        "command": args.command,
        "input_sha256": sha256(args.fpath),
        "synthetic_smoke_example": any(
            json.loads(line).get("_cebra_example") == "synthetic-smoke"
            for line in args.fpath.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
        "regex_smoke_mode": bool(getattr(args, "smoke_example", False)),
        "nlp_profile": SMOKE_PROFILE if getattr(args, "smoke_example", False) else NLP_PROFILE,
        "model_sha256": sha256(model_path) if model_path else None,
        "full_benchmark_reproduction": False,
    }


def train(args):
    caes, _, baos, _ = engines()
    new_output(args.output, [args.fpath])
    meta_path = model_metadata_path(args.output)
    new_output(meta_path, [args.fpath])
    rows = clean_rows(load_rows(args.fpath, smoke=args.smoke_example, gold=True), baos)
    nlp = load_nlp(args.smoke_example)
    with tempfile.TemporaryDirectory(prefix="cebra-train-") as directory:
        clean = Path(directory) / "train.jsonl"
        write_rows(clean, rows)
        opts = argparse.Namespace(fpath=str(clean), output=str(args.output), summary_output=None,
                                  train_offset=0, train_limit=len(rows), max_iter=1000, seed=42, drop_features="")
        with nlp_override(caes, nlp):
            caes.train_model(opts)
    metadata = provenance(args, args.output)
    metadata.update(feature_names=caes.FEATURE_NAMES, scikit_learn=importlib.metadata.version("scikit-learn"),
                    training_samples=len(rows), seed=42, C=1, max_iter=1000,
                    training_ids=[str(row["id"]) for row in rows if "id" in row])
    write_json(meta_path, metadata)


def check_calibration_manifest(args, rows, metadata):
    manifest = read_json(args.split_manifest)
    if manifest.get("split") != "calibration" or manifest.get("sha256") != sha256(args.fpath):
        raise ValueError("A calibration-only manifest matching this input SHA-256 is required.")
    ids = [str(row.get("id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)) or manifest.get("question_ids") != ids:
        raise ValueError("Calibration question IDs must be unique and exactly match the ordered manifest.")
    if manifest.get("samples") != len(rows):
        raise ValueError("Calibration sample count mismatch.")
    if set(ids).intersection(metadata.get("training_ids", [])):
        raise ValueError("Calibration overlaps this scorer's recorded training IDs.")
    if sha256(args.fpath) == metadata.get("input_sha256"):
        raise ValueError("Training input cannot also be used for calibration.")
    return manifest


def calibrate(args):
    caes, controls, baos, _ = engines()
    model_path, metadata, model, names, config = resolve_model(args, caes)
    new_output(args.output, [args.fpath, model_path, args.split_manifest])
    rows = load_rows(args.fpath, smoke=args.smoke_example, gold=True)
    manifest = check_calibration_manifest(args, rows, metadata)
    rows = clean_rows(rows, baos)
    nlp = load_nlp(args.smoke_example)
    budget_rows = [controls.budget_only_replay(deepcopy(row), args.budget) for row in rows]
    with tempfile.TemporaryDirectory(prefix="cebra-calibration-") as directory:
        clean = Path(directory) / "calibration.jsonl"
        budget = Path(directory) / "budget.jsonl"
        write_rows(clean, rows)
        write_rows(budget, budget_rows)
        curve = baos.sweep_for_budget_match(
            clean, budget, nlp=nlp, model=model, feature_names=names,
            thresholds=baos.parse_csv_floats(baos.DEFAULT_THRESHOLDS), min_keep_values=[0, 1],
            max_recall_drop=0.01, sweep_limit=len(rows))
    lock = {"schema_version": 1, **provenance(args, model_path), **curve,
            "dataset": manifest.get("dataset"), "split_manifest_sha256": sha256(args.split_manifest),
            "calibration_ids": manifest["question_ids"], "budget_reference": args.budget,
            "max_recall_drop": 0.01, "K_ref": curve["budget_only_metrics"]["final_context_k"],
            "test_data_used_for_selection": False, "feature_names": names}
    write_json(args.output, lock)
    print(json.dumps({"selected": curve["selected"], "K_ref": lock["K_ref"]}, indent=2))


def retain(args):
    caes, _, baos, _ = engines()
    model_path, metadata, model, names, config = resolve_model(args, caes)
    inputs = [args.fpath, model_path] + ([args.lock] if args.lock else [])
    new_output(args.output, inputs)
    sidecar = Path(str(args.output) + ".meta.json")
    new_output(sidecar, inputs)
    rows = load_rows(args.fpath, smoke=args.smoke_example)
    if args.lock:
        lock = read_json(args.lock)
        if lock.get("schema_version") != 1 or lock.get("model_sha256") != sha256(model_path):
            raise ValueError("Calibration lock/model mismatch.")
        if lock.get("feature_names") != names:
            raise ValueError("Calibration lock feature order mismatch.")
        if lock.get("nlp_profile") != (SMOKE_PROFILE if args.smoke_example else NLP_PROFILE):
            raise ValueError("Calibration lock NLP profile mismatch.")
        if set(str(r.get("id", "")) for r in rows).intersection(lock.get("calibration_ids", [])):
            raise ValueError("Retention input overlaps calibration IDs; use disjoint evaluation input.")
        point = lock["selected"]
    elif config:
        point = config["operating_point"]
    else:
        raise ValueError("Custom models require --lock; frozen --config supplies an existing operating point.")
    threshold, minimum = point["threshold"], point["min_keep_per_iter"]
    if not 0 <= threshold <= 1 or minimum not in (0, 1):
        raise ValueError("Invalid operating point.")
    nlp = load_nlp(args.smoke_example)
    rows = clean_rows(rows, baos)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cebra-retain-") as directory:
        clean = Path(directory) / "input.jsonl"
        write_rows(clean, rows)
        baos.replay_to_file_streaming(clean, nlp=nlp, model=model, feature_names=names,
                                     threshold=threshold, min_keep_per_iter=minimum,
                                     output_path=args.output, skip_existing=False)
    write_json(sidecar, {**provenance(args, model_path), "operating_point": point,
                        "output_sha256": sha256(args.output), "samples": len(rows),
                        "calibration_lock_sha256": sha256(args.lock) if args.lock else None})
    print(f"Wrote {len(rows)} questions to {args.output}")


def evaluate(args):
    _, _, _, metrics = engines()
    new_output(args.output, [args.fpath])
    rows = load_rows(args.fpath)
    if any(not isinstance(row.get("oracle_ids", row.get("oracle")), list) for row in rows):
        raise ValueError("Recall evaluation requires oracle or oracle_ids; retention itself does not.")
    # Delegate the complete report, including ID splitting and schema coverage, unchanged.
    previous = sys.argv
    try:
        sys.argv = ["cebra-evaluate", "--fpath", str(args.fpath), "--output", str(args.output)]
        metrics.main()
    finally:
        sys.argv = previous


def parser():
    p = argparse.ArgumentParser(description="CEBRA: state-aware chunk reduction on fixed retrieval trajectories.")
    sub = p.add_subparsers(dest="command", required=True)
    for name, function, help_text in [
        ("train", train, "Train the nine-feature CAES scorer on training trajectories."),
        ("calibrate", calibrate, "Select a BAOS point using calibration-only data."),
        ("retain", retain, "Apply a frozen scorer and operating point."),
        ("evaluate", evaluate, "Evaluate retained supporting-evidence recall and context size."),
    ]:
        cmd = sub.add_parser(name, help=help_text, description=help_text)
        cmd.add_argument("--fpath", type=Path, required=True, help="Input trajectory JSONL.")
        cmd.add_argument("--output", type=Path, required=True, help="New output file; existing files are never overwritten.")
        cmd.set_defaults(func=function)
        if name != "evaluate":
            cmd.add_argument("--smoke-example", action="store_true", help="Explicit regex features for marked synthetic smoke rows only; NOT paper reproduction.")
        if name in {"calibrate", "retain"}:
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument("--config", type=Path, help="Frozen dataset configuration (model paths relative to config).")
            group.add_argument("--model", type=Path, help="Trusted scorer with a matching .meta.json sidecar.")
        if name == "calibrate":
            cmd.add_argument("--split-manifest", type=Path, required=True, help="Calibration split, input SHA-256, sample count, and ordered question IDs.")
            cmd.add_argument("--budget", type=int, choices=range(1, 101), default=3, help="Budget-only reference cap; paper default 3 (not a CEBRA per-question cap).")
        if name == "retain":
            cmd.add_argument("--lock", type=Path, help="A calibration lock; otherwise use the frozen config point.")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError, ImportError) as error:
        p.exit(2, f"CEBRA error: {error}\n")


if __name__ == "__main__":
    main()
