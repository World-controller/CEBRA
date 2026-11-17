# Reproducibility guide

## What this release provides

This release provides executable CAES/BAOS code, two frozen scorers, three operating-point configurations, and a synthetic end-to-end CPU check. It does not distribute complete historical retrieval trajectories, corpus embeddings, large reader weights, or all table-level experiment outputs. A runnable smoke check is not equivalent to reproducing the paper's results.

Use Python 3.11 and `requirements-nlp.txt` for the tested CPU release profile: NumPy 1.26.4, scikit-learn 1.9.0, spaCy 3.7.5 and `en_core_web_sm` 3.7.0. This is the explicit release runtime profile, not a claim that every historical experiment ran in this environment. Exact historical comparisons also require the original trajectory, NLP preprocessing, split and scorer bindings.

Normal runs reject a missing/different spaCy pipeline instead of silently substituting regex features. The smoke flag is accepted only when **every** input row is explicitly marked `"_cebra_example": "synthetic-smoke"`. Do not mark real benchmark data this way.

## Frozen inference

```bash
python src/cebra.py retain \
  --config configs/hotpotqa.json \
  --fpath data/hotpotqa/test.jsonl \
  --output outputs/hotpotqa-retained.jsonl

python src/cebra.py evaluate \
  --fpath outputs/hotpotqa-retained.jsonl \
  --output outputs/hotpotqa-retention-metrics.json
```

Use `configs/2wiki.json` or `configs/musique.json` for their respective frozen settings. These commands need real input files prepared separately; the paths above describe user-supplied files, not bundled downloads. Model paths inside a config resolve relative to that config.

The wrapper preserves original round order, labels, query-update flags and serialized retention fields. It removes stale replay annotations from a fresh copy before scoring. It does not use supporting-document IDs to select evidence at inference. The final reader context is deduplicated using the existing extraction logic; raw retained candidates and unique context size are different quantities.

## Training a new scorer (optional)

```bash
python src/cebra.py train \
  --fpath data/custom/train.jsonl \
  --output outputs/custom-caes.pkl
```

This calls the existing nine-feature implementation with Z-score standardization, class-balanced logistic regression, `C=1`, seed 42 and `max_iter=1000`. The command uses all supplied training rows and writes a `.meta.json` sidecar. Train, calibration and test files must be separate. **The bundled formal scorers must not be overwritten or retrained for frozen-result reproduction.**

Only load trusted pickle artifacts: deserializing a pickle can execute code. Hash matching detects changes but does not establish trust in a third-party model.

## Calibration and locking

Prepare a manifest for the calibration-only JSONL:

```json
{
  "dataset": "custom",
  "split": "calibration",
  "samples": 2,
  "sha256": "<SHA-256 of calibration.jsonl>",
  "question_ids": ["calibration-question-1", "calibration-question-2"]
}
```

The ordered IDs and count must match the input. Obtain the checksum with `shasum -a 256 data/custom/calibration.jsonl`. Keep test examples in another file: the calibration command does not accept a test file or use a combined calibration/test prefix. The manifest is a declaration of the split, not proof of independent dataset construction. Training-ID overlap is rejected when recorded by the custom model; historical transfer scorers do not embed their complete training-ID lists.

```bash
python src/cebra.py calibrate \
  --model outputs/custom-caes.pkl \
  --fpath data/custom/calibration.jsonl \
  --split-manifest data/custom/calibration.manifest.json \
  --output outputs/custom-baos.json

python src/cebra.py retain \
  --model outputs/custom-caes.pkl \
  --lock outputs/custom-baos.json \
  --fpath data/custom/test.jsonl \
  --output outputs/custom-retained.jsonl
```

Calibration calls the existing `run_four_configs.sweep_for_budget_match` / `choose_candidate` functions, **not** the early generic `cmrg_calibrated sweep` command. It uses thresholds 0.1 through 0.9, minimum-per-round values 0 and 1, recall tolerance 0.01, and the budget-only reference cap 3 by default. `K_ref` is the measured mean unique context size of that reference, not a hard CEBRA cap.

Selection retains the original priority: consider `min_keep=0` first, using recall and empty-context gates; if necessary, use the recorded relaxed-empty path, then the full candidate pool and existing fallback. Candidates are ordered by distance to `K_ref`, higher recall, fewer empty contexts, lower threshold and lower minimum. The result preserves `relaxed_empty_gate` and all candidate metrics. A selected fallback is **not** silently relabeled as satisfying every gate. Check the curve before using it in a new study. Model hash, NLP profile, input/manifest hashes and calibration IDs are stored in the lock; retention checks them and rejects overlap with those calibration IDs.

## Testing all four commands without benchmark data

Append `--smoke-example` to `train`, `calibrate`, and `retain`, and use the three disjoint files in `examples/smoke/`:

```bash
python src/cebra.py train --fpath examples/smoke/train.jsonl --output outputs/example-model.pkl --smoke-example
python src/cebra.py calibrate --model outputs/example-model.pkl --fpath examples/smoke/calibration.jsonl --split-manifest examples/smoke/calibration.manifest.json --output outputs/example-lock.json --smoke-example
python src/cebra.py retain --model outputs/example-model.pkl --lock outputs/example-lock.json --fpath examples/smoke/test.jsonl --output outputs/example-retained.jsonl --smoke-example
python src/cebra.py evaluate --fpath outputs/example-retained.jsonl --output outputs/example-metrics.json
```

This trains a disposable example scorer, never a replacement for a paper scorer. Synthetic scores have no scientific interpretation. Outputs are created at new paths only; input files and existing artifacts are not overwritten.

## Evaluation coverage

`evaluate` delegates the existing retention metric report unchanged: supporting-evidence recall (fraction), mean unique chunks (`final_context_k`), raw kept candidates, duplicate ratio, empty-context counts, recall per chunk and schema coverage. It supports composite document IDs separated by `//`. It does not generate answers, judge correctness, compute EM/F1, measure latency or reproduce every paper-table column. See the advanced guide for separate stages.
