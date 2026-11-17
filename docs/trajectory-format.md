# Trajectory format

A file contains one UTF-8 JSON object per question. The public wrapper supports the existing maximum of four contiguous rounds, numbered `"0"` through `"3"`; unsupported round counts are rejected rather than truncated.

```json
{
  "id": "question-1",
  "query": "Which county contains Cedar Town?",
  "oracle": ["document-1"],
  "0": {
    "query": "Cedar Town county",
    "docs": [
      {
        "id": "document-1",
        "title": "Cedar Town",
        "text": "Cedar Town is located in Amber County.",
        "label": "<CONTINUE>",
        "info": "Amber County"
      }
    ]
  }
}
```

This snippet is illustrative fictional text, not a benchmark row. Each document needs a nonempty ID and an upstream label: `<CONTINUE>`, `<FINISH>` or `<TERMINATE>`. Empty document lists are valid. An empty file is not a dataset.

- `id`: question identifier, required and unique for calibration.
- `query` (or `question`): original question; each round can also carry its current query.
- `oracle`: supporting-document ID list, required for training/calibration, not used during inference. `evaluate` also accepts the legacy `oracle_ids` field.
- `docs`: candidates in their original retrieval order. Never relabel or reorder them for a frozen experiment.
- `info`: query-update information from the upstream labeler, or an empty string.

Retention adds the existing fields `keep_for_qa`, `use_for_next_query` and `chain_marginal_score`. No model-facing schema rename is introduced. Other source fields are preserved; stale retention annotations are cleared before applying the current policy. Decisions use the round-entry evidence state; retained terms/IDs are updated after that round's scoring.

Supporting annotations should not be sent to a reader. Build the final context with the existing deduplicating extraction function and pass only the original question plus retained title/text. Raw candidates can contain the same document in multiple rounds; their count is not the unique reader-context count.

Every retention output has a `.meta.json` sidecar with input/output and model hashes, operating point, NLP profile and the smoke-example flag. Do not discard it when moving results.
