import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from efficient_rag.cmrg import CMRGConfig, public_cmrg_fields, score_candidates
def load_jsonl(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def replay_sample(sample: dict, nlp, config: CMRGConfig) -> dict:
    kept_context = []
    question = sample.get("query") or sample.get("question") or ""
    for iter_idx in range(4):
        iter_key = str(iter_idx)
        if iter_key not in sample:
            break
        docs = sample[iter_key].get("docs", [])
        labels = [doc.get("label", "") for doc in docs]
        infos = [doc.get("info", "") for doc in docs]
        outputs = score_candidates(
            question=question,
            query=sample[iter_key].get("query", question),
            chunks=docs,
            labels=labels,
            infos=infos,
            kept_context=kept_context,
            nlp=nlp,
            config=config,
        )
        for doc, output in zip(docs, outputs):
            doc.update(
                public_cmrg_fields(output, debug_features=config.debug_features)
            )
            if output["keep_for_qa"]:
                kept_context.append(
                    {
                        "id": doc.get("id"),
                        "title": doc.get("title", ""),
                        "text": doc.get("text", ""),
                        "_cmrg_terms": output["_cmrg_terms"],
                    }
                )
    return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--keep_threshold", type=float, default=0.35)
    parser.add_argument("--min_keep_per_iter", type=int, default=1)
    parser.add_argument("--debug_features", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    data = load_jsonl(args.fpath)
    if args.limit is not None:
        data = data[: args.limit]
    nlp = load_nlp()
    config = CMRGConfig(
        keep_threshold=args.keep_threshold,
        min_keep_per_iter=args.min_keep_per_iter,
        control_next_query=False,
        debug_features=args.debug_features,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in data:
            f.write(json.dumps(replay_sample(sample, nlp, config), ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
