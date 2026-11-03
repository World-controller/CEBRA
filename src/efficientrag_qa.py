import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from tqdm.rich import tqdm_rich

try:
    from baseline.retrieve.direct import (
        DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
        DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
        DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
    )
except ModuleNotFoundError:
    _FALLBACK_PROMPT = (
        "Answer the question using the provided knowledge. "
        "Return JSON with an 'answer' field.\n"
        "Question: {question}\nKnowledge: {knowledge}"
    )
    DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA = _FALLBACK_PROMPT
    DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE = _FALLBACK_PROMPT
    DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA = _FALLBACK_PROMPT
from conf import RETRIEVE_RESULT_PATH

PROMPT_MAPPING = {
    "hotpotQA": DIRECT_RETRIEVE_ANSWER_PROMPT_HOTPOTQA,
    "2WikiMQA": DIRECT_RETRIEVE_ANSWER_PROMPT_WIKIMQA,
    "musique": DIRECT_RETRIEVE_ANSWER_PROMPT_MUSIQUE,
}


class EfficientRAG_QA:
    def __init__(
        self,
        model: Any,
        data: List[Dict],
        dataset: str,
        num_workers: int = 10,
    ) -> None:
        self.model = model
        self.data = data
        self.num_workers = num_workers
        self.prompt_template = PROMPT_MAPPING[dataset]

    def parse_samples_in_parallel(self) -> List[Dict]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            tasks = {
                executor.submit(self.parse_sample, sample): idx
                for idx, sample in enumerate(self.data)
            }
            results = []
            for future in tqdm_rich(as_completed(tasks), total=len(self.data)):
                idx = tasks[future]
                try:
                    res = future.result()
                    results.append((idx, res))
                except Exception as e:
                    import traceback

                    print(f"Error processing sample {idx}: {e}")
                    traceback.print_exc()
        results = [r[1] for r in sorted(results, key=lambda x: x[0])]
        return results

    def extract_chunks(self, sample: Dict):
        chunks = []
        seen_chunks = set()
        has_cmrg_keep = any(
            "keep_for_qa" in chunk
            for iter in range(4)
            if f"{iter}" in sample
            for chunk in sample[f"{iter}"]["docs"]
        )
        for iter in range(4):
            if f"{iter}" not in sample:
                break
            for chunk in sample[f"{iter}"]["docs"]:
                if has_cmrg_keep and "keep_for_qa" in chunk:
                    should_use = bool(chunk["keep_for_qa"])
                else:
                    should_use = chunk["label"] != "<TERMINATE>"
                if should_use:
                    # 去重重复检索块，避免长 prompt 超过本地 vLLM 上下文。
                    chunk_key = chunk.get("id") or chunk.get("text")
                    if chunk_key in seen_chunks:
                        continue
                    seen_chunks.add(chunk_key)
                    chunks.append(chunk)
        return chunks

    def parse_sample(self, sample: Dict) -> Dict:
        from utils import ask_model

        knowledge_list = self.extract_chunks(sample)
        chunks = "\n".join([chunk["text"] for chunk in knowledge_list])
        question = sample["query"]
        prompt = self.prompt_template.format(question=question, knowledge=chunks)
        result = ask_model(
            self.model,
            prompt,
            type="json",
            check_if_valid=lambda x: isinstance(x, dict) and "answer" in x,
            mode="chat",
        )
        if not isinstance(result, dict):
            print(f"Empty model response for question: {question}")
            result = {"answer": ""}
        chunk_ids = [p["id"] for p in knowledge_list]
        response = {
            # "question_id": sample["id"],
            "question": question,
            "answer": sample["answer"],
            "model_output": result["answer"],
            "oracle_ids": sample["oracle"],
            "chunk_ids": chunk_ids,
        }
        return response


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpath", type=str, required=True)
    parser.add_argument("--model", type=str, default="llama")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    return args


def main(opts: argparse.Namespace):
    from language_models import get_model
    from utils import load_jsonl, write_jsonl

    model = get_model(opts.model)
    dataset = load_jsonl(opts.fpath)
    if opts.offset or opts.limit is not None:
        # 用于 QA sanity：先切小片段验证 generator JSON 输出，再跑全量。
        dataset = dataset[opts.offset : None if opts.limit is None else opts.offset + opts.limit]
    efficient_rag_qa = EfficientRAG_QA(
        model, dataset, opts.dataset, num_workers=opts.workers
    )
    results = efficient_rag_qa.parse_samples_in_parallel()
    save_path = os.path.join(
        RETRIEVE_RESULT_PATH,
        "efficient_rag",
        f"{opts.dataset}-{opts.suffix}_qa_results.jsonl",
    )
    write_jsonl(results, save_path)


if __name__ == "__main__":
    options = parse_args()
    main(options)
