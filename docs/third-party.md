# Third-party acknowledgments

CEBRA's original contribution is state-aware evidence scoring and budget-aware retention. The repository also retains upstream retrieval infrastructure and baseline implementations.

## Retrieval infrastructure

The initial retrieval infrastructure is derived from [NIL-zhuang/EfficientRAG-official](https://github.com/NIL-zhuang/EfficientRAG-official). It is used as a fixed retrieval substrate and as a baseline in the experiments, not presented as newly authored CEBRA code. Historical `efficient_rag`, `efficientrag_*`, and `cmrg_*` identifiers remain for compatibility.

The inherited MIT license and Microsoft copyright notice are preserved in the root [LICENSE](../LICENSE). Repository support and contribution policies are maintained by this project; Microsoft is not its support provider.

For work relying on this retrieval infrastructure, cite:

```bibtex
@inproceedings{zhuang2024efficientrag,
  title = {{EfficientRAG}: Efficient Retriever for Multi-Hop Question Answering},
  author = {Zhuang, Ziyuan and Zhang, Zhiyang and Cheng, Sitao and Yang, Fangkai and Liu, Jia and Huang, Shujian and Lin, Qingwei and Rajmohan, Saravan and Zhang, Dongmei and Zhang, Qi},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  pages = {3392--3411},
  year = {2024},
  doi = {10.18653/v1/2024.emnlp-main.199}
}
```

## Models and datasets

Meta Llama 3 and Llama 3.3 are external reader models. Their community licenses and acceptable-use policies apply independently of the MIT code license. See [NOTICE](../NOTICE), the [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), and the [Llama 3.3 model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct). Reader weights are not bundled here. The existing “Built with Meta Llama 3” notice is retained.

Contriever, DeBERTa, spaCy language pipelines and benchmark datasets have their own applicable licenses. Install or download them from their official sources. Existing baseline module docstrings describe implementation provenance and limitations; they are not replaced with blanket CEBRA authorship claims.
