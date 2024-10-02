# MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference

In this work, we follow the framework of LongBench to build MPCache.

## Dataset Preparation
You can download and load the LongBench dataset through the Huggingface datasets ([HF Repo](https://huggingface.co/datasets/THUDM/LongBench)):
```python
from datasets import load_dataset

datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in datasets:
    data = load_dataset('THUDM/LongBench', dataset, split='test')
```
You can also download the datasets from the website [this link](https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip).


## Evaluation

**Packages and environment:**

Install the required packages with pip tool: 
```bash
pip install -r requirements.txt
```
For LLaMA-2-based model inference on long sequences, we follow LongBench and use FlashAttention during the prefill stage for saving GPU memory.
The relevant dependencies can be installed according to the codebase of [Flash Attention](https://github.com/Dao-AILab/flash-attention).



First, run pred_mine.py to perform the model inference on longchat-v1.5-7b-32k:
```bash
CUDA_VISIBLE_DEVICES=0 python pred_mine.py --model longchat-v1.5-7b-32k
```
You can also run inference on multi-gpus in parallel (one model per gpu):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python pred_mine.py --model longchat-v1.5-7b-32k
```
Then, we can obtain the output of the model under all LongBench datasets under the `pred_mine/` folder corresponding to the model name:
After inference, we can run eval_mine.py to evaluate the model performance:
```bash
python eval_mine.py --model longchat-v1.5-7b-32k
```
We can get the results on the datasets in `result.json`.
