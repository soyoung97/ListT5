Official code repository for the paper: [ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval](https://arxiv.org/abs/2402.15838) (ACL 2024, main)
- The code will be updated soon!
- Please email to `soyoung.yoon@snu.ac.kr` for any questions about this work!
- 2024.3.20 Added minimal running examples!
- 2024.5.16 Accepted to ACL 2024 main! :)
- 2024.6.15 Started on the code release... Expected to update the full code within a week! Stay tuned:)

### Model checkpoints (huggingface)
1. [RankT5-base](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-base`
2. [RankT5-large](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-large`
3. [RankT5-3b](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-3b`
4. [ListT5-base](https://huggingface.co/Soyoung97/ListT5-base): `Soyoung97/ListT5-base`
5. [ListT5-3b](https://huggingface.co/Soyoung97/ListT5-3b): `Soyoung97/ListT5-3b`

### Evaluation Datasets (BEIR)

**Please note that the license follows the original BEIR/MSMARCO license (for academic purposes only) and we are not responsible for any copyright issues.**
1. [BEIR top-100 by BM25](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top100/tree/main) `Soyoung97/beir-eval-bm25-top100`
2. [BEIR top-1000 by BM25](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top1000/tree/main) `Soyoung97/beir-eval-bm25-top1000`
3. [BEIR top-100 by COCO-DR Large](https://huggingface.co/datasets/Soyoung97/beir-eval-cocodr-large-top100/tree/main) `Soyoung97/beir-eval-cocodr-large-top100`

Tip: click on the indiviual file link, copy the `download` link, and use wget to download each file on your server.
example:
```
wget https://huggingface.co/datasets/Soyoung97/beir-eval-cocodr-large-top100/resolve/main/nfcorpus.jsonl
```
- Also planning to release the full training data used! [TBU]

### Running environments
```
conda env create -f listt5_conda_env.yml
```
Note: torch=2.1.0 and transformers=4.33.3 (Other versions can be incompatible)
a version mismatch can result in errors such as:
```
AttributeError: 'EncoderWrapper' object has no attribute 'embed_tokens'
```
### How to use

#### Minimal Running example
Minimal running example can be run by the following example code.

```
from FiDT5 import FiDT5 # Need to clone the repository for this code
from transformers import T5Tokenizer
model = FiDT5.from_pretrained('Soyoung97/ListT5-base')
texts = [
"Query: When did Thomas Edison invent the light bulb?, Index: 1, Context: Lightning strike at Seoul National University", 
"Query: When did Thomas Edison invent the light bulb?, Index: 2, Context: Thomas Edison tried to invent a device for car but failed",
"Query: When did Thomas Edison invent the light bulb?, Index: 3, Context: Coffee is good for diet",
"Query: When did Thomas Edison invent the light bulb?, Index: 4, Context: KEPCO fixes light problems",
"Query: When did Thomas Edison invent the light bulb?, Index: 5, Context: Thomas Edison invented the light bulb in 1879"]
tok = T5Tokenizer.from_pretrained('t5-base')
raw = tok(texts, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0), 'attention_mask': raw['attention_mask'].unsqueeze(0)}
output = model.generate(**input_tensors, max_length=7, return_dict_in_generate=True, output_scores=True)
output_text = tok.batch_decode(output.sequences, skip_special_tokens=True) 
output_text
>>> [3 1 4 2 5]
```
#### Run tournament sort with ListT5
```
python3 run_listt5.py --input_path ./trec-covid.jsonl --output_path ./outputs/listt5-trec-covid.jsonl --bsize 20 
```
Sample code outputs ndcg@10: 0.78285 for ListT5-base with out_k=2.

#### Run baseline models (MonoT5, RankT5)

```
python3 run_monot5_rankt5.py --input_path ./trec-covid.jsonl --output_path ./outputs/monot5-trec-covid.jsonl --bsize 20 --mode monot5
```
ndcg@10: 78.26 for monot5 

```
python3 run_monot5_rankt5.py --input_path ./trec-covid.jsonl --output_path ./outputs/rankt5-trec-covid.jsonl --bsize 20 --model Soyoung97/rankt5-base --mode rankt5
```
ndcg@10: 77.731 for rankt5

Please download the above evaluation datasets to try out evaluation for other datasets.
You can easily run evaluation with your own dataset by adhering with the data format just like the beir dataset.
