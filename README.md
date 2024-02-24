- official repository for the paper: "ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval"
- The code will be updated soon!


### Model checkpoints (huggingface)
1. [RankT5-base](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-base`
2. [RankT5-large](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-large`
3. [RankT5-3b](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-3b`
4. [ListT5-base](https://huggingface.co/Soyoung97/ListT5-base): `Soyoung97/ListT5-base`
5. [ListT5-3b](https://huggingface.co/Soyoung97/ListT5-3b): `Soyoung97/ListT5-3b`


### How to use
```
from FiDT5 import FiDT5
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

