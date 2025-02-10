Official code repository for the paper: [ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval](https://arxiv.org/abs/2402.15838) (ACL 2024, main)
- Please email to `soyoung.yoon@snu.ac.kr` for any questions about this work!
- 2024.3.20 Added minimal running examples!
- 2024.5.16 Accepted to ACL 2024 main! :)
- 2024.6.15 Started on the code release...
- 2024.7.25 Uploaded instructions to download training data. Expected to update the training code before the conference starts! Stay tuned:)
- 2024.10.15 I'm preparing to organize and open-source the training code, but until then, please refer to the comments [here](https://github.com/soyoung97/ListT5/issues/2#issuecomment-2413030507) to get some information about training ListT5.
- 2024.12.27 I have finally uploaded the training code for ListT5 (and the positional bias experiments)! Thank you for waiting, and feel free to ask any questions! :)
- 2024.01.05 I have tested if the training code is replicable and left the replication log at README!

### Model checkpoints (huggingface)
1. [RankT5-base](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-base`
2. [RankT5-large](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-large`
3. [RankT5-3b](https://huggingface.co/Soyoung97/RankT5-base): `Soyoung97/RankT5-3b`
4. [ListT5-base](https://huggingface.co/Soyoung97/ListT5-base): `Soyoung97/ListT5-base`
5. [ListT5-3b](https://huggingface.co/Soyoung97/ListT5-3b): `Soyoung97/ListT5-3b`

### Evaluation Datasets (BEIR)

**Please note that the license follows the original BEIR/MSMARCO license (for academic purposes only) and we are not responsible for any copyright issues.**
1. [BEIR top-100 by BM25](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top100) `Soyoung97/beir-eval-bm25-top100`
2. [BEIR top-1000 by BM25](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top1000) `Soyoung97/beir-eval-bm25-top1000`
3. [BEIR top-100 by COCO-DR Large](https://huggingface.co/datasets/Soyoung97/beir-eval-cocodr-large-top100) `Soyoung97/beir-eval-cocodr-large-top100`

Tip: click on the indiviual file link, copy the `download` link, and use wget to download each file on your server.
example:
```
wget https://huggingface.co/datasets/Soyoung97/beir-eval-cocodr-large-top100/resolve/main/nfcorpus.jsonl
```

- The training data used are processed at: https://huggingface.co/datasets/Soyoung97/ListT5-train-data/ and if you wish to get access, please contact soyoung.yoon@snu.ac.kr with your huggingface id!
- The training data format looks like the following:
<img width="1656" alt="train_data_sample" src="https://github.com/user-attachments/assets/174904f2-be98-4b24-a17b-4f479576af35">

### Running environments
```
conda env create -f listt5_conda_env.yml
```
Note: torch=2.1.0 and transformers=4.33.3 (Other versions can be incompatible)
a version mismatch can result in errors such as:
```
AttributeError: 'EncoderWrapper' object has no attribute 'embed_tokens'
```
For training ListT5, We checked the process runs correctly on python=3.10.13.
Other versions may work as well. We also annotated some of the working package versions on the `requirements.txt` file.
Please note that the file is manually annotated, so you may need to install other packages as well.


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

### Running the training code

First, navigate to the `./train_code/` directory. The following are the codes we used to train ListT5-base and ListT5-3B:
The checkpoints and log files will be saved under the `/outputs/` directory.

T5-base: (the ListT5-base model is the one saved with tfmr\_20000)
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --name YOUR_EXP_NAME --do_train --learning_rate 1e-04 \
--base_model t5-base --train_batch_size 32 --eval_batch_size 32 --num_workers 0 --max_input_length 230 \
--max_output_length 8 --train-files /PATH/To/TRAIN/FILE/marco-train-coco-top1000-5-20perq.jsonl \
--eval-files /PATH/TO/VALIDATION/FILE/marco-dev-coco-top1000-5-500.jsonl --lr_scheduler linear \
--gradient_accumulation_steps 2 --eval_steps 2000 --num_train_epochs 10 --listwise_k 5
```

*Note that validation file is only used to check the loss, not the evaluation result (e.g., ndcg) obtained using torunament sort.
*Note that the reported ListT5-base model is only trained with **20000 steps** and then did **early exit**. While we give the learning rate scheduler and warmup steps to 10 epochs, this equals to stopping the training after only 0~1 epochs. Running this on 4x A6000 gpus will only take about 8-10 hours. Similarly, the 3B model is only trained with 3000k steps.

T5-3B: (the ListT5-3B model is the one saved with tfmr\_3000)
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 train.py --name YOUR_EXP_NAME --do_train --learning_rate 1e-05 --base_model t5-3b --train_batch_size 2 --eval_batch_size 2 --num_workers 0 --max_input_length 128 --max_output_length 7 --train-files /PATH/To/TRAIN/FILE/marco-train-coco-top1000-5-20perq.jsonl --eval-files /PATH/TO/VALIDATION/FILE/marco-dev-coco-top1000-5-500.jsonl --lr_scheduler linear --run_3b --gradient_accumulation_steps 16 --eval_steps 4000 --num_train_epochs 5 --listwise_k 5
```

## Other tips
* add `--wandb` option if you want to see the loss graph and log experiments.
* may take 5-10 mins at first to start (deepspeed setup)
* if you want to train in order reversed - relevnant index first way, add `--sub_mode posfirst`
* resume training: `--resume_from_checkpoint PATH_TO_DEEPSPEED_CKPT_SUCH_AS/epoch\=00-stepglobal_step\=23999.0.ckpt/` Must run with the same number of gpus on initial training.
* Use ForkedPdb.set_trace() on debugging for multi-gpu training.
* Running training on T5-base takes 33GB each for 4x A6000 48GB.

Once the training starts, the process shall print something like the following:
<details>
  <summary>Example output screen</summary>

  ```text
    88/92544 [01:19<23:04:41,  1.11it/s, v_num=1]
================================================================================
(1474878) train
input: ['Query: temperature to seal driveway, Index: 1, Context: Pricing depends on the extent of the repair and size of the driveway, but starts around $500. D.I.Y. MATERIAL COSTS* Crack filler material costs $20-$40 on average, and comes in 1-gallon buckets.* Sealer typically ... 8 percent lower than for concrete moist cured for the entire period.he curing period for cold weather concrete is longer than the standard period due to reduced rate of strength gain. Compressive strength of concrete cured and maintained at 50 degrees Fahrenheit is expected to gain strength half as quickly as concrete cured at 73 degrees Fahrenheit.']

output: 3
================================================================================
================================================================================
(11323048) train
input: ['Query: when did yorktown begin, Index: 1, Context: The American Revolution began in 1775 as open conflict between the united North American thirteen colonies and Great Britain. By the Treaty of Paris that ended the war in 1783, the colonies had won their independence. While no one event can be pointed to as the actual cause of the revolution, the war began as a disagreement over the way in which Great Britain treated the colonies versus the way the colonies felt they should be treated.', 'Query: when did yorktown begin, Index: 5, Context: Prior to the battle General Cornwallis occupied the town of Yorktown in order to establish a defensible deep-water port. Although this final major battle took place in 1781 the American Revolutionary War was not officially over until the signing of the Treaty of Paris in September of 1783.']

output: 1
================================================================================
================================================================================
(6697679) train
input: ['Query: what kind of soil do avocados need, Index: 1, Context: Place the glass in a bright windowsill. In about three to six weeks the top of the avocado pit will begin to split and a stem sprout will emerge from the top and roots will begin to grow at the base. When the stem grows to about five or six inches, pinch out the top set of leaves.', 'Query: what kind of soil do avocados need, Index: 5, Context: If you choose to grow the avocado tree in a container, the soil should be a specialized planting mixture designed for avocados along with some cultivated soil from your garden. The miniature soil environment can trap moisture that encourages root rot, so coupling the tree with thirsty flowers below is a smart way to equalize the soil and water ratio.']

output: 3
================================================================================
================================================================================
(10360998) train
input: ["Query: how did the failure of the schlieffen plan lead to a stalemate, Index: 1, Context: One of these reasons was that the assassination was an example of the Hapsburg's loss of control and if Austria were to decline to Germany's offer, Germany would be completely surrounded by enemies. 1 The German government also knew that Russia would lose a major base in Europe if they were to lose Yugoslavia.t this point, Russia had pretty much surrendered to the Germans. At that point, Germany also made an alliance with Finland and deposited 150,000 soldiers in their country-soldiers that could have been used in the actual war.", 'Query: how did the failure of the schlieffen plan lead to a stalemate, Index: 5, Context: However, there were also failures. The League sometimes failed to enforce the Treaty of Versailles (e.g., the Poles captured Vilna in 1920, and Lithuania seized Memel in 1923). The League could not stop powerful nations (e.g., in 1923, when France invaded the Ruhr, and Italy occupied Corfu).he League of Nations aimed to stop wars, improve people s lives and jobs, encourage disarmament and enforce the Treaty of Versailles. Judged against these aims, the League was quite successful in the 1920s. It stopped border disputes turning into wars.']

output: 5
================================================================================
Epoch 0:   0%|▏                                                                                                                                              | 94/92544 [01:24<23:01:20,  1.12it/s, v_num=
```
</details>

### Model Training Replication Log

With the command above, I re-ran the training code on January 5, 2025, tested on a subset of BEIR, and confirmed that the results are replicable. ([training data](https://huggingface.co/datasets/Soyoung97/ListT5-train-data), [evaluation data](https://huggingface.co/datasets/Soyoung97/beir-eval-bm25-top100))  The slight difference in NDCG@10 performance may be due to hardware differences, as the initial ListT5 was trained on an NVIDIA A100, while the recent experiments ran on A6000. The re-run results were similar to, or slightly better than—the initial model. I have uploaded the [new model](https://huggingface.co/Soyoung97/ListT5-base-A6000) for reference. The results are obtained with out\_k=2.

| Dataset    | init model ([ListT5-base](https://huggingface.co/Soyoung97/ListT5-base), on A100) | re-run model ([ListT5-base-A6000](https://huggingface.co/Soyoung97/ListT5-base-A6000)) |
|------------|--------------------|-------------------------|
| trec-covid | 78.3%             | 79.0%                  |
| news       | 48.5%             | 49.2%                  |
| scidocs    | 17.6%             | 17.7%                  |
| scifact    | 74.1%             | 74.1%                  |
| dbpedia    | 43.7%             | 43.9%                  |
| nfcorpus   | 35.6%             | 35.9%                  |
| touche     | 33.4%             | 32.5%                  |
| **Avg.**   | **47.3%**         | **47.5%**              |


### Running the positional bias experiments
The code to run the positional bias experiments are located in the `./positional_bias_experiments/` directory.

The code in L.134 of `measure_per_consist_gpt.py` is to handle outliers, which was explained at Appendix page 17 of our ListT5 paper (We discard queries that doesn’t have positive indexes in the bm25 top100 dataset, or those that don’t have 4 distinct negative contexts.).


#### Citation
If you find this paper \& source code useful, please consider citing our paper:
```
@misc{yoon2024listt5listwisererankingfusionindecoder,
      title={ListT5: Listwise Reranking with Fusion-in-Decoder Improves Zero-shot Retrieval}, 
      author={Soyoung Yoon and Eunbi Choi and Jiyeon Kim and Hyeongu Yun and Yireun Kim and Seung-won Hwang},
      year={2024},
      eprint={2402.15838},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2402.15838}, 
}
```
