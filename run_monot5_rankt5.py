import argparse
import torch
import transformers
import numpy as np
import tqdm
import jsonlines
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tqdm import tqdm
import sys
import os
from beir_eval import run_rerank_eval
from beir_length_mapping import BEIR_LENGTH_MAPPING
from pathlib import Path

sys.setrecursionlimit(10**7)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as f:
        for instance in f:
            data.append(instance)
    return data

def write_jsonl(path, data):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(data)
    print(f"Written jsonl file to {path}!")
    return

def set_seed(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    return

class MonoRankT5Runner():
    def __init__(self, args):
        self.args = args
        self.test_file = read_jsonl(self.args.input_path)
        self.model, self.tok = self.load_model_and_tokenizer()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(self.model)
            self.prof.start_profile()

    def group2chunks(self, l, n=5):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def load_model_and_tokenizer(self):
        start = time.time()
        print("Loading model..")
        model = T5ForConditionalGeneration.from_pretrained(self.args.model).to('cuda')
        model.eval()
        end = time.time()
        tok = T5Tokenizer.from_pretrained(self.args.model)
        print(f"Loading model done! Took {end-start} seconds")
        return model, tok

    def run_inference(self, input_tensors):
        output = self.model.generate(**input_tensors, max_length=2,
                return_dict_in_generate=True, output_scores=True)
        return output

    def run_baseline(self):
        temp = []
        for i, instance in tqdm(enumerate(self.test_file), total=len(self.test_file)):
            question = instance[self.args.question_text_key]
            topk_ctxs = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in instance[self.args.firststage_result_key]][:self.args.topk]
            if self.args.mode == 'monot5':
                input_texts = [f"Query: {question} Document: {x} Relevant:" for x in topk_ctxs]
            elif self.args.mode == 'rankt5':
                input_texts = [f"Query: {question} Document: {x}" for x in topk_ctxs]
            grouped_input_texts = list(self.group2chunks(input_texts, n=self.args.bsize))
            scores_holder = []
            for batch_input_texts in grouped_input_texts:
                input_tensors = self.tok(batch_input_texts, return_tensors='pt',
                        padding='max_length', max_length=self.args.max_input_length, truncation=True).to('cuda')
                outputs = self.run_inference(input_tensors)
                del input_tensors
                scores = torch.stack(outputs.scores)
                if self.args.mode == 'monot5':
                    yesno_softmax_scores = torch.nn.functional.log_softmax(scores[0][:, [1176, 6136]], dim=1)[:, 0].tolist() # true, false
                    scores_holder += yesno_softmax_scores
                elif self.args.mode == 'rankt5':
                    rankt5_scores = scores[0][:, 32089].tolist() # <extra_id_10>
                    scores_holder += rankt5_scores
            all_scores_tensor = torch.tensor(scores_holder)
            rank = torch.argsort(all_scores_tensor).tolist()
            rank.reverse()
            reranked_instances = []
            for rank_id in rank:
                template = instance[self.args.firststage_result_key][rank_id]
                reranked_scores = scores_holder[rank_id]
                template[self.args.score_key] = reranked_scores
                reranked_instances.append(template)
            instance[self.args.firststage_result_key] = reranked_instances
            temp.append(instance)
        print(f"Inference done. writing to. {self.args.output_path}...")
        write_jsonl(self.args.output_path, temp)
        ndcg_10, string = run_rerank_eval(temp, combined=True)
        print(f"ndcg: {ndcg_10}")
        if self.args.measure_flops:
            self.prof.stop_profile()
            self.flops = self.prof.get_total_flops()
            print(f"FLOPS: {self.flops}")
        return ndcg_10, string


def main():
    parser = argparse.ArgumentParser()
    # Dataset key setup
    parser.add_argument('--firststage_result_key', default='bm25_results', type=str)
    parser.add_argument('--docid_key', default='docid', type=str)
    parser.add_argument('--pid_key', default='pid', type=str)
    parser.add_argument('--qrels_key', default='qrels', type=str)
    parser.add_argument('--score_key', default='bm25_score', type=str)
    parser.add_argument('--question_text_key', default='q_text', type=str)
    parser.add_argument('--text_key', default='text', type=str)
    parser.add_argument('--title_key', default='title', type=str)


    parser.add_argument('--model', default='castorini/monot5-base-msmarco-10k', type=str)
    parser.add_argument('--topk', default=100, type=int)
    parser.add_argument('--input_path', default='./trec-covid.jsonl', type=str)
    parser.add_argument('--output_path', default='./outputs-monot5/trec-covid.jsonl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--max_input_length', default=-1, type=int)
    parser.add_argument('--bsize', default=100, type=int)
    parser.add_argument('--eval_step_size', default=100, type=int)
    parser.add_argument('--mode', default='monot5', type=str) #or rankt5
    parser.add_argument('--measure_flops', action='store_true')
    args = parser.parse_args()
    print(f"Args: {args}")
    if args.measure_flops:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
    if args.max_input_length == -1:
        input_path = args.input_path.split('/')[-1]
        for name in BEIR_LENGTH_MAPPING:
            if name in input_path:
                args.max_input_length = BEIR_LENGTH_MAPPING[name]
        if args.max_input_length == -1:
            print(f"Could not find automatic max_input_length assignment from the following dataset keys: {BEIR_LENGTH_MAPPING.keys()}")
            print(f"Please modify the data name in input_path or specify max input length by giving it by arguments.")
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    runner = MonoRankT5Runner(args)
    runner.run_baseline()

if __name__ == '__main__':
    main()
