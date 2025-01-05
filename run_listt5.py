import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from transformers import LlamaForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration, LlamaTokenizer
import pickle
import itertools
import time
import argparse
from tqdm import tqdm
import jsonlines
import json
from pprint import pprint
import pandas as pd
import torch
import math
import glob
import copy
import numpy as np
import pickle
from transformers import pipeline
import sys
from pathlib import Path
from FiDT5 import FiDT5
import random
from beir_eval import run_rerank_eval
from beir_length_mapping import BEIR_LENGTH_MAPPING

sys.setrecursionlimit(10**7)


def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

class ListT5Evaluator():
    def __init__(self, args):
        self.idx = 0
        self.imsi = []
        self.args = args
        try:
            self.tok = T5Tokenizer.from_pretrained(self.args.model_path, legacy=False)
        except:
            print(f"Some issue with loading the T5Tokenizer from training ({self.args.model_path}), make sure the transformers version is 4.33.3 when running training.")
            print(f"Fallback to t5-base tokenizer (should not be too much of a problem)")
            self.tok = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        self.test_file = read_jsonl(self.args.input_path)
        print(f"Input path: {self.args.input_path}")
        self.idx2tokid = self.tok.encode(' '.join([str(x) for x in range(1, self.args.listwise_k+1)]))[:-1]
        self.model = self.load_model()
        self.num_forward = 0

    def write_json_file(self, path, data):
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Writing to {path} done!")

    def write_jsonl_file(self, path, data):
        if self.args.measure_flops:
            self.prof.stop_profile()
            self.flops = self.prof.get_total_flops()
        else:
            self.flops = 0
        print(f"Flops: {self.flops}!")
        with jsonlines.open(path, 'w') as writer:
            writer.write_all(data)
        print(f"Writing to {path} done!")

    def load_model(self):
        start = time.time()
        print("Loading model..")
        print(f"Loading fid model from {self.args.model_path}")
        model = FiDT5.from_pretrained(self.args.model_path).to('cuda')
        end = time.time()
        print(f"Done! took {end-start} second")
        model.eval()
        if self.args.measure_flops:
            self.prof = FlopsProfiler(model)
            self.prof.start_profile()
        return model

    def make_input_tensors(self, texts):
        raw = self.tok(texts, return_tensors='pt',
                padding=self.args.padding, max_length=self.args.max_input_length,
                truncation=True).to('cuda')
        input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
                'attention_mask': raw['attention_mask'].unsqueeze(0)}
        return input_tensors

    def make_listwise_text(self, question, ctxs):
        out = []
        for i in range(len(ctxs)):
            text = f"Query: {question}, Index: {i+1}, Context: {ctxs[i]}"
            out.append(text)
        return out

    def run_inference(self, input_tensors):
        output = self.model.generate(**input_tensors,
                max_length = self.args.max_gen_length,
                return_dict_in_generate=True, output_scores=True)
        self.num_forward += 1
        return output

    def get_rel_index(self, output, k=-1):
        if k == -1:
            k = self.args.out_k
        gen_out = self.tok.batch_decode(output.sequences, skip_special_tokens=True)
        out_rel_indexes = []
        for i, iter_out in enumerate(gen_out):
            out = iter_out.split(' ')[-k:]
            try:
                out_rel_index = [int(x) for x in out]
            except:
                print('!!'*30)
                print(f'Error in get_out_k. Output: {out}')
                print('!!'*30)
                out_rel_index = [1 for _ in range(k)]
            out_rel_indexes.append(out_rel_index)
        return out_rel_indexes

    def get_out_k(self, question, full_ctxs, index, use_cache=True, k=-1):
        if len(set(index)) == 1:
            return index[:k]
        else:
            index.sort()
        if k == -1:
            k = self.args.out_k
        if use_cache and self.best_cache.get(tuple(set(index))) is not None:
            return self.best_cache.get(tuple(set(index)))[-k:]
        ctxs = [full_ctxs[x] for x in index]
        full_input_texts = self.make_listwise_text(question, ctxs)
        input_tensors = self.make_input_tensors(full_input_texts)
        #he = time.time()
        output = self.run_inference(input_tensors)
        #hehe = time.time()
        #self.imsi.append(hehe-he)
        #print(self.imsi)
        out_k_rel_index = self.get_rel_index(output, k=k)[0]
        try:
            out_k_def_index = [index[x - 1] for x in out_k_rel_index]
        except IndexError:
            print(f"IndexError! {out_rel_index}")
            out_k_def_index = index[-k:]
        self.best_cache[tuple(set(index))] = out_k_def_index
        return out_k_def_index

    def get_leftover_idx(self, exclude, k, full_list):
        out = []
        i = 0
        exclude = list(set(exclude + self.global_exclude))
        allow_exclude = False
        if set(full_list) - set(exclude) == set():
            print(f"Get leftover: exclude: {exclude}, glob: {self.global_exclude}, k: {k}, full_list: {full_list}")
            print(f"Cannot get any dummy from this setup - exclude: {exclude}, full_list: {full_list}. Exceptionally adding duplicate dummies with exclude")
            allow_exclude = True
        while len(out) != k:
            if i == len(full_list): # adding dummy overflowed - start again! (can have duplicates)
                i = 0
            if allow_exclude or (full_list[i] not in exclude):
                out.append(full_list[i])
            i += 1
        return out

    def remove_duplicates(self, indexes):
        out = []
        for x in indexes:
            if x not in out:
                out.append(x)
        return out

    def group2chunks(self, l, n=5):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def get_full_order_in_one_loop(self, question, topk_ctxs, full_list_idx):
        assert len(full_list_idx) <= self.args.listwise_k
        expanded_index = full_list_idx[:]
        pointer = 0
        while len(expanded_index) != self.args.listwise_k:
            expanded_index.append(full_list_idx[pointer])
            pointer += 1
            pointer = pointer % len(full_list_idx)
        full_output = self.get_out_k(question, topk_ctxs, expanded_index, use_cache=False, k=self.args.listwise_k)
        dup_removed_full_order = []
        while len(full_output) != 0:
            idx = full_output.pop()
            if idx not in dup_removed_full_order:
                dup_removed_full_order.append(idx)
        print(f"Exceptional case: had only {len(topk_ctxs)} ctxs, full list: {full_list_idx}. Output order: {dup_removed_full_order}")
        return dup_removed_full_order

    def run_one_loop(self, question, topk_ctxs, full_list_idx):
        saved_index = []
        if (self.args.out_k * 2) > self.args.listwise_k:
            full_list_idx = self.remove_duplicates(full_list_idx)
        grouped_list_idxs = list(self.group2chunks(full_list_idx, n=self.args.listwise_k))
        # step 1: run chunkwise and select out_k from each chunk
        for cut_list in grouped_list_idxs:
            if len(cut_list) < self.args.listwise_k:
                other_index = self.get_leftover_idx(cut_list, self.args.listwise_k - len(cut_list), full_list_idx)
                saved_index += self.get_out_k(question, topk_ctxs, cut_list + other_index)
            else:
                if len(set(cut_list)) == 1:
                    saved_index.append(cut_list[0])
                else:
                    saved_index += self.get_out_k(question, topk_ctxs, cut_list)
        # step 2: aggregation
        if len(saved_index) < self.args.listwise_k: # agg, fill out missing and run final
            other_index = self.get_leftover_idx(saved_index, self.args.listwise_k - len(saved_index), full_list_idx)
            full_index = saved_index + other_index
            topk_out = self.get_out_k(question, topk_ctxs, full_index)
            return topk_out[-1]
        elif len(saved_index) > self.args.listwise_k:
            return self.run_one_loop(question, topk_ctxs, saved_index)
        elif len(saved_index) == 1:
            return saved_index[0]
        else:# length is exactly the same as listwise_k
            return self.get_out_k(question, topk_ctxs, saved_index)[-1]

    def check_valid_list(self, full_list):
        for exc in self.global_exclude:
            while exc in full_list:
                exc_idx = full_list.index(exc)
                new_val = (exc + 1) % len(full_list)
                while new_val in self.global_exclude:
                    new_val = (new_val + 1) % len(full_list)
                full_list[exc_idx] = new_val
        return full_list
    def run_batchwise_caching(self, batch_holder):
        for iter_start in tqdm(range(0, self.args.topk, self.args.listwise_k)):
            iter_end = iter_start + self.args.listwise_k
            questions = [x['question'] for x in batch_holder]
            topk_ctxs = [x['topk_ctxs'][iter_start:iter_end] for x in batch_holder]
            # make batchwise input
            full_input_texts_batchwise = [self.make_listwise_text(q, c) for q,c in zip(questions, topk_ctxs)]
            if len(full_input_texts_batchwise[0]) != self.args.listwise_k:
                continue
            raw_tensors_batchwise = [self.tok(x,
                padding = self.args.padding,
                return_tensors='pt',
                max_length = self.args.max_input_length,
                truncation=True) for x in full_input_texts_batchwise]
            batch_inputids = torch.stack([x['input_ids'] for x in raw_tensors_batchwise]).to('cuda')
            batch_attnmasks = torch.stack([x['attention_mask'] for x in raw_tensors_batchwise]).to('cuda')
            #he = time.time()
            output = self.run_inference({'input_ids': batch_inputids, 'attention_mask': batch_attnmasks})
            #ho = time.time()
            #self.imsi.append(ho-he)
            #print(self.imsi)
            del batch_inputids
            del batch_attnmasks
            batch_best_rel_ids = [[x-1 for x in topk] for topk in self.get_rel_index(output)]
            cand_def_ids = tuple(range(iter_start, iter_end)) # we can assume that it's already sorted
            batch_best_def_ids = [[cand_def_ids[x] for x in y] for y in batch_best_rel_ids]
            # add result to cache
            for i in range(len(batch_holder)):
                batch_holder[i]['best_cache'][tuple(set(cand_def_ids))] = batch_best_def_ids[i]
        return batch_holder

    def get_top100_goldidx(self, instance):
        if instance.get('qrels') is None:
            top100_goldidx = []
            return top100_goldidx
        top100_pids = [x[self.args.pid_key] for x in instance[self.args.firststage_result_key][:self.args.topk]]
        top100_goldidx = []
        gold_pids = [x for x in instance[self.args.qrels_key] if instance[self.args.qrels_key][x] != 0]
        for pid in gold_pids:
            try:
                top100_goldidx.append(top100_pids.index(pid))
            except ValueError:
                continue
        return top100_goldidx

    def run_tournament_sort(self):
        skip_idx = 0
        short_idx = 0
        normal_idx = 0
        cached_output = []
        print(f"Running first batchwise iteration..")
        batch_holder = []
        #if os.path.exists(self.args.output_path):
        #    temp = read_jsonl(self.args.output_path)
        #    print(f'Starting from len: {len(temp)}')
        #    len_temp = len(temp)
        #    self.test_file = self.test_file[len_temp:]
        #else:
        temp = []
        for i, instance in tqdm(enumerate(self.test_file), total=len(self.test_file)):
            question = instance[self.args.question_text_key]
            topk_ctxs = [f"{x[self.args.title_key]} {x[self.args.text_key]}".strip() for x in instance[self.args.firststage_result_key]][:self.args.topk]

            # handling exceptions
            # (2) prepare for skipping those that don't have gold in topk(100)
            top100_goldidx = self.get_top100_goldidx(instance)
            if len(top100_goldidx) == 0 and self.args.skip_no_candidate:
                if self.args.verbose:
                    print('No gold in bm25 top100. skip this instance')
                cached_output.append({'i': i, 'question': question, 'topk_ctxs': topk_ctxs, 'goldidx': [], 'best_cache': {}})
                skip_idx += 1
                # (3) don't batch calculate those that have shorter topk than topk(100)
            elif len(topk_ctxs) < self.args.topk:
                cached_output.append({'i': i, 'question': question, 'topk_ctxs': topk_ctxs, 'goldidx': top100_goldidx, 'best_cache': {}})
                short_idx += 1
            else:
                normal_idx += 1
                temp_instance = {'i': i, 'question': question, 'topk_ctxs': topk_ctxs, 'goldidx': top100_goldidx, 'best_cache': {}}
                batch_holder.append(temp_instance)
                if (len(batch_holder) == self.args.bsize) or (((i+1) == len(self.test_file)) and len(batch_holder) > 0):
                    output = self.run_batchwise_caching(batch_holder)
                    cached_output += output
                    batch_holder = [] # reset batch holder variables
        if len(batch_holder) != 0:
            cached_output += self.run_batchwise_caching(batch_holder)
            batch_holder = []
        cached_output = sorted(cached_output, key=lambda x: x['i']) # rearrange cache in orig order
        print(f"Running the rest, skip idx was {skip_idx}/{len(self.test_file)}, instance that has shorter ctx than {self.args.topk} was {short_idx}/{len(self.test_file)}, normal: {normal_idx}/{len(self.test_file)}")
        if len(cached_output) != len(self.test_file):
            print(f"Len of cached_output is {len(cached_output)}, where len of test file is {len(self.test_file)}.. should be the same!")
            import pdb; pdb.set_trace()
        for instance, cache in tqdm(zip(self.test_file, cached_output), total=len(cached_output)):
            if instance[self.args.question_text_key] != cache['question']:
                print(f"Something wrong!")
                import pdb; pdb.set_trace()
            top100_goldidx = cache['goldidx']
            saved_topones = []
            self.global_exclude = []
            self.best_cache = cache['best_cache']
            topk_ctxs = cache['topk_ctxs']
            question = cache['question']
            if len(top100_goldidx) == 0 and self.args.skip_no_candidate:
                if self.args.verbose:
                    print(f"No gold in candidate list. skipping.")
                temp.append(instance)
            elif len(topk_ctxs) == 0:
                temp.append(instance)
            elif len(topk_ctxs) == 1:
                if self.args.verbose:
                    print(f"Length of topk ctxs is 1. Skipping reranking.")
                temp.append(instance)
            else:
                full_list_idx = list(range(len(topk_ctxs)))
                if len(full_list_idx) <= self.args.listwise_k:
                    print('get full order in one loop')
                    saved_topones = self.get_full_order_in_one_loop(question, topk_ctxs, full_list_idx)
                    print('get full order in one loop done')
                else:
                    if len(full_list_idx) != len(cache['topk_ctxs']):
                        print("sth wrong!!!")
                        import pdb; pdb.set_trace()
                    for topk_i in range(min(self.args.rerank_topk, len(full_list_idx))):
                        top1_def_idx = self.run_one_loop(question, topk_ctxs, full_list_idx)
                        top1_rel_idx = full_list_idx.index(top1_def_idx)
                        self.global_exclude.append(top1_def_idx)
                        if (len(full_list_idx) <= self.args.rerank_topk) and (len(self.global_exclude) == len(full_list_idx)):
                            saved_topones.append(top1_def_idx)
                            break
                        if (self.args.out_k * 2) > self.args.listwise_k:
                             full_list_idx = full_list_idx[:top1_rel_idx] + full_list_idx[top1_rel_idx:]
                        else:
                            full_list_idx[top1_rel_idx] = (top1_def_idx + self.args.dummy_number) % len(full_list_idx)
                        full_list_idx = self.check_valid_list(full_list_idx)
                        saved_topones.append(top1_def_idx)
                        if set(top100_goldidx).issubset(set(saved_topones)) and self.args.skip_issubset:
                            if self.args.verbose:
                                print(f'Subset reached for gold: {top100_goldidx}, pred: {saved_topones}. Skipping the rest')
                            break
                if len(saved_topones) == len(full_list_idx): # no need for dummy, exceptional case
                    full_rank = saved_topones
                else:
                    full_rank = saved_topones[:]
                    for i in range(len(topk_ctxs)):
                        if i not in saved_topones:
                            full_rank.append(i)
                if len(saved_topones) != len(set(saved_topones)):
                    print("Something wrong!")
                    import pdb; pdb.set_trace()
                if self.args.verbose:
                    print(f"[{i}] gold: {top100_goldidx}, pred: {saved_topones}")
                if len(set(top100_goldidx).intersection(set(saved_topones))) == 0 and len(top100_goldidx) > 0 and self.args.verbose:
                    print(f'Could not find any idx in gold for top 10 :(')

                reranked_instances = []
                for i, rank_id in enumerate(full_rank):
                    template = instance[self.args.firststage_result_key][rank_id]
                    template['orig_'+self.args.score_key] = template[self.args.score_key]
                    template[self.args.score_key] = 100000 - i
                    reranked_instances.append(template)
                instance[self.args.firststage_result_key] = reranked_instances
                temp.append(instance)
                if len(temp) % 50 == 0:
                    self.write_jsonl_file(self.args.output_path, temp)
                    print(f"Writing jsonl to {self.args.output_path} done! for length: {len(temp)}")
        print(f'%%%%%%%%%DONE%%%%%%%%%%%')
        self.write_jsonl_file(self.args.output_path, temp)
        print(f"Writing jsonl to {self.args.output_path} done, for full length!")
        try:
            ndcg_10, string = run_rerank_eval(temp, combined=True)
            print(f"ndcg: {ndcg_10}")
        except:
            print('Error happened during running run_rerank_eval. skipping evaluation')
            return 'None', 'None'
        return ndcg_10, string

def run_reranker(args):
    module = ListT5Evaluator(args)
    ndcg_10, scores = module.run_tournament_sort()
    flops = module.flops
    num_forward = module.num_forward
    return ndcg_10, scores, flops, num_forward

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


    parser.add_argument('--model_path', default='Soyoung97/ListT5-base', type=str)
    parser.add_argument('--topk', default=100, type=int, help='number of initial candidate passages to consider') # or 1000
    parser.add_argument('--max_input_length', type=int, default=-1) # depends on each individual data setup
    parser.add_argument('--padding', default='max_length', type=str)
    parser.add_argument('--listwise_k', default=5, type=int)
    parser.add_argument('--rerank_topk', default=10, type=int)
    parser.add_argument('--out_k', default=2, type=int)
    parser.add_argument('--dummy_number', default=21, type=int)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--bsize', default=20, type=int) # depends on your gpu and input length. We recommend input_length->bsize as 256->128, 512->32, 1024->16, 1280->8 for t5-3b with GB gpu.
    parser.add_argument('--input_path', type=str, default='./trec-covid.jsonl')
    parser.add_argument('--output_path', type=str, default='./outputs/trec-covid.jsonl')

    # profiling setup
    parser.add_argument('--measure_flops', action='store_true')
    parser.add_argument('--skip_no_candidate', action='store_true', help='skip instances with no gold qrels included at first-stage retrieval for faster inference, only works when gold qrels are available')
    parser.add_argument('--skip_issubset', action='store_true', help='skip the rest of reranking when the gold qrels is a subset of reranked output for faster inference, only works when gold qrels are available')
    args = parser.parse_args()
    if args.measure_flops:
        from deepspeed.profiling.flops_profiler import FlopsProfiler
    res = {}
    random.seed(args.seed)
    args.max_gen_length = args.listwise_k + 2
    pprint(args)
    if args.max_input_length == -1:
        input_path = args.input_path.split('/')[-1]
        for name in BEIR_LENGTH_MAPPING:
            if name in input_path:
                args.max_input_length = BEIR_LENGTH_MAPPING[name]
                print(f"Setting max input length to {args.max_input_length} for {name}")
        if args.max_input_length == -1:
            print(f"Could not find automatic max_input_length assignment from the following dataset keys: {BEIR_LENGTH_MAPPING.keys()}. Please modify the input_length data name or specify max input length by giving it by arguments.")
            raise Exception
    Path(args.output_path).parent.mkdir(exist_ok=True, parents=True)
    start_time = time.time()
    ndcg_10, scores, flops, num_forwards = run_reranker(args)
    res['flops'] = flops
    res['num_forwards'] = num_forwards
    res[args.output_path] = scores
    res['ndcg@10'] = ndcg_10
    end_time = time.time()
    res['time_duration'] = end_time - start_time
    print(res)
    return res



if __name__ == '__main__':
    main()
