import jsonlines
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
import sys
sys.path.append('../')
from models.FiDT5 import FiDT5
import random
from tqdm import tqdm
import numpy as np
import glob

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

def write_jsonl(path, out):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(out)
    print('Write done!')

def make_input_tensors(tok, texts):
    raw = tok(texts, return_tensors='pt', padding='max_length', max_length=args.max_length, truncation=True).to('cuda')
    input_tensors = {'input_ids': raw['input_ids'].unsqueeze(0),
            'attention_mask': raw['attention_mask'].unsqueeze(0)}
    return input_tensors

def make_listwise_text(question, ctxs):
    out = []
    for i in range(len(ctxs)):
        text = f"Query: {question}, Index: {i+1}, Context: {ctxs[i]}"
        out.append(text)
    return out

def run_main(model_path, path, max_length, option='measure_goldn'):
    model = FiDT5.from_pretrained(model_path, encoder_output_k=max_length).to('cuda')
    data = read_jsonl(path)

    tok = T5Tokenizer.from_pretrained('t5-base', legacy=False)
    hits = [0,0,0,0,0]
    iteration = 0
    has_dup_in_gen = 0
    pos_hits = 0
    small_ctx = 0
    goldn = 0
    cache = []
    for total_i, instance in tqdm(enumerate(data), total=len(data)):
        question = instance['q_text']
        answer = [x for x in instance['qrels'].keys() if instance['qrels'][x] != 0]
        ctxs = instance['bm25_results']
        neg_ctxs = [x for x in instance['bm25_results'] if x['pid'] not in answer]
        pos_ctxs = [x for x in instance['bm25_results'] if x['pid'] in answer]
        if option == 'measure_goldn':
            goldn += len(pos_ctxs)
        elif option == 'permutation':
            if len(ctxs) < 5 or len(pos_ctxs) == 0 or len(neg_ctxs) < 5:
                continue
            pos_ctxs = [pos_ctxs[0]]
            for pos_ctx in pos_ctxs:
                iteration += 1
                neg_texts = random.sample(neg_ctxs, k=4)
                pos = (pos_ctx['title'] + ' ' + pos_ctx['text']).strip()
                pos = tok.decode(tok.encode(pos, max_length=max_length, truncation=True), skip_special_tokens=True)
                negs = [(x['title'] + ' ' + x['text']).strip() for x in neg_texts]
                negs = [tok.decode(tok.encode(x, max_length=max_length, truncation=True), skip_special_tokens=True) for x in negs]
                cache.append({'i': iteration, 'question': question,
                    'pos': pos,
                    'negs': negs, 'outputs': {}})
                if True:
                    for pos_idx in range(5):
                        full_texts = neg_texts[:pos_idx] + [pos_ctx] + neg_texts[pos_idx:]
                        #bm25_scores = [x['bm25_score'] for x in full_texts]
                        #sort_list = [str(x+1) for x in np.argsort(bm25_scores)]
                        input_texts = [f"Query: {question}, Index: {i+1}, Context: {x['text']}" for i, x in enumerate(full_texts)]
                        assert len(input_texts) == 5
                        source = make_input_tensors(tok, input_texts)
                        output = model.generate(**source, max_length=8, return_dict_in_generate=True, output_scores=True)
                        gen_out = tok.batch_decode(output.sequences, skip_special_tokens=True)[0].split(' ')
                        model_top_output_idx = int(gen_out[-1]) - 1
                        cache[-1]['outputs'][str(pos_idx)] = ' '.join(gen_out)
                        if pos_idx == model_top_output_idx:
                            hits[pos_idx] += 1
                    print(f"{iteration}: {hits}")
    print(f"Total iteration: {iteration}")
    print(f"1: {hits[0]/(iteration)}")
    print(f"2: {hits[1]/(iteration)}")
    print(f"3: {hits[2]/(iteration)}")
    print(f"4: {hits[3]/(iteration)}")
    print(f"5: {hits[4]/(iteration)}")
    write_jsonl('./trec-covid_input.jsonl', cache)
    if option == 'measure_goldn':
        avg_goldn = goldn / len(data)
        print(f"For {path}, goldn: {avg_goldn}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='../outputs/mmarco/fid/coco_from1000_sort5/tfmr_0_step20000', type=str)
    parser.add_argument('--dataname', default='fiqa', type=str)
    parser.add_argument('--max_length', default=512, type=int)
    args = parser.parse_args()
    path = f"../../dataset/beir_bm25/final_fromindex/{args.dataname}.jsonl"
    #paths = glob.glob('../../dataset/beir_bm25/final_fromindex/*.jsonl')
    #for path in paths:
    random.seed(0)
    run_main(args.model_path, path, args.max_length, option='permutation')
