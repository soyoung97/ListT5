import json
import openai
import re
import os
import numpy as np
import argparse
import random
import time
import pandas as pd
from tqdm import tqdm
import jsonlines
import time
import glob
# Replace with your API key
openai.api_key = 'ADD_YOUR_API_KEY'


def load_data(json_file_path):
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    data = []
    with jsonlines.open(path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data

def write_jsonl(path, out):
    with jsonlines.open(path, 'w') as writer:
        writer.write_all(out)
    print(f"Write jsonl to {path} done!")


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def gpt3_evaluate(data, args):
    outputs = []
    data = data[args.idx:args.idx+30000]
    toklen = 0
    output_path = f"./gpt_out_fiqa_1106.jsonl"
    if os.path.exists(output_path):
        prev = read_jsonl(output_path)
        prev_len = len(prev)
        print(f"Read {prev_len} files. Starting from there.")
    else:
        prev_len = 0
    data = data[prev_len:]
    args.idx = prev_len
    for index, line in tqdm(enumerate(data), total=len(data)):
        # 117760 dataset
        print(f"Iteration: {args.idx + index}")
        res = []
        query = line['question']
        line['outputs'] = {}
        full_output = []
        for pos_idx in range(5):
            retry = True
            passages = line['negs'][:pos_idx] + [line['pos']] + line['negs'][pos_idx:]
            text = f"I will provide you with 5 passages, each indicated by a numerical identifier []. Rank the passages based on their relevance to the search query: {query}.\n\n"
            text += f"[1] {passages[0]}\n[2] {passages[1]}\n[3] {passages[2]}\n[4] {passages[3]}\n[5] {passages[4]}\n\n"
            text += f"Search Query: {query}.\n\n"
            text += f"Rank the 5 passages above based on their relevance to the search query. All the passages should be included and listed using identifiers, in descending order of relevance. "
            text += "The output format should be [] > [], e.g., [4] > [2]. Only respond with the ranking results, do not say any word or explain."
            while retry:
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo-1106",
                        messages=[
                        #a{"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                        ],
                        n=1,  # Generate 1 output per attempt
                        stop=None,
                        temperature=0,
                    )
                    response_text = response.choices[0].message.content.strip()
                    token_num = response['usage']['total_tokens']
                    toklen += token_num
                    print(f"\n{index}, pos {pos_idx}: ({token_num}/{toklen}) \n\n{text}\n\nGPT3 output>>> {response_text} (pos_idx(+1): {pos_idx+1})")
                    line['outputs'][pos_idx] = response_text
                except Exception as e:
                    error = str(e)
                    print(f"Error!! {error}\n")
                    if "overloaded with other requests." in error:
                        continue
                    else:
                        print("Sleeping 10 sec and trying again.")
                        time.sleep(10)
                        continue
                retry = False
        output_path = f"./gpt_out_fiqa_1106.jsonl"
        if os.path.exists(output_path):
            full_output = read_jsonl(output_path)
        else:
            print(f"No prev {output_path}. Creating one.")
            full_output = []
        full_output.append(line)
        write_jsonl(output_path, full_output)
    return

def main(data, args):
    results = gpt3_evaluate(data, args)
    print('Done!!')

def get_out_index(string, option='gpt'):
    if option=='gpt':
        return int(string[1]) - 1
    else:
        return int(string.split(' ')[-1]) - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ChatGPT outputs from a JSON file")
    parser.add_argument("--file_path", help="Path to the first input JSON file containing the data instances",
            type=str, default='./gpt_out_fiqa.jsonl')
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument('--option', type=str, default='gpt')
    args = parser.parse_args()
    #args.file_path = './scripts/fiqa_input_second_withinfo.jsonl'
    data = read_jsonl(args.file_path) #, args.seed
    #main(data, args)
    res = [0,0,0,0,0]
    cleanup_res = []
    agreement = {'same': 0, 'diff': 0}
    for instance in data:
        if instance['i'] in [278, 699, 714]:
            continue
        outputs = instance['outputs']
        agg_results = []
        for i in range(5):
            out_index = get_out_index(outputs[str(i)], option=args.option)
            if out_index == i:
                res[i] += 1
            agg_results.append(out_index)
        if agg_results in [[0,1,2,3,4], [1,0,0,0,0], [2,2,1,1,1], [3,3,3,2,2], [4,4,4,4,1]]:
            agreement['same'] += 1
        else:
            agreement['diff'] += 1
    print(res)
    print([x/len(data) for x in res])
    #print([x/(len(data)-3) for x in res])
    print(f"Agreement: {agreement}")
    #print([agreement[x]/(len(data)-3) for x in agreement])
    print([agreement[x]/len(data) for x in agreement])
    #main(data, args)
