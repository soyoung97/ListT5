import jsonlines
import json
from multiprocessing import Pool
import os
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
try:
    import utils.io_modules as io_modules
    from utils.eval_official.eval_retrieval import evaluate as official_evaluate
except:
    from eval_official.eval_retrieval import evaluate as official_evaluate

def make_gold_mapping(dataset):
    multi_res = defaultdict(list)
    print("\n Making gold mapping ... \n")
    for idx in tqdm(range(len(dataset['input']))):
        output_text = dataset['output'][idx]
        if 'title:' in output_text and 'content:' in output_text:
            output_text = output_text.split('title:')[1].split(', content:')[0].strip()
        #    output_text = output_text.split('@@')[0].strip()
        query, key = dataset['input'][idx], output_text
        multi_res[query].append(key)
    return dict(multi_res)

def make_gold_mapping_multiprocessing(dataset):
    res = defaultdict(list)
    pool = Pool(processes=os.cpu_count())
    inputs, outputs = dataset['input'], dataset['output']
    division_len = int(len(dataset['input'])/os.cpu_count() - 1)
    divided_data = [[{'input': inputs[x:x+division_len], 'output': outputs[x:x+division_len]}] for x in range(0, len(inputs), division_len)]
    results = pool.starmap(make_gold_mapping, divided_data)
    out = defaultdict(list)

    print("\nMultiprocessing done. Gathering...\n")
    for res in results:
        for key, value in tqdm(res.items()):
            out[key].append(value)
    return dict(out)

def save_dict_and_jsonl_dpr(result_dict, em_list=None, recall_list=None, output_dir=''):
    gold_res = []
    guess_res = []
    full_res = []
    for uid, idx in enumerate(range(len(result_dict['input']))):
        inp, out, pred, score = result_dict['input'][idx], result_dict['output'][idx], result_dict['predict'][idx], result_dict['score'][idx]
        # reformatting out
        gold_provenance = [{"wikipedia_title": out_text} for out_text in out]
        guess_provenance = [{"wikipedia_title": pred_text} for pred_text in pred]
        gold_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "", "provenance": gold_provenance}]})
        guess_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "", "provenance": guess_provenance}]})
        full_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "",
            "gold_provenance": gold_provenance, "guess_provenance": guess_provenance}]})
    # save result_dict
    dict_path = f"{output_dir}/dataset/result_dict.json"
    jsonl_path = f"{output_dir}/dataset/result"

    io_modules.write_json_file(dict_path, result_dict)
    print(f"\nDumped json result dict into : {dict_path}\n")

    for affix, res in zip(['gold', 'guess', 'full'], [gold_res, guess_res, full_res]):
        io_modules.write_jsonl_file(f"{jsonl_path}_{affix}.jsonl", res)
        print(f"\nWrite done for jsonl path into : {jsonl_path}_{affix}.jsonl\n")
        print(f"Example0:\n{res[0]}\n1:\n{res[1]}")

def format_official_jsonl_to_df(jsonl_file):
    """
    input: {'id': '11', 'input': 'question', 'output': [{'answer': '',
        'provenance': [{'wikipdeia_title': 'asdf'}, {}...]}
    return sth like:
   id                                              input                                             output
   0   0                    what do the 3 dots mean in math  [Therefore sign, Oganesson, Ogden Nash, Proper...
   1  16       who plays peter in what we do in the shadows  [What We Do in the Shadows, Mark O'Brien (acto...
   2   3  where was the world economic forum held this year  [World Trade Organization, World Trade Center ...
   3  12                   who sings got my mind set on you  [Got My Mind Set on You, I've Gotta Be Me, Get...
    """
    data = pd.DataFrame.from_dict(jsonl_file)
    raw_out = data['output'].tolist()
    raw_out = [[y['wikipedia_title'] for y in x[0]['provenance']] for x in raw_out]
    data['output'] = raw_out
    return data

def format_output_to_official_jsonl(input_list, gold_mapping, pred_list, output_dir=''):
    
    gold_res = []
    guess_res = []
    full_res = []
    seen_inp = set()
    for uid in range(len(input_list)):
        inp, pred = input_list[uid], pred_list[uid]
        # reformatting out
        if inp in seen_inp:
            continue
        else:
            seen_inp.add(inp)
        out = gold_mapping[inp]
        gold_provenance = [{"wikipedia_title": out_text} for out_text in out]
        guess_provenance = [{"wikipedia_title": pred_text} for pred_text in pred]
        gold_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "", "provenance": gold_provenance}]})
        guess_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "", "provenance": guess_provenance}]})
        full_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "",
            "gold_provenance": gold_provenance, "guess_provenance": guess_provenance}]})
    # save result_dict
    jsonl_path = f"{output_dir}/official_result"

    for affix, res in zip(['gold', 'guess', 'full'], [gold_res, guess_res, full_res]):
        io_modules.write_jsonl_file(f"{jsonl_path}_{affix}.jsonl", res)
        print(f"\nWrite done for jsonl path into : {jsonl_path}_{affix}.jsonl\n")
        print(f"Example0:\n{res[0]}\n1:\n{res[1]}")

    # get official metrics right away
    command = f"python ./utils/eval_official/eval_retrieval.py --guess {jsonl_path}_guess.jsonl --gold {jsonl_path}_gold.jsonl"
    print("--------------------Running official metrics right away ..... -------------------")
    print(f"Command: {command}")
    res = official_evaluate(f"{jsonl_path}_gold.jsonl", f"{jsonl_path}_guess.jsonl", [5], ['wikipedia_title'])
    #os.system(command)
    return res['Rprec']

def make_official_guess(input_list, pred_list, output_dir=''):
    guess_res = []
    seen_inp = set()
    for uid in range(len(input_list)):
        inp, pred = input_list[uid], pred_list[uid]
        # reformatting out
        if inp in seen_inp:
            continue
        else:
            seen_inp.add(inp)
        guess_provenance = [{"wikipedia_title": pred_text} for pred_text in pred]
        guess_res.append({'id': str(uid), 'input': inp, 'output': [{"answer": "", "provenance": guess_provenance}]})
    # save result_dict
    jsonl_path = f"{output_dir}/official_result"
    Path(jsonl_path).parent.mkdir(exist_ok=True, parents=True)
    io_modules.write_jsonl_file(f"{jsonl_path}_guess.jsonl", guess_res)
    print(f"\nWrite done for jsonl path into : {jsonl_path}_guess.jsonl\n")
    print(f"Example0:\n{guess_res[0]}\n1:\n{guess_res[1]}")

    # get official metrics right away
    command = f"python ./utils/eval_official/eval_retrieval.py --guess {jsonl_path}_guess.jsonl --gold ../dataset/full_kilt/nq/nq-dev-kilt-cleanedup.jsonl"
    print("--------------------Running official metrics right away ..... -------------------")
    print(f"Command: {command}")
    res = official_evaluate("../dataset/full_kilt/nq/nq-dev-kilt-cleanedup.jsonl", f"{jsonl_path}_guess.jsonl", [5], ['wikipedia_title'])
    #os.system(command)
    return res['Rprec']

def read_jsonl_to_df(paths, split):
    data = []
    for path in paths:
        data += io_modules.read_jsonl_file(path)
    full_inputs = []
    full_outputs = []
    for line in data:
        inputs = line['input']
        answers = line['output']
        titles = []
        for ans in answers:
            provs = ans.get('provenance')
            if provs is not None:
                for prov in provs:
                    title = prov.get('wikipedia_title')
                    if title is None:
                        title = prov.get('title')
                        if title is not None:
                            titles.append(title)
                    else:
                        titles.append(title)
        if split == 'train':
            titles = list(set(titles))
            for title in titles:
                full_inputs.append(inputs)
                full_outputs.append(title)
        else:
            full_inputs.append(inputs)
            full_outputs.append(titles[0])
    df = pd.DataFrame({'input': full_inputs, 'output': full_outputs})
    return df
