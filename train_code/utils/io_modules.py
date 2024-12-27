import jsonlines
import json
import pandas as pd
import pickle

def read_csv_file(data_path):
    df = pd.read_csv(data_path)
    return df

def read_pyserini_file(data_path):
    df = pd.read_csv(data_path, delimiter=' ', header=None)
    return df

def read_jsonl_file(file_path):
    data = []
    with jsonlines.open(file_path, 'r') as reader:
        for instance in reader:
            data.append(instance)
    return data


def write_jsonl_file(file_path, out):
    with jsonlines.open(file_path, 'w') as writer:
        writer.write_all(out)
    print(f"Wrote jsonl file to: {file_path}!")

def read_tsv_file(file_path, header=['question', 'pos', 'neg']):
    data = pd.read_csv(file_path, sep='\t', names=header)
    return data

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        res = json.load(f)
    return res

def write_json_file(file_path, res):
    with open(file_path, 'w') as f:
        json.dump(res, f, indent=4)
    print(f"Wrote json file to: {file_path}!")


def read_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def write_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Writing pickle file to {file_path} Done!")


def read_file_lists_to_df(paths):
    if 'csv' == paths[0][-3:]:
        print("\nreading csv(df) file. ")
        dfs = [pd.read_csv(path) for path in paths]
        if len(dfs) == 1:
            return dfs[0]
        else:
            full = dfs.pop()
            for df in dfs:
                full = full.append(df, ignore_index=True)
            return full
    elif 'jsonl' == paths[0][-5:]:
        print("\nreading jsonlines file. ")
        full_data = []
        for path in paths:
            data = read_jsonl_file(path)
            full_data.extend(data)
        return full_data
    elif 'json' == paths[0][-4:]:
        full_data = []
        print("\nreading json file. ")
        for path in paths:
            data = read_json_file(path)
            full_data.extend(data)
    else:
        raise NotImplementedError
