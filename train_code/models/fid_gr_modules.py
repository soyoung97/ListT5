import os
import sys
import json
import copy
import torch
import pickle
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from torch.utils.data import DataLoader
from itertools import chain
import utils.format as format_utils
import utils.io_modules as io_utils
import jsonlines
import pandas as pd
from torch.utils.data import Dataset
import json
from models.shared_modules import SharedDataset, SharedModel
from pprint import pprint
from torchmetrics import BLEUScore
from pathlib import Path
from itertools import chain
from transformers import T5ForConditionalGeneration
import math
import time
import random
from tqdm import tqdm

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class FiDGRDataset(SharedDataset):
    def __init__(self, tokenizer, split, args):
        super().__init__(tokenizer, split, args)
        self.debugi = 0
        self.split = split
        self.dataset = self.format_msmarco_data(self.dataset, split)
        self.len = len(self.dataset)

    def format_msmarco_data(self, dataset, split):
        res = []
        data = [x for x in dataset if len(x['ret']) == self.args.listwise_k]
        print(f"Orig: {len(dataset)}, After: {len(data)}")
        del self.dataset
        return data

    def __len__(self):
        return self.len

    def print_io(self, idx, input_, output_):
        print_str = "=" * 80
        print_str += f"\n({idx}) {self.split}\ninput: {input_}\n"
        print_str += f"\noutput: {output_}\n"
        print_str += "=" * 80
        print(print_str)

    def convert_listwise_to_features(self, idx):
        raw = self.dataset[idx]
        try:
            pos_idx = int(raw['pos_idx']) - 1  # index starts with 0
        except:
            print(f"Cannot convert [{raw['pos_idx']}] to int!!")
            pos_idx = raw['pos_idx'].split(' ')[0]
            pos_idx = int(pos_idx) - 1
        # positive idx normalization process
        try:
            pos_psg = raw['ret'][pos_idx]
        except IndexError:
            print(f"Error!! idx {pos_idx}, while len {len(raw['ret'])}")
            raise Exception
        neg_psgs = raw['ret'][:pos_idx] + raw['ret'][pos_idx + 1 :]
        rand_pos_idx = random.choice(range(self.args.listwise_k))
        shuffled_psgs = neg_psgs[:rand_pos_idx] + [pos_psg] + neg_psgs[rand_pos_idx:]

        bm25_scores = [x['bm25_score'] for x in shuffled_psgs]
        max_scores = max(bm25_scores)
        bm25_scores[rand_pos_idx] = max_scores + 10
        sort_list = [str(x+1) for x in np.argsort(bm25_scores)]
        if 'posfirst' in self.args.sub_mode:
            sort_list = reversed(sort_list)
        converted = ' '.join(sort_list)

        input_texts = [f"Query: {raw['query']}, Index: {i+1}, Context: {x['text']}" for i, x in enumerate(shuffled_psgs)]
        source = self.tokenizer(input_texts, padding='max_length', max_length=self.args.max_input_length, truncation=True, return_tensors='pt')
        target = self.tokenize_t5(converted, max_length=self.args.max_output_length)
        if self.should_print() or self.args.debug:
            self.print_io(idx, [input_texts[0], input_texts[-1]], raw['pos_idx'])
        return source, target

    def __getitem__(self, idx):
        source, target = self.convert_listwise_to_features(idx)
        res = {
            'idx': idx,
            "source_ids": source['input_ids'],
            "target_ids": target['input_ids'],
            "source_mask": source['attention_mask'],
            "target_mask": target['attention_mask'],
        }
        return res

class FiDGRModel(SharedModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        if torch.cuda.current_device() == 0:
            self.print = True
        else:
            self.print = False
        # If in testing mode, load ckpt for inference
        if not self.args.do_train:
            self.test_input_list = []
            self.test_pred_list = []
            self.test_score_list = []
        self.gen_time = 0
        self.text_len = 0

    def get_dataset(self, split):
        dataset = FiDGRDataset(
            tokenizer=self.tokenizer, split=split, args=self.args
        )
        return dataset

    def forward(self, input_ids, attention_mask, lm_labels, decoder_attention_mask):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=None,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _loss(self, batch):
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=self.encode_label_ids({'input_ids': batch['target_ids']})['input_ids'],
            decoder_attention_mask=batch["target_mask"],
        )
        #ForkedPdb().set_trace()
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        self.log(
                'global_step',
                self.global_step,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=True)
        loss = self._loss(batch)
        self.log(
            "train loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def _test_step(self, batch, return_elem=False):
        start = time.time()
        out = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            max_length=self.args.max_output_length,
            num_beams=self.args.val_beam_size,
            num_return_sequences=self.args.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.get(
                batch_id, sent.tolist()
            ),
            output_scores=True,
            return_dict_in_generate=True
        )
        _generated_ids = out.sequences
        scores = list(self.group2chunks(out.sequences_scores.cpu().tolist(), n=self.args.val_beam_size))
        self.text_len += batch['source_ids'].shape[0]
        _generated_text = self.ids_to_text(_generated_ids)
        inum = len(_generated_ids) // self.args.val_beam_size
        generated_text = [
            _generated_text[
                i * self.args.val_beam_size : (i + 1) * self.args.val_beam_size
            ]
            for i in range(inum)
        ]
        generated_text = list(generated_text)
        if return_elem:
            split = 'test'
        else:
            split = 'val'
        idxs = batch['idx'].tolist()
        input_questions = self.ids_to_text(batch['source_ids'])
        print(f"Test example: \nInput: {input_questions[0]},\nOutput: {generated_text[0]}\n Score: {scores[0]}")
        if return_elem:
            assert (
                len(input_questions)
                == len(list(generated_text))
            )
            return {
                "input": input_questions,
                "pred": generated_text,
                "score": scores
            }
        else:
            raise NotImplementedError

    def _val_step(self, batch, return_elem=False):
        loss = self._loss(batch)

        self.log(
                'val_loss',
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=True)
        return loss.item()


    def validation_step(self, batch, batch_idx):
        return self._val_step(batch)

    #def on_validation_epoch_end(self, outputs):
    #    epoch_loss = float(np.array(outputs).mean())
    #    self.log(
    #            'val_loss',
    #            epoch_loss,
    #            on_epoch=True,
    #            prog_bar=False,
    #            logger=True,
    #            sync_dist=True)
    #    return

    def test_step(self, batch, batch_idx):
        ret_dict = self._test_step(batch, return_elem=True)
        self.test_input_list.extend(ret_dict["input"])
        self.test_pred_list.extend(ret_dict["pred"])
        self.test_score_list.extend(ret_dict['score'])

    def _gather_object(self, obj):
        if self.print:
            print(f'## Gathering list from {self.args.n_gpu} process!')
        gathered = [None for _ in range(self.args.n_gpu)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def test_epoch_end(self, outputs):
        os.makedirs(self.args.output_dir, exist_ok=True)
        _input = self.test_input_list
        _pred = self.test_pred_list
        _score = self.test_score_list
        gold = self.get_dataset('test').dataset
        output = []
        for pred_questions, scores, gold_line in zip(_pred, _score, gold):
            gold_line['pred'] = {
                    'questions': pred_questions,
                    'scores': scores}
            output.append(gold_line)
        #jsonl_data = self.pack_results(_input, _pred, _score)
        file_path = f"{self.args.output_dir}/generated_output.json"
        io_utils.write_json_file(file_path, output)
        #print(f"gen_time: {self.gen_time}, text len: {self.text_len}. by 1 sec: {self.text_len/self.gen_time}, by 1: {self.gen_time/self.text_len}")
        #self.text_len = 0
        #self.gen_time = 0

        assert len(_input) == len(_pred)
        self.rprec = 0

 
