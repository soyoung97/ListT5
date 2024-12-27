from torch.utils.data import Dataset, DataLoader
import utils.format as utils_format
import utils.io_modules as utils_io
import pandas as pd
import os
import re
import torch
import pickle
import numpy as np
import string
from datasets import load_dataset
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, Adafactor, T5Model, BartTokenizer, BartForConditionalGeneration
from models.FiDT5 import FiDT5
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

class SharedDataset(Dataset):
    def __init__(self, tokenizer, split, args):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer
        if split == 'train':
            data_paths = self.args.train_files
        elif split == 'validation':
            data_paths = self.args.eval_files
        elif split == 'test':
            data_paths = self.args.test_files
        elif split == 'predict':
            data_paths = self.args.test_files
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")
        self.dataset = []
        for data_path in data_paths:
            if 'jsonl' in data_path:
                self.dataset += utils_io.read_jsonl_file(data_path)
            elif 'json' in data_path:
                self.dataset += utils_io.read_json_file(data_path)
            elif 'tsv' in data_path:
                self.dataset += utils_io.read_tsv_file(data_path).to_dict('records')

    def squeeze_t5(self, out):
        return {"input_ids": out['input_ids'].squeeze(0), "attention_mask": out['attention_mask'].squeeze(0)}

    def tokenize_t5(self, text, max_length=-1):
        if max_length == -1:
            max_length = self.args.max_input_length
        return self.squeeze_t5(self.tokenizer(text, padding='max_length',
            max_length=max_length, truncation=True, return_tensors='pt'))

    def should_print(self, prob=0.0005):
        return bool(np.random.choice(2, 1, p=[1-prob, prob]))

class SharedModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        #if True: #$ 't5' in self.args.base_model:
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.args.base_model, legacy=False)
        except:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base', legacy=False)
        if self.args.do_train:
            self.model = self.select_model_by_mode(self.args.base_model)
        else:
            self.model = self.select_model_by_mode(self.args.test_model_path)
        self.print_excel_formatted_args()

    def should_print(self, prob=0.00001):
        return bool(np.random.choice(2, 1, p=[1-prob, prob]))


    def print_excel_formatted_args(self):
        dict_args = vars(self.args)
        args_keys = list(dict_args.keys())
        args_keys.sort()
        print("\nkeys: ")
        print("; ".join(args_keys))
        ans = []
        for key in args_keys:
            val = dict_args[key]
            ans.append(str(val))
        print("\nvalues: ")
        print("; ".join(ans))
        return ans

    def select_model_by_mode(self, model_path):
        if self.args.load_from_fid:
            model = FiDT5.from_pretrained(model_path, encoder_output_k=self.args.encoder_output_k)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_path, device_map='auto')
            fid_model = FiDT5(model.config, encoder_output_k=self.args.encoder_output_k)
            fid_model.load_t5(model.state_dict())
            model = fid_model
        print(f"\nLoading model from {model_path}!\n")
        return model

    def train_dataloader(self):
        train_dataset = self.get_dataset('train')
        print(f"## Number of train dataset: {len(train_dataset)}")
        return DataLoader(train_dataset, shuffle=True, batch_size=self.args.train_batch_size, drop_last=False, num_workers=self.args.num_workers)
        # commenting due to large size. Uncomment if necessary.
        #self.dataloader_train = train_dataloader
        #return train_dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset('validation')
        print(f"## Number of val dataset: {len(val_dataset)}")
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.args.eval_batch_size, drop_last=False, num_workers=self.args.num_workers)
        #self.dataloader_val = val_dataloader
        return val_dataloader

    def test_dataloader(self):
        test_dataset = self.get_dataset('test')
        print(f"## Number of test dataset: {len(test_dataset)}")
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=self.args.eval_batch_size, drop_last=False, num_workers=self.args.num_workers)
        #self.dataloader_test = test_dataloader
        return test_dataloader

    def lmap(self, f, x):
        return list(map(f, x))

    def ids_to_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def encode_label_ids(self, labels):
        try:
            labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        except RuntimeError:
            return self.encode_label_ids({'input_ids': labels['input_ids'].detach().clone()})
        return labels

    def decode_label_ids(self, labels):
        try:
            labels['input_ids'][labels['input_ids'] == -100] = self.tokenizer.pad_token_id
            text_labels = self.tokenizer.batch_decode(labels['input_ids'], skip_special_tokens=True)
        except RuntimeError:
            return self.decode_label_ids({'input_ids': labels['input_ids'].detach().clone()})
        return text_labels

    def group2chunks(self, l, n=5):
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def configure_optimizers(self):
        if self.args.run_3b:
            #return FusedAdam(self.parameters())
            return DeepSpeedCPUAdam(self.parameters())
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            warmup_init=False,
            scale_parameter=False,
            relative_step=False,
        )

        if self.args.lr_scheduler == "constant":
            return [optimizer]
        elif self.args.lr_scheduler == 'constant_warmup':
            print("Using constant scheduler with warmup")
            scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.args.warmup_steps)
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        elif self.args.lr_scheduler == 'linear':
            print("Using linear scheduler")
            scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
            )
            scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
            return [optimizer], [scheduler]
        elif self.args.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            if self.args.accelerator == 'deepspeed':
                denominator = self.args.n_gpu
            elif self.args.accelerator == 'dp':
                denominator = 1
            steps_per_epoch = (
                (len_data // denominator) + 1
            ) // self.args.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.args.learning_rate,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                epochs=self.args.num_train_epochs,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            self.scheduler = scheduler
            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "name": "learning_rate"}
            ]
        else:
            raise NotImplementedError("Choose lr_schduler from (constant|exponential)")

    def pack_results(self, inputs, pred, score):
        output = []
        for idx, question in enumerate(inputs):
            instance = {
                    'idx': idx,
                    'question': question,
                    'pred': {
                        'questions': pred[idx],
                        'scores': score[idx]
                        }
                    }
            output.append(instance)
        return output

    def on_save_checkpoint(self, checkpoint):
        self.prev_save_epoch = self.current_epoch
        save_path = os.path.join(
            self.args.output_dir, f"tfmr_{self.current_epoch}_step{self.global_step}"
        )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
