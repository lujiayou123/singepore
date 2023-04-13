'''
Author: k devjiang@outlook.com
Date: 2023-04-09 10:40:15
LastEditors: k devjiang@outlook.com
LastEditTime: 2023-04-13 02:51:45
FilePath: /srl4e/code/transfer.py
Description: 现代文2诗歌体，数据量很够，主要目的是测试数据量少的时候（10-20条），看看模型本身的low shot能力
'''

import random

import dill
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
import pandas as pd
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, T5Config, T5Tokenizer,
                          MT5ForConditionalGeneration)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


max_length = 120
batch_size = 32
global_lr = 3e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


# flan_t5_base_path = r'/home/xtjiang/pretrained_models/flan-t5-base'
# mt5_base_path = 'google/mt5-base'


tokenizer = T5Tokenizer.from_pretrained("K024/mt5-zh-ja-en-trimmed")
model = MT5ForConditionalGeneration.from_pretrained("K024/mt5-zh-ja-en-trimmed")


class T5Dataset(Dataset):
    def __init__(self, texts, targets) -> None:
        super(T5Dataset, self).__init__()
        self.texts = texts
        self.targets = targets

    def __getitem__(self, index):
        add_prefix = 'modern2poem: '
        print(add_prefix+self.texts[index].strip())
        print(self.targets[index])
        print(len(self.texts))
        print(index)
        return (add_prefix+self.texts[index].strip(), self.targets[index])

    def __len__(self):
        return len(self.texts)


class Trainer():
    def __init__(self) -> None:
        self.model = model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=global_lr)

    def load_model(self):
        model.load_state_dict(torch.load(
            '/home/xtjiang/srl4e/0317-add-target-emotion/16.pt'))

    def train(self, train_dataloader):
        report_loss = 0
        self.model.train()
        for (text, target) in tqdm(train_dataloader, position=0, leave=True):
            input = tokenizer(text, padding='max_length', truncation=True,
                              max_length=max_length, return_tensors='pt')
            input_ids = input['input_ids'].to(device)
            att_mask = input['attention_mask'].to(device)
            target_input = tokenizer(
                target, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
            target_input_ids = target_input['input_ids'].to(device)
            outputs = self.model(
                input_ids, attention_mask=att_mask, labels=target_input_ids)
            loss = outputs[0]
            report_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(report_loss)

    def test(self, test_dataloader, epoch):
        self.model.eval()
        pred_sequences = []
        gold_sequences = []
        input_sequences = []
        for (text, target) in tqdm(test_dataloader, position=0, leave=True):
            input = tokenizer(text, padding='max_length', truncation=True,
                              max_length=max_length, return_tensors='pt')
            input_ids = input['input_ids'].to(device)
            att_mask = input['attention_mask'].to(device)
            outputs = model.generate(
                input_ids=input_ids,
                min_length=1,
                max_length=100,)
            for i in range(outputs.shape[0]):
                output_text = ''.join(tokenizer.decode(
                    outputs[i], skip_special_tokens=True))
                # delete the space token between '<tag>', the reason is "<" is an added token for T5
                pred_sequences.append('generated:'+output_text)
            gold_sequences.extend(target)
            input_sequences.extend(text)
        data = pd.DataFrame()
        data["inputs"] = input_sequences
        data["preds"] = pred_sequences
        data["golds"] = gold_sequences
        data.to_csv(f'./output/lowshot_{epoch}.csv')
        print(data["preds"])
        return data
    
    def results_to_file(self, preds, golds):
        pass


if __name__ == '__main__':
    modern_train_data = 'parallel_data/modern_zh.train'
    poem_train_data = 'parallel_data/poem.train'
    modern_test_data = 'parallel_data/modern_zh.test'
    poem_test_data = 'parallel_data/poem.test'

    train_src = open(modern_train_data,'r', encoding="utf-8").readlines()[:10]
    train_tgt = open(poem_train_data,'r', encoding="utf-8").readlines()[:10]
    test_src = open(modern_test_data,'r', encoding="utf-8").readlines()[10:]
    test_tgt = open(poem_test_data,'r', encoding="utf-8").readlines()[10:]

    train_dataset = T5Dataset(train_src, train_tgt)
    # valid_dataset = T5Dataset(valid_texts,valid_cues,valid_labels)
    test_dataset = T5Dataset(test_src, test_tgt)
    train_dataloader = DataLoader(
        train_dataset, 4, num_workers=0, shuffle=True,)
    # valid_dataloader = DataLoader(valid_dataset,16,num_workers=0,shuffle=True, drop_last=True)
    test_dataloader = DataLoader(
        test_dataset, 2, num_workers=0, shuffle=False, drop_last=True)

    trainer = Trainer()
    for e in range(0, 50):
        trainer.train(train_dataloader)
        # torch.save(model.state_dict(), './ft_save_model.pt')
        trainer.test(test_dataloader, e)
