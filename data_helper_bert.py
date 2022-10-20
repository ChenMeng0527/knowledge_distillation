# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from config.bert_config import BertConfig
from utils import *


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
config = BertConfig()


def encode_fn(text_list):
    """将输入句子编码成BERT需要格式"""
    tokenizers = tokenizer(
                            text_list,
                            padding=True,
                            truncation=True,
                            max_length=config.base_config.max_seq_len,
                            return_tensors='pt',  # 返回的类型为pytorch tensor
                            is_split_into_words=True
                        )
    # [ 101,  791, 1921, 1920, 4669, 1920, 6649,  102,    0,    0]
    input_ids = tokenizers['input_ids']
    # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    token_type_ids = tokenizers['token_type_ids']
    # [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    attention_mask = tokenizers['attention_mask']

    return input_ids, token_type_ids, attention_mask



class BertDataSet(Dataset):
    '''
    bert输入类
    输入:
        input_ids:[ 101,  791, 1921, 1920, 4669, 1920, 6649,  102,    0,    0]
        token_type_ids:[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        attention_mask:[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    '''
    def __init__(self, data_path):
        texts, labels = [], []
        label2idx = config.label2idx
        with open(data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                labels.append(label2idx[line["label"]])
                texts.append(line["sentence"].split())
        self.labels = torch.tensor(labels)
        # 将输入文本list进行encode, 输入三个
        self.input_ids, self.token_type_ids, self.attention_mask = encode_fn(texts)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.labels[index]

