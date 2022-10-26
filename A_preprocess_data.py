# -*- coding: utf-8 -*-


import os
from collections import Counter
from utils import *
from config.base_config import BaseConfig

base_config = BaseConfig()
train_data_path = base_config.train_data_path

def preprocess_data():
    '''
    1：将train所有文本进行word编码，总word前面加上 "<PAD>", "<UNK>"
        {"<PAD>": 0, "<UNK>": 1, "，": 2, "？": 3, "的": 4, "么": 5, "一": 6, "国": 7, "！": 8, "是": 9,
    2：将label进行编码
    '''
    if not os.path.exists("./model"):
        os.mkdir("./model")
    if not os.path.exists("./log"):
        os.mkdir("./log")

    label2idx = {}
    words = []
    with open(train_data_path, encoding='utf-8') as f:
        for line in f:
            line = json.loads(line)
            label = line["label"]
            text = line["sentence"]
            if label not in label2idx:
                # 亮点:如果label没在的话，直接现有label总长度为最新的index
                label2idx[label] = len(label2idx)
            words.extend(list(text))

    # word进行编码
    words_counter = Counter(words).most_common(base_config.vocab_size - 2)
    words = ["<PAD>", "<UNK>"] + [w for w, c in words_counter]
    word_map = {word: idx for idx, word in enumerate(words)}
    dump_json(word_map, base_config.word2idx_path)
    dump_json(label2idx, base_config.label2idx_path)


if __name__ == '__main__':
    preprocess_data()