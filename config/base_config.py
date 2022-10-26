# -*- coding: utf-8 -*-
'''
base中配置
'''


class BaseConfig():
    def __init__(self):
        self.train_data_path = "./data/train.json"      # 训练数据
        self.test_data_path = "./data/test.json"        # 测试数据
        self.dev_data_path = "./data/dev.json"          # 验证数据

        self.label2idx_path = "./data/label2idx.json"   # index
        self.word2idx_path = "./data/word2idx.json"     # index
        self.vocab_size = 5000                          # 最大词组
        self.max_seq_len = 100                          # 文本最长度