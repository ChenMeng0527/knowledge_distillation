# -*- coding: utf-8 -*-

'''
bert输出 + dropout + FC
'''

import torch
import torch.nn as nn
from transformers import BertModel


class Bert(nn.Module):

    def __init__(self, config):
        super(Bert, self).__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-chinese")
        self.fc = nn.Linear(config.hidden_size, config.class_num)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        # 输出为pool后的结果，每个句子一个embedding  [B,E]
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        out = self.fc(pooled_output)
        return out

