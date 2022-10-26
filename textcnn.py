# -*- coding: utf-8 -*-

'''
textcnn模型
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    '''
    CNN 类
    '''
    def __init__(self, config):
        super(TextCNN, self).__init__()

        # embedding,可以预加载,可是设置是否训练
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # embedding:[V,E]
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=0)


        # CNN:每个尺寸的卷积核有一个网络
        # nn.Conv2d(in_channels=3,out_channels=64,kernel_size=4,stride=1,padding=0)
        # nn.Conv2d参数: [1, config.num_filters, (k, config.embed_size)]
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        # 每篇文章最终为1个值，则最终为一篇文章最终为 num_filters * len(config.filter_sizes) 个
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)


    def conv_and_pool(self, x, conv):
        '''

        '''
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        '''
        x为文本index:
           尺寸：[B,S]
           [[1,1,3,4,0],[1,2,2,2,1]]
        '''
        # 得到:[B,S,E]
        out = self.embedding(x)
        # 得到:[B,1,S,E]
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out