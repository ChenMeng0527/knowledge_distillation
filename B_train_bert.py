# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from bert import Bert
from config.bert_config import BertConfig
from data_helper_bert import BertDataSet
from sklearn.metrics import classification_report
from utils import get_logger
from tqdm import tqdm

def dev(model, data_loader, config):
    '''
    对数据集进行预测并查看指标
    '''
    device = config.device
    idx2label = {idx: label for label, idx in config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    # 不需要更新模型
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), \
                                                                batch[1].to(device), \
                                                                batch[2].to(device), \
                                                                batch[3].to(device)
            logits = model(input_ids, token_type_ids, attention_mask)
            # 预测01
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    # 预测、真实 label
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    # acc
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    # 调用sklearn查看指标
    table = classification_report(true_labels, pred_labels)
    return acc, table


def train():
    # bert 配置文件
    config = BertConfig()
    device = config.device

    # log
    logger = get_logger(config.log_path, 'train_bert')

    # bert模型
    model = Bert(config)
    model.to(device)
    model.train()

    # 对train / dev数据集进行encode
    train_dataset = BertDataSet(config.base_config.train_data_path)
    dev_dataset = BertDataSet(config.base_config.dev_data_path)
    # 调用DataLoader加载tensor
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)

    # Adam训练器
    optimizer = AdamW(model.parameters(), lr=config.lr)
    # 交叉熵
    criterion = nn.CrossEntropyLoss()
    # 指标
    best_acc = 0.

    for epoch in tqdm(range(config.epochs)):
        for i, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            input_ids, token_type_ids, attention_mask, labels = batch[0].to(device), \
                                                                batch[1].to(device), \
                                                                batch[2].to(device), \
                                                                batch[3].to(device)
            # print(input_ids,token_type_ids,attention_mask)
            logits = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(logits, labels)
            # loss求导
            loss.backward()
            # 优化器进行参数更新
            optimizer.step()

            # 定时查看batch预测指标
            if i % 100 == 0:
                preds = torch.argmax(logits, dim=1)
                acc = torch.sum(preds == labels)*1. / len(labels)
                print(i, "TRAIN: epoch: {} step: {} acc: {}, loss: {}".format(epoch, i, acc, loss.item()))
                logger.info("TRAIN: epoch: {} step: {} acc: {}, loss: {}".format(epoch, i, acc, loss.item()))


        # 每一轮结束后对验证数据进行预测并对比指标
        acc, cls_report = dev(model, dev_dataloader, config)
        print(acc, cls_report)
        logger.info("DEV: epoch: {} acc: {}".format(epoch, acc))
        logger.info("DEV classification report:\n{}".format(cls_report))

        # 更新模型
        if acc > best_acc:
            print('更新模型---', epoch)
            torch.save(model.state_dict(), config.model_path)
            best_acc = acc


    # --------------测试集进行----------------
    # 对测试数据
    test_dataset = BertDataSet(config.base_config.test_data_path)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    best_model = Bert(config)
    # 加载最优模型
    best_model.load_state_dict(torch.load(config.model_path))
    acc, cls_report = dev(best_model, test_dataloader, config)
    logger.info("TEST: ACC:{}".format(acc))
    logger.info("TEST classification report:\n{}".format(cls_report))


if __name__ == "__main__":
    train()
