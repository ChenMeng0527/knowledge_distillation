# -*- coding: utf-8 -*-


from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from textcnn import TextCNN
from data_helper_textcnn import *
from config.textcnn_config import TextCNNConfig
from sklearn.metrics import classification_report
from utils import get_logger


def dev(model, data_loader, config):
    '''
    验证集上评估
    '''
    device = config.device
    idx2label = {idx: label for label, idx in config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    # 不进行更新
    with torch.no_grad():
        for i, (texts, labels) in enumerate(data_loader):
            # batch数据
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            logits = model(texts)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)
    table = classification_report(true_labels, pred_labels)
    return acc, table


def train():
    # ---配置文件---
    config = TextCNNConfig()
    # ---log---
    logger = get_logger(config.log_path, "train_textcnn")

    # ---输入数据，将文本label都index化---
    # 文本: [[1,2,3,5,0],[2,1,2,4,5]]
    # label: [0,2,4,0]
    train_loader = DataLoader(CnnDataSet(config.base_config.train_data_path), batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(CnnDataSet(config.base_config.dev_data_path), batch_size=config.batch_size, shuffle=False)

    # ---模型---
    model = TextCNN(config)
    model.train()
    model.to(config.device)

    # 优化器
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    best_acc = 0.

    for epoch in range(config.num_epochs):
        for i, (texts, labels) in enumerate(train_loader):
            # 梯度为0
            optimizer.zero_grad()

            # batch数据进行预测
            texts = texts.to(config.device)
            labels = labels.to(config.device)
            logits = model(texts)

            # 损失函数
            loss = F.cross_entropy(logits, labels.long())

            # 损失函数求导，然后优化器更新
            loss.backward()
            optimizer.step()

            # 100batch时候打印日志
            if i % 100 == 0:
                labels = labels.data.cpu().numpy()
                preds = torch.argmax(logits, dim=1)
                preds = preds.data.cpu().numpy()
                acc = np.sum(preds == labels) * 1. / len(preds)
                logger.info("TRAIN: epoch: {} step: {} acc: {} loss: {} ".format(epoch + 1, i, acc, loss.item()))

        acc, table = dev(model, dev_loader, config)

        logger.info("DEV: acc: {} ".format(acc))
        logger.info("DEV classification report: \n{}".format(table))

        if acc > best_acc:
            torch.save(model.state_dict(), config.model_path)
            best_acc = acc


    # ---测试数据预估---
    test_loader = DataLoader(CnnDataSet(config.base_config.test_data_path), batch_size=config.batch_size, shuffle=False)
    best_model = TextCNN(config)
    best_model.load_state_dict(torch.load(config.model_path))
    acc, table = dev(best_model, test_loader, config)

    logger.info("TEST acc: {}".format(acc))
    logger.info("TEST classification report:\n{}".format(table))


if __name__ == "__main__":
    train()