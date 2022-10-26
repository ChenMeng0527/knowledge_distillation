# coding = utf-8

'''
用蒸馏后的模型进行serving
'''

import torch
from config import textcnn_config
from textcnn import TextCNN
from data_helper_textcnn import CnnDataSet

# 1: 加载模型
kd_model_path = ''
model = TextCNN(textcnn_config)
model.load_state_dict(kd_model_path)


def dev(model, data_loader, config):
    device = config.device
    idx2label = {idx: label for label, idx in config.textcnn_config.label2idx.items()}
    model.to(device)
    model.eval()
    pred_labels, true_labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            cnn_ids, labels, input_ids, token_type_ids, attention_mask = batch[0].to(device), batch[1].to(device), \
                                                                         batch[2].to(device), batch[3].to(device), \
                                                                         batch[4].to(device)
            logits = model(cnn_ids)
            preds = torch.argmax(logits, dim=1)
            pred_labels.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())
    pred_labels = [idx2label[i] for i in pred_labels]
    true_labels = [idx2label[i] for i in true_labels]
    acc = sum([1 if p == t else 0 for p, t in zip(pred_labels, true_labels)]) * 1. / len(pred_labels)

    return acc, table



test_loader = DataLoader(KDdataset(config.base_config.test_data_path), batch_size=config.batch_size, shuffle=False)
best_model = TextCNN(config.textcnn_config)
best_model.load_state_dict(torch.load(config.model_path))
acc, table = dev(best_model, test_loader, config)



# 2: 输入text,模型将其预测
text = '中华人民共和国'
CnnDataSet('').get_signal_text_encode(text)


# 3: 获取线上输入
text = '中华人民共和国'

# 4: 进行预测