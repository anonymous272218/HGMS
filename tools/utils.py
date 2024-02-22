import json
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from nltk.stem import porter
import sys
from rouge_score.rouge_scorer import RougeScorer

sys.setrecursionlimit(10000)

stemmer = porter.PorterStemmer()

def calc_rouge(hyps, refer):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    for i in range(len(hyps)):
        score = scorer.score(refer[i], hyps[i])
        rouge1 += score["rouge1"].fmeasure
        rouge2 += score["rouge2"].fmeasure
        rougeLsum += score["rougeLsum"].fmeasure
    cnt = len(hyps)
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt

    return rouge1, rouge2, rougeLsum

def calc_ext_loss(outputs, label, criterion):
    output_flattened = outputs.view(-1)
    label_flat = label.view(-1)
    loss = criterion(output_flattened, label_flat)
    loss *= label_flat != -100
    loss = loss.view(outputs.size(0), outputs.size(1))
    loss = loss.sum(-1)
    loss = loss.mean()
    return loss

def stem_word_list(word_list):
    return [stemmer.stem(word) if len(word) > 3 else word for word in word_list]


def transfer_data(data, device):
    result = {}
    for key in data.keys():
        if key in ['num_nodes', 'syntax_sent_nums', 'text', 'bert_input']:
            result[key] = data[key]
        elif isinstance(data[key], list):
            items = []
            for item in data[key]:
                items.append(item.to(device))
            result[key] = items
        else:
            result[key] = data[key].to(device)
    return result


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe


def read_jsonl(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json(path):
    with open(path, "r") as file:
        obj = json.load(file)
    return obj


def save_json(path, obj):
    with open(path, "w") as file:
        json.dump(obj, file)


def read_text(fname):
    data = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data


def pad_matrix(matrix_list, batch_first=True, padding_value=0.0):
    max_cols = max(matrix.size(1) for matrix in matrix_list)
    result = []
    for matrix in matrix_list:
        matrix = torch.cat((matrix, torch.full((matrix.size(0), max_cols - matrix.size(1)), fill_value=padding_value,
                                               dtype=matrix.dtype)), dim=1)
        result.append(matrix)
    result = pad_sequence(result, batch_first=batch_first, padding_value=padding_value)
    return result


def plot_loss_curve(train_loss, valid_loss, save_path):
    # 创建 x 轴数据，例如为训练的 epoch 数
    epochs = range(1, len(train_loss) + 1)
    # 绘制训练损失曲线
    plt.plot(epochs, train_loss, label='Train Loss')
    # 绘制验证损失曲线
    plt.plot(epochs, valid_loss, label='Validation Loss')
    # 添加图例
    plt.legend()
    # 添加标题和轴标签
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    # 保存图像文件
    plt.savefig(save_path)
    # 显示图像
    # plt.show()
    # 清空图形
    plt.clf()
