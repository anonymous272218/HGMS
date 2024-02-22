import torch
import torch.nn as nn
from tools.config import config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, invert_permutation
import torch.nn.functional as F
from dgl.nn.pytorch.gt import GraphormerLayer
from dgl.nn.pytorch.gt import SpatialEncoder
# from tools.syntax import label2id


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

        # 定义输出层，用于生成句子特征
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 乘以2因为是双向的，需要拼接两个方向的输出

    def forward(self, sents, sent_length):
        # 初始化隐藏状态和细胞状态
        sorted_len, sorted_indices = torch.sort(sent_length, descending=True)
        sorted_indices = sorted_indices.to(config.device)
        unsorted_indices = invert_permutation(sorted_indices)
        sorted_sents = sents.index_select(dim=0, index=sorted_indices)
        origin_batch_size = sents.size(0)
        batch_size = (sorted_len != 0).sum()
        parcel = pack_padded_sequence(sorted_sents[:batch_size], sorted_len[:batch_size], batch_first=True)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=config.device)  # *2因为是双向
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=config.device)

        # 前向传播
        out, _ = self.lstm(parcel, (h0, c0))  # LSTM输出为(out, (hn, cn))，我们只取out

        unpacked, unpacked_len = pad_packed_sequence(out, batch_first=True)
        lstm_embedding = unpacked[torch.arange(unpacked.size(0)), unpacked_len - 1]
        # 获取句子特征
        sentence_feature = self.fc(lstm_embedding)
        sentence_feature = F.pad(sentence_feature, (0, 0, 0, origin_batch_size - batch_size))
        sentence_feature_unsorted = sentence_feature.index_select(dim=0, index=unsorted_indices)

        return sentence_feature_unsorted


class WordLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(WordLSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size, bias=False)

    def forward(self, sents, sent_length):
        sorted_len, sorted_indices = torch.sort(sent_length, descending=True)
        sorted_indices = sorted_indices.to(config.device)
        unsorted_indices = invert_permutation(sorted_indices)
        sorted_sents = sents.index_select(dim=0, index=sorted_indices)
        origin_batch_size = sents.size(0)
        batch_size = (sorted_len != 0).sum()
        parcel = pack_padded_sequence(sorted_sents[:batch_size], sorted_len[:batch_size], batch_first=True)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=config.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=config.device)

        out, _ = self.lstm(parcel, (h0, c0))

        lstm_embedding, unpacked_len = pad_packed_sequence(out, batch_first=True)

        word_feature = self.fc(lstm_embedding)
        word_feature = F.pad(word_feature, (0, 0, 0, 0, 0, origin_batch_size - batch_size))
        word_feature_unsorted = word_feature.index_select(dim=0, index=unsorted_indices)

        return word_feature_unsorted


