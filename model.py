import torch
import torch.nn as nn
from tools.config import config

from dgl.nn.pytorch.gt import GraphormerLayer
from dgl.nn.pytorch.gt import SpatialEncoder
from module.sent_encoder import LSTMEncoder
from torch.nn.utils.rnn import pad_sequence
from module.position_encoding import PositionalEncoding
from transformers import RobertaModel


class GraphExt(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.feat_dim)
        self.layer_norm2 = nn.LayerNorm(config.feat_dim)
        self.wo = nn.Linear(config.feat_dim, 1, bias=True)
        self.io = nn.Linear(config.feat_dim, 1, bias=True)
        self.activation = nn.Sigmoid()
        self.encoder = GraphEncoder()

    def forward(self, data):
        sent_feature, img_feature, word_feature = self.encoder(data)
        sent_score = self.wo(self.layer_norm1(sent_feature)).squeeze(-1)
        img_score = self.io(self.layer_norm2(img_feature)).squeeze(-1)
        return sent_score, self.activation(img_score)


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embed = nn.Embedding(config.vocab_size, config.word_emb_dim, padding_idx=0)

        self.node_type_embed = nn.Embedding(3 + 1, config.feat_dim, padding_idx=0)

        # self.lstm_encoder = LSTMEncoder(config.word_emb_dim, config.lstm_hidden_dim, config.lstm_n_layers,
        #                                 config.feat_dim)
        self.sent_position_encoding = PositionalEncoding(config.feat_dim, max_len=config.sent_max_num)

        self.spatial_embed0_ss = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)
        self.spatial_embed0_sw = SpatialEncoder(max_dist=config.max_dist+1, num_heads=config.g_n_head)
        self.spatial_embed0_ws = SpatialEncoder(max_dist=config.max_dist+1, num_heads=config.g_n_head)
        self.spatial_embed0_ww = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)

        self.spatial_embed1_ss = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)
        self.spatial_embed1_si = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)
        self.spatial_embed1_is = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)
        self.spatial_embed1_ii = SpatialEncoder(max_dist=config.max_dist, num_heads=config.g_n_head)

        self.edge_embed0_ss = nn.Embedding(11, config.g_n_head, padding_idx=0)
        self.edge_embed0_sw = nn.Embedding(11, config.g_n_head, padding_idx=0)
        self.edge_embed0_ws = nn.Embedding(11, config.g_n_head, padding_idx=0)

        self.edge_embed1_ss = nn.Embedding(10, config.g_n_head, padding_idx=0)
        self.edge_embed1_si = nn.Embedding(10, config.g_n_head, padding_idx=0)
        self.edge_embed1_is = nn.Embedding(10, config.g_n_head, padding_idx=0)
        self.edge_embed1_ii = nn.Embedding(10, config.g_n_head, padding_idx=0)

        self.layers0 = nn.ModuleList([
            GraphormerLayer(
                feat_size=config.feat_dim,
                hidden_size=config.g_hidden_dim,
                num_heads=config.g_n_head,
                attn_bias_type="mul",
                norm_first=True,
                dropout=0
            )
            for _ in range(config.g_n_layer)
        ])

        self.layers1 = nn.ModuleList([
            GraphormerLayer(
                feat_size=config.feat_dim,
                hidden_size=config.g_hidden_dim,
                num_heads=config.g_n_head,
                attn_bias_type="mul",
                norm_first=True,
                dropout=0
            )
            for _ in range(config.g_n_layer)
        ])

        self.bert_model = RobertaModel.from_pretrained(config.bert_model_path, add_pooling_layer=False)
        for param in self.bert_model.parameters():
            param.requires_grad = False
        for layer in list(self.bert_model.encoder.layer)[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.sent_feature_proj = nn.Linear(768, config.feat_dim)
        # self.image_feature_proj = nn.Linear(config.image_feature_dim, config.feat_dim)
        # self.word_feature_proj = nn.Linear(config.word_emb_dim, config.feat_dim)

    def forward(self, data):
        sent_feature, img_feature, word_feature = self.get_input(data)

        sent_wide = data['s_label'].size(1)
        attn_bias0 = torch.zeros(list(data['shortest_dist'].shape) + [config.g_n_head], device=config.device)
        attn_bias0[:, :sent_wide, :sent_wide] = self.spatial_embed0_ss(data['shortest_dist'][:, :sent_wide, :sent_wide])
        attn_bias0[:, :sent_wide, sent_wide:] = self.spatial_embed0_sw(data['shortest_dist'][:, :sent_wide, sent_wide:]+1)
        attn_bias0[:, sent_wide:, :sent_wide] = self.spatial_embed0_ws(data['shortest_dist'][:, sent_wide:, :sent_wide]+1)
        attn_bias0[:, sent_wide:, sent_wide:] = self.spatial_embed0_ww(data['shortest_dist'][:, sent_wide:, sent_wide:])

        edge_embed0 = torch.zeros(attn_bias0.shape, device=config.device)
        edge_embed0[:, :sent_wide, :sent_wide] = self.edge_embed0_ss(data['weight_matrix'][:, :sent_wide, :sent_wide])
        edge_embed0[:, :sent_wide, sent_wide:] = self.edge_embed0_sw(data['weight_matrix'][:, :sent_wide, sent_wide:])
        edge_embed0[:, sent_wide:, :sent_wide] = self.edge_embed0_ws(data['weight_matrix'][:, sent_wide:, :sent_wide])
        attn_bias0 += edge_embed0

        attn_bias1 = torch.zeros(list(data['si_shortest_dist'].shape) + [config.g_n_head], device=config.device)
        attn_bias1[:, :sent_wide, :sent_wide] = self.spatial_embed1_ss(data['si_shortest_dist'][:, :sent_wide, :sent_wide])
        attn_bias1[:, :sent_wide, sent_wide:] = self.spatial_embed1_si(data['si_shortest_dist'][:, :sent_wide, sent_wide:])
        attn_bias1[:, sent_wide:, :sent_wide] = self.spatial_embed1_is(data['si_shortest_dist'][:, sent_wide:, :sent_wide])
        attn_bias1[:, sent_wide:, sent_wide:] = self.spatial_embed1_ii(data['si_shortest_dist'][:, sent_wide:, sent_wide:])

        edge_embed1 = torch.zeros(attn_bias1.shape, device=config.device)
        edge_embed1[:, :sent_wide, :sent_wide] = self.edge_embed1_ss(data['si_weight_matrix'][:, :sent_wide, :sent_wide])
        edge_embed1[:, :sent_wide, sent_wide:] = self.edge_embed1_si(data['si_weight_matrix'][:, :sent_wide, sent_wide:])
        edge_embed1[:, sent_wide:, :sent_wide] = self.edge_embed1_is(data['si_weight_matrix'][:, sent_wide:, :sent_wide])
        edge_embed1[:, sent_wide:, sent_wide:] = self.edge_embed1_ii(data['si_weight_matrix'][:, sent_wide:, sent_wide:])
        attn_bias1 += edge_embed1
        attn_mask0 = data.pop('pad_mask')
        attn_mask1 = data.pop('si_pad_mask')

        for i in range(config.g_n_layer):
            sw = torch.cat((sent_feature, word_feature), dim=1)
            sw = self.layers0[i](sw, attn_bias=attn_bias0, attn_mask=attn_mask0)
            sent_feature = sw[:, :sent_wide]
            word_feature = sw[:, sent_wide:]
            si = torch.cat((sent_feature, img_feature), dim=1)
            si = self.layers1[i](si, attn_bias=attn_bias1, attn_mask=attn_mask1)
            sent_feature = si[:, :sent_wide]
            img_feature = si[:, sent_wide:]

        return sent_feature, img_feature, word_feature

    def get_input(self, data):
        # sent_embedding = self.sent_feature_proj(data.pop('b_feature'))
        # sent_embedding = self.get_sent_embedding_by_lstm(data)
        sent_embedding = self.get_sent_embedding_by_bert(data)
        sent_embedding += self.sent_position_encoding(sent_embedding.size(1))
        img_embedding = self.get_img_embedding(data)
        word_embedding = self.get_word_embedding(data)

        sent_embedding += self.node_type_embed(
            torch.tensor([1] * sent_embedding.size(1), dtype=torch.int32, device=config.device))
        img_embedding += self.node_type_embed(
            torch.tensor([2] * img_embedding.size(1), dtype=torch.int32, device=config.device))
        word_embedding += self.node_type_embed(
            torch.tensor([3] * word_embedding.size(1), dtype=torch.int32, device=config.device))

        return sent_embedding, img_embedding, word_embedding

    def get_sent_embedding_by_lstm(self, data):
        sents = data.pop("s_words")
        shape = sents.shape
        sents = sents.view(shape[0] * shape[1], shape[2])
        length = (sents != 0).sum(dim=1).cpu()
        sents = self.word_embed(sents)
        sent_embedding = self.lstm_encoder(sents, length)
        sent_embedding = sent_embedding.view(shape[0], shape[1], config.feat_dim)

        return sent_embedding

    def get_sent_embedding_by_bert(self, data):
        bert_input = data['bert_input']
        result = []
        for encoded_input in bert_input:
            output = self.bert_model(**(encoded_input.to(config.device)))
            sent_embedding = output.last_hidden_state[:, 0]
            result.append(sent_embedding)
        sent_embedding = pad_sequence(result, batch_first=True)
        sent_embedding = self.sent_feature_proj(sent_embedding)
        return sent_embedding

    def get_word_embedding_by_lstm(self, sents):
        length = (sents != 0).sum(dim=1).cpu()
        sents = self.word_embed(sents)
        word_embedding = self.word_lstm_encoder(sents, length)

        return word_embedding

    def get_img_embedding(self, data):
        img_embedding = data.pop("i_feature")
        # img_embedding = self.image_feature_proj(img_embedding)
        return img_embedding

    def get_word_embedding(self, data):
        word_embedding = self.word_embed(data.pop("w_wid"))
        # word_embedding = self.word_feature_proj(word_embedding)
        return word_embedding
