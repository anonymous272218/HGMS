import dgl
import torch
import torch.utils.data
import nltk
import pickle
from nltk.corpus import stopwords
from tools.logger import logger
from tools.utils import read_text, read_jsonl, pad_matrix
from tools.config import config
from torch.nn.utils.rnn import pad_sequence
from tools.oracle_select_new import greedy_selection
from transformers import RobertaTokenizer
from tools import bm25

FILTER_WORD = stopwords.words('english')
punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '\'\'', '\'', '`', '``',
                '-', '--', '|', '/']
FILTER_WORD.extend(punctuations)
FILTER_WORD = set(FILTER_WORD)

bert_tokenizer = RobertaTokenizer.from_pretrained(config.bert_model_path)


def calc_bm25(corpus, word_list):
    bm25_model = bm25.BM25(corpus)
    bm25_scores = []
    for word in word_list:
        scores = bm25_model.get_scores([word])
        bm25_scores.append(scores)
    return bm25_scores


class Example(object):
    def __init__(self, hash_name, article_sents, abstract_sents, vocab, image_features, caption_sents, img2sent_weight,
                 label, img_label):
        self.hash_name = hash_name
        self.sent_lens = []
        self.sent_words = []
        self.graph_sents = []
        self.bert_input = bert_tokenizer(article_sents, return_tensors='pt', max_length=128, truncation=True,
                                         padding=True)

        sents_clipped = []
        self.origin = {
            'hash': hash_name,
            'article': article_sents,
            'abstract': "\n".join(abstract_sents)
        }

        # Process the article
        self.article_tokens = []
        for sent in article_sents:
            tokens = nltk.word_tokenize(sent)[:config.sent_max_len]
            self.article_tokens.append([token.lower() for token in tokens])
            sents_clipped.append(' '.join(tokens))
            self.sent_words.append([vocab.word2id(w.lower()) for w in tokens])

            pos_tags = nltk.pos_tag(tokens)
            tag_blacklist = ['CC', 'DT', 'IN', 'MD', 'PDT', 'POS', 'RP', 'TO', 'UH']
            filtered_sent_words = []

            for word, tag in pos_tags:
                if tag not in tag_blacklist:
                    filtered_sent_words.append(word)
            self.graph_sents.append([word.lower() for word in filtered_sent_words])

        self.abs_ids = []
        # self.abs_ids.append(vocab.word2id('<s>'))
        # for sent in abstract_sents:
        #     self.abs_ids.extend([vocab.word2id(w.lower()) for w in nltk.word_tokenize(sent) + ['<seg>']])
        # self.abs_ids.append(vocab.word2id('</s>'))
        # self.abs_ids = self.abs_ids[:config.abs_max_len]
        # label
        # sent_label = greedy_selection([sent.lower() for sent in sents_clipped], [sent.lower() for sent in abstract_sents], len(article_sents))
        # sent_label_tensor = torch.zeros(len(article_sents), dtype=torch.int64)
        # sent_label = [label for label in sent_label if label < len(article_sents)]
        # sent_label_tensor[sent_label] = 1
        self.sent_label = label
        self.img_label = img_label

        self.image_features = image_features
        self.img2sent_weight = img2sent_weight
        self.caption_sents = caption_sents


class ExampleSet(torch.utils.data.Dataset):

    def __init__(self, datasets_dir, corpus_type, vocab, img_weight_limit):

        self.vocab = vocab
        self.datasets_dir = datasets_dir
        self.stored_blip_feature_dir = config.stored_blip_feature_dir / corpus_type
        self.img_weight_limit = img_weight_limit

        self.example_list = read_jsonl(datasets_dir / f"{corpus_type}.label.jsonl")

        self.filter_words = FILTER_WORD
        self.filter_ids = [vocab.word2id(w.lower()) for w in FILTER_WORD]
        self.filter_ids.append(vocab.word2id("<pad>"))

        logger.info("Loading %s dataset completed, total size is %d", corpus_type, len(self.example_list))

    def get_example(self, index):
        e = self.example_list[index]
        text = e["text"][:config.sent_max_num]
        label = e["label"]
        hash_name = e.get('hash')
        if config.img_max_num != 0:
            with open(self.stored_blip_feature_dir / (hash_name + '.pkl'), 'rb') as file:
                loaded_data = pickle.load(file)
                image_features = loaded_data['image_feature'][:config.img_max_num]
                img2sent_weight = loaded_data['itm_score'][:config.img_max_num, :config.sent_max_num]

            caption_sents = e.get("caption", [])[:config.img_max_num]
            img_label = e['i_label']
        else:
            image_features = torch.empty(0, config.image_feature_dim)
            img2sent_weight = torch.empty(0, len(text))
            caption_sents = []
            img_label = []

        example = Example(hash_name, text, e.get("summary"), self.vocab, image_features,
                          caption_sents, img2sent_weight, label, img_label)
        return example

    def create_siw_graph(self, graph_sents, img2sent_weight):
        # 不同类型节点起点 [句子, 图片, 词]
        s_start = 0
        i_start = s_start + len(graph_sents)
        w_start = i_start + len(img2sent_weight)

        w2s_edge_src_words = []
        w2s_edge_dst = []
        for i, words in enumerate(graph_sents):
            valid_words = [word for word in set(words) if word not in self.filter_words]
            w2s_edge_src_words.extend(valid_words)
            w2s_edge_dst.extend([s_start + i] * len(valid_words))

        word_set_used = set(w2s_edge_src_words)
        word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
        nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}
        w2s_edge_src = [word2nid_map[word] for word in w2s_edge_src_words]

        w2s_indices = (torch.tensor(w2s_edge_src, dtype=torch.int32), torch.tensor(w2s_edge_dst, dtype=torch.int32))
        s2w_indices = w2s_indices[::-1]

        # img to sentence
        i2s_indices = (img2sent_weight > self.img_weight_limit).nonzero(as_tuple=True)
        i2s_weight_seq = img2sent_weight[i2s_indices]
        i2s_indices = (i2s_indices[0].to(torch.int32) + i_start, i2s_indices[1].to(torch.int32))
        s2i_indices = i2s_indices[::-1]

        g = dgl.graph([], idtype=torch.int32, device=torch.device('cpu'))
        g.add_edges(s2i_indices[0], s2i_indices[1])
        g.add_edges(i2s_indices[0], i2s_indices[1])
        g.add_edges(s2w_indices[0], s2w_indices[1])
        g.add_edges(w2s_indices[0], w2s_indices[1])

        # 移除所有度为1的词节点
        w_degrees = g.in_degrees()[w_start:]
        num_remove = (w_degrees == 1).sum()
        node_id = (w_degrees == 1).nonzero() + w_start
        nodes_to_remove = node_id[0:num_remove].view(-1).tolist()
        # remove
        g.remove_nodes(nodes_to_remove)
        word_set_used -= set([nid2word_map[nid] for nid in nodes_to_remove])
        word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
        nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}

        while len(word_set_used) > config.word_max_num:
            num_remove = len(word_set_used) - config.word_max_num
            w_degrees = g.in_degrees()[w_start:]
            min_degree = torch.min(w_degrees, dim=0)[0]
            node_id = (w_degrees == min_degree).nonzero() + w_start
            nodes_to_remove = node_id[0:num_remove].view(-1).tolist()
            # remove
            g.remove_nodes(nodes_to_remove)
            word_set_used -= set([nid2word_map[nid] for nid in nodes_to_remove])
            word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
            nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}

        num_nodes_dict = {'s': len(graph_sents), 'w': len(word_set_used)}

        shortest_dist = dgl.shortest_dist(g)

        wid_set_used = [self.vocab.word2id(w.lower()) for w in word_set_used]

        data_dict = {
            'w_wid': torch.tensor(list(wid_set_used), dtype=torch.int32),
            'in_degrees': g.in_degrees(),
            'shortest_dist': shortest_dist.to(torch.int8),
            'num_nodes': num_nodes_dict,
        }
        return data_dict

    def create_sw_graph(self, graph_sents, article_tokens):
        # 不同类型节点起点 [句子, 图片, 词]
        s_start = 0
        w_start = s_start + len(graph_sents)

        w2s_edge_src_words = []
        w2s_edge_dst = []
        for i, words in enumerate(graph_sents):
            valid_words = [word for word in set(words) if word not in self.filter_words]
            w2s_edge_src_words.extend(valid_words)
            w2s_edge_dst.extend([s_start + i] * len(valid_words))

        word_set_used = set(w2s_edge_src_words)
        word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
        nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}
        w2s_edge_src = [word2nid_map[word] for word in w2s_edge_src_words]

        w2s_indices = (torch.tensor(w2s_edge_src, dtype=torch.int32), torch.tensor(w2s_edge_dst, dtype=torch.int32))
        s2w_indices = w2s_indices[::-1]

        g = dgl.graph([], idtype=torch.int32, device=torch.device('cpu'))
        g.add_edges(s2w_indices[0], s2w_indices[1])
        g.add_edges(w2s_indices[0], w2s_indices[1])

        # 移除所有度为1的词节点
        w_degrees = g.in_degrees()[w_start:]
        num_remove = (w_degrees == 1).sum()
        node_id = (w_degrees == 1).nonzero() + w_start
        nodes_to_remove = node_id[0:num_remove].view(-1).tolist()
        # remove
        g.remove_nodes(nodes_to_remove)
        word_set_used -= set([nid2word_map[nid] for nid in nodes_to_remove])
        word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
        nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}

        while len(word_set_used) > config.word_max_num:
            num_remove = len(word_set_used) - config.word_max_num
            w_degrees = g.in_degrees()[w_start:]
            min_degree = torch.min(w_degrees, dim=0)[0]
            node_id = (w_degrees == min_degree).nonzero() + w_start
            nodes_to_remove = node_id[0:num_remove].view(-1).tolist()
            # remove
            g.remove_nodes(nodes_to_remove)
            word_set_used -= set([nid2word_map[nid] for nid in nodes_to_remove])
            word2nid_map = {word: w_start + index for index, word in enumerate(word_set_used)}
            nid2word_map = {w_start + index: word for index, word in enumerate(word_set_used)}

        num_nodes_dict = {'s': len(graph_sents), 'w': len(word_set_used)}

        shortest_dist = dgl.shortest_dist(g)

        wid_set_used = [self.vocab.word2id(w.lower()) for w in word_set_used]

        if len(word_set_used) > 0:
            word2sent_weight = calc_bm25(article_tokens, list(word_set_used))
            word2sent_weight = torch.tensor(word2sent_weight)
            if torch.max(word2sent_weight) > 0:
                word2sent_weight = word2sent_weight / torch.max(word2sent_weight)

        sent2sent_weight = bm25.get_bm25_weights(article_tokens)
        sent2sent_weight = torch.tensor(sent2sent_weight)
        sent2sent_weight.fill_diagonal_(0)
        if torch.max(sent2sent_weight) > 0:
            sent2sent_weight = sent2sent_weight / torch.max(sent2sent_weight)

        sent_len = len(article_tokens)
        weight_matrix = torch.zeros(shortest_dist.shape, device=shortest_dist.device)
        weight_matrix[:sent_len, :sent_len] = sent2sent_weight
        if len(word_set_used) > 0:
            weight_matrix[:sent_len, sent_len:] = word2sent_weight.transpose(0, 1)
            weight_matrix[sent_len:, :sent_len] = word2sent_weight

        data_dict = {
            'w_wid': torch.tensor(list(wid_set_used), dtype=torch.int32),
            'in_degrees': g.in_degrees(),
            'shortest_dist': shortest_dist.to(torch.int8),
            'weight_matrix': weight_matrix,
            'num_nodes': num_nodes_dict,
        }
        return data_dict

    def create_si_graph(self, img2sent_weight):
        # 不同类型节点起点 [句子, 图片]
        num_sents = img2sent_weight.size(1)
        num_images = img2sent_weight.size(0)
        s_start = 0
        i_start = s_start + num_sents

        # img to sentence
        i2s_indices = (img2sent_weight >= self.img_weight_limit).nonzero(as_tuple=True)
        i2s_weight_seq = img2sent_weight[i2s_indices]
        i2s_indices = (i2s_indices[0].to(torch.int32) + i_start, i2s_indices[1].to(torch.int32))
        s2i_indices = i2s_indices[::-1]

        g = dgl.graph([], idtype=torch.int32, device=torch.device('cpu'),
                      num_nodes=num_sents + num_images)
        g.add_edges(s2i_indices[0], s2i_indices[1], {'weight': i2s_weight_seq})
        g.add_edges(i2s_indices[0], i2s_indices[1], {'weight': i2s_weight_seq})

        path_length, path = dgl.shortest_dist(g, return_paths=True)
        g.add_edges(0, 0, {'weight': torch.ones(1)})  # 用于处理 path 中的-1
        max_dist = len(path[0][0])
        if max_dist == 0:
            weight_matrix = torch.zeros(path_length.shape)
        else:
            weight_path = g.edata['weight'][path.view(-1)].view(-1, max_dist)
            weight_matrix = torch.prod(weight_path, dim=1).view(g.num_nodes(), g.num_nodes())
            weight_matrix[weight_matrix == 1] = 0

        data_dict = {
            'si_in_degrees': g.in_degrees(),
            'si_shortest_dist': path_length.to(torch.int8),
            'si_weight_matrix': weight_matrix,
        }
        return data_dict

    def __getitem__(self, index):
        """
        :param index: int; the index of the example
        :return 
            g: graph for the example
            index: int; the index of the example in the dataset
        """
        item = self.get_example(index)
        hash_name = item.hash_name
        origin = item.origin
        sent_words = item.sent_words
        article_tokens = item.article_tokens
        graph_sents = item.graph_sents
        bert_input = item.bert_input
        sent_label = item.sent_label
        img_label = item.img_label
        image_features = item.image_features
        img2sent_weight = item.img2sent_weight
        caption_sents = item.caption_sents
        abs_ids = item.abs_ids

        # data_dict = self.create_siw_graph(graph_sents, img2sent_weight)
        data_dict = self.create_sw_graph(graph_sents, article_tokens)
        si_dict = self.create_si_graph(img2sent_weight)
        data_dict.update(si_dict)

        data_dict["s_label"] = sent_label
        data_dict["i_label"] = img_label
        data_dict["s_words"] = sent_words
        data_dict["i_feature"] = image_features
        data_dict['num_nodes']['i'] = len(image_features)
        data_dict['abs_ids'] = abs_ids
        data_dict['bert_input'] = bert_input
        return data_dict, origin

    def __len__(self):
        return len(self.example_list)


def graph_collate_fn(batch):
    '''
    :param batch: (G, input_pad)
    :return:
    '''

    b_data, origin = map(list, zip(*batch))
    data = {
        'w_wid': [],
        "s_words": [],
        "s_label": [],
        "i_label": [],
        "i_feature": [],
        'in_degrees': [],
        'shortest_dist': [],
        'weight_matrix': [],
        'num_nodes': [],
        'si_in_degrees': [],
        'si_shortest_dist': [],
        'si_weight_matrix': [],
        "abs_ids": [],
        'bert_input': []
    }

    for d in b_data:
        for k, v in d.items():
            data[k].append(v)

    data['w_wid'] = pad_sequence(data['w_wid'], batch_first=True)

    data["i_feature"] = pad_sequence(data["i_feature"], batch_first=True)

    s_words = []
    for sents in data['s_words']:
        sent_tensors = [torch.tensor(sent, dtype=torch.int32) for sent in sents]
        s_words.append(pad_sequence(sent_tensors, batch_first=True))

    data['s_words'] = pad_matrix(s_words, batch_first=True)

    s_label = []
    for i, labels in enumerate(data['s_label']):
        label_tensor = torch.zeros(data['num_nodes'][i]['s'], dtype=torch.float32)
        label_tensor[labels[:config.m]] = 1
        s_label.append(label_tensor)
    data['s_label'] = pad_sequence(s_label, batch_first=True, padding_value=-100)

    i_label = []
    for i, labels in enumerate(data['i_label']):
        label_tensor = torch.zeros(data['num_nodes'][i]['i'], dtype=torch.float32)
        label_tensor[labels] = 1
        i_label.append(label_tensor)
    data['i_label'] = pad_sequence(i_label, batch_first=True, padding_value=-100)

    abs_ids = [torch.tensor(ids, dtype=torch.int64) for ids in data['abs_ids']]
    data['abs_ids'] = pad_sequence(abs_ids, batch_first=True, padding_value=0)

    # batch_siw_graph(data)
    batch_sw_graph(data)
    batch_si_graph(data)
    batch_memory_mask(data)
    return data, origin


def batch_memory_mask(data):
    batch_size = len(data['num_nodes'])
    s_wide = data['s_words'].size(1)
    i_wide = data['i_feature'].size(1)
    w_wide = data['w_wid'].size(1)
    batch_wide = s_wide + i_wide + w_wide
    s_left = 0
    i_left = data['s_words'].size(1)
    w_left = i_left + data['i_feature'].size(1)

    pad_mask = torch.ones((batch_size, batch_wide), dtype=torch.bool)
    for i, num_nodes in enumerate(data['num_nodes']):
        s_right = s_left + num_nodes['s']
        i_right = i_left + num_nodes['i']
        w_right = w_left + num_nodes['w']
        pad_mask[i, s_left:s_right] = False
        pad_mask[i, i_left:i_right] = False
        pad_mask[i, w_left:w_right] = False

    data['memory_mask'] = pad_mask


def batch_siw_graph(data):
    batch_size = len(data['num_nodes'])
    s_wide = data['s_words'].size(1)
    i_wide = data['i_feature'].size(1)
    w_wide = data['w_wid'].size(1)
    batch_wide = s_wide + i_wide + w_wide
    s_left = 0
    i_left = data['s_words'].size(1)
    w_left = i_left + data['i_feature'].size(1)

    pad_mask = torch.ones((batch_size, batch_wide, batch_wide), dtype=torch.bool)
    for i, num_nodes in enumerate(data['num_nodes']):
        s_right = s_left + num_nodes['s']
        i_right = i_left + num_nodes['i']
        w_right = w_left + num_nodes['w']
        pad_mask[i, :, s_left:s_right] = False
        pad_mask[i, :, i_left:i_right] = False
        pad_mask[i, :, w_left:w_right] = False

    data['pad_mask'] = pad_mask

    in_degree_tensor = torch.zeros((batch_size, batch_wide), dtype=torch.int32)
    # 填充最短路径
    shortest_dist_tensor = torch.full((batch_size, batch_wide, batch_wide), fill_value=-1,
                                      dtype=torch.int32)
    shortest_dist_tensor[:, torch.arange(0, batch_wide), torch.arange(0, batch_wide)] = 0  # 最短路径的对角线赋值为0

    for i in range(batch_size):
        num_nodes = data['num_nodes'][i]
        s_right = s_left + num_nodes['s']
        i_right = i_left + num_nodes['i']
        w_right = w_left + num_nodes['w']
        o_s_start = 0
        o_i_start = num_nodes['s']
        o_w_start = num_nodes['s'] + num_nodes['i']
        o_s_end = num_nodes['s']
        o_i_end = num_nodes['s'] + num_nodes['i']
        o_w_end = num_nodes['s'] + num_nodes['i'] + num_nodes['w']

        in_degree_tensor[i, s_left:s_right] = data['in_degrees'][i][o_s_start:o_s_end]
        in_degree_tensor[i, i_left:i_right] = data['in_degrees'][i][o_i_start:o_i_end]
        in_degree_tensor[i, w_left:w_right] = data['in_degrees'][i][o_w_start:o_w_end]

        for pad_i, o_i in [(shortest_dist_tensor[i], data['shortest_dist'][i])]:
            pad_i[s_left:s_right, s_left:s_right] = o_i[o_s_start:o_s_end, o_s_start:o_s_end]
            pad_i[s_left:s_right, i_left:i_right] = o_i[o_s_start:o_s_end, o_i_start:o_i_end]
            pad_i[s_left:s_right, w_left:w_right] = o_i[o_s_start:o_s_end, o_w_start:o_w_end]

            pad_i[i_left:i_right, s_left:s_right] = o_i[o_i_start:o_i_end, o_s_start:o_s_end]
            pad_i[i_left:i_right, i_left:i_right] = o_i[o_i_start:o_i_end, o_i_start:o_i_end]
            pad_i[i_left:i_right, w_left:w_right] = o_i[o_i_start:o_i_end, o_w_start:o_w_end]

            pad_i[w_left:w_right, s_left:s_right] = o_i[o_w_start:o_w_end, o_s_start:o_s_end]
            pad_i[w_left:w_right, i_left:i_right] = o_i[o_w_start:o_w_end, o_i_start:o_i_end]
            pad_i[w_left:w_right, w_left:w_right] = o_i[o_w_start:o_w_end, o_w_start:o_w_end]

    data['in_degrees'] = in_degree_tensor
    data['shortest_dist'] = shortest_dist_tensor


def batch_sw_graph(data):
    batch_size = len(data['num_nodes'])
    s_wide = data['s_words'].size(1)
    w_wide = data['w_wid'].size(1)
    batch_wide = s_wide + w_wide
    s_left = 0
    w_left = s_left + data['s_words'].size(1)

    pad_mask = torch.ones((batch_size, batch_wide, batch_wide), dtype=torch.bool)
    for i, num_nodes in enumerate(data['num_nodes']):
        s_right = s_left + num_nodes['s']
        w_right = w_left + num_nodes['w']
        pad_mask[i, :, s_left:s_right] = False
        pad_mask[i, :, w_left:w_right] = False

    data['pad_mask'] = pad_mask

    in_degree_tensor = torch.zeros((batch_size, batch_wide), dtype=torch.int32)
    shortest_dist_tensor = torch.full((batch_size, batch_wide, batch_wide), fill_value=-1,
                                      dtype=torch.int32)  # 填充最短路径
    shortest_dist_tensor[:, torch.arange(0, batch_wide), torch.arange(0, batch_wide)] = 0  # 最短路径的对角线赋值为0
    weight_matrix_tensor = torch.full((batch_size, batch_wide, batch_wide), fill_value=0, dtype=torch.int32)
    for i in range(batch_size):
        data['weight_matrix'][i] = (data['weight_matrix'][i] * 10).to(torch.int32)
    for i in range(batch_size):
        num_nodes = data['num_nodes'][i]
        s_right = s_left + num_nodes['s']
        w_right = w_left + num_nodes['w']
        o_s_start = 0
        o_w_start = num_nodes['s']
        o_s_end = num_nodes['s']
        o_w_end = num_nodes['s'] + num_nodes['w']

        in_degree_tensor[i, s_left:s_right] = data['in_degrees'][i][o_s_start:o_s_end]
        in_degree_tensor[i, w_left:w_right] = data['in_degrees'][i][o_w_start:o_w_end]

        for pad_i, o_i in [(shortest_dist_tensor[i], data['shortest_dist'][i]),
                           (weight_matrix_tensor[i], data['weight_matrix'][i])]:
            pad_i[s_left:s_right, s_left:s_right] = o_i[o_s_start:o_s_end, o_s_start:o_s_end]
            pad_i[s_left:s_right, w_left:w_right] = o_i[o_s_start:o_s_end, o_w_start:o_w_end]

            pad_i[w_left:w_right, s_left:s_right] = o_i[o_w_start:o_w_end, o_s_start:o_s_end]
            pad_i[w_left:w_right, w_left:w_right] = o_i[o_w_start:o_w_end, o_w_start:o_w_end]

    data['in_degrees'] = in_degree_tensor
    data['shortest_dist'] = shortest_dist_tensor
    data['weight_matrix'] = weight_matrix_tensor


def batch_si_graph(data):
    batch_size = len(data['num_nodes'])
    s_wide = data['s_words'].size(1)
    i_wide = data['i_feature'].size(1)
    batch_wide = s_wide + i_wide
    s_left = 0
    i_left = data['s_words'].size(1)

    pad_mask = torch.ones((batch_size, batch_wide, batch_wide), dtype=torch.bool)
    for i, num_nodes in enumerate(data['num_nodes']):
        s_right = s_left + num_nodes['s']
        i_right = i_left + num_nodes['i']
        pad_mask[i, :, s_left:s_right] = False
        pad_mask[i, :, i_left:i_right] = False

    data['si_pad_mask'] = pad_mask

    in_degree_tensor = torch.zeros((batch_size, batch_wide), dtype=torch.int32)
    # 填充最短路径
    shortest_dist_tensor = torch.full((batch_size, batch_wide, batch_wide), fill_value=-1,
                                      dtype=torch.int32)
    shortest_dist_tensor[:, torch.arange(0, batch_wide), torch.arange(0, batch_wide)] = 0  # 最短路径的对角线赋值为0
    weight_matrix_tensor = torch.full((batch_size, batch_wide, batch_wide), fill_value=0, dtype=torch.int32)
    for i in range(batch_size):
        data['si_weight_matrix'][i] = (data['si_weight_matrix'][i] * 10).to(torch.int32)
    for i in range(batch_size):
        num_nodes = data['num_nodes'][i]
        s_right = s_left + num_nodes['s']
        i_right = i_left + num_nodes['i']
        o_s_start = 0
        o_i_start = num_nodes['s']
        o_s_end = num_nodes['s']
        o_i_end = num_nodes['s'] + num_nodes['i']

        in_degree_tensor[i, s_left:s_right] = data['si_in_degrees'][i][o_s_start:o_s_end]
        in_degree_tensor[i, i_left:i_right] = data['si_in_degrees'][i][o_i_start:o_i_end]

        for pad_i, o_i in [(shortest_dist_tensor[i], data['si_shortest_dist'][i]),
                           (weight_matrix_tensor[i], data['si_weight_matrix'][i])]:
            pad_i[s_left:s_right, s_left:s_right] = o_i[o_s_start:o_s_end, o_s_start:o_s_end]
            pad_i[s_left:s_right, i_left:i_right] = o_i[o_s_start:o_s_end, o_i_start:o_i_end]

            pad_i[i_left:i_right, s_left:s_right] = o_i[o_i_start:o_i_end, o_s_start:o_s_end]
            pad_i[i_left:i_right, i_left:i_right] = o_i[o_i_start:o_i_end, o_i_start:o_i_end]

    data['si_in_degrees'] = in_degree_tensor
    data['si_shortest_dist'] = shortest_dist_tensor
    data['si_weight_matrix'] = weight_matrix_tensor
