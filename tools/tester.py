import torch
import pickle
import os
from tools.logger import logger
from tools.config import config
from tools.utils import calc_ext_loss
from module.vocabulary import FileVocab, DictVocab


class ExtTester(object):
    def __init__(self, model, m, test_dir=None, limited=False):
        super()
        self.model = model
        self.limited = limited
        self.m = m
        self.test_dir = test_dir
        self.extracts = []

        self.batch_number = 0
        self.running_loss = 0
        self.example_num = 0
        self.total_sentence_num = 0

        self._hyps = []
        self._refer = []

        self.pred, self.true, self.match, self.match_true = 0, 0, 0, 0
        self._F = 0
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

        self.IP = 0
        self.ip_match = 0
        self.ip_count = 0
        self.hash2imgref = {}
        with open(config.datasets_dir / 'hash2imgref.pkl', 'rb') as file:
            self.hash2imgref = pickle.load(file)

    def evaluation(self, data, origin, blocking=False, threshold=float('-inf')):

        self.batch_number += 1
        sent_score, img_score = self.model.forward(data)
        loss = calc_ext_loss(sent_score, data['s_label'], self.criterion)
        self.running_loss += float(loss)

        for i in range(len(sent_score)):
            original_article_sents = origin[i]['article']
            refer = origin[i]['abstract']

            sent_num = data['num_nodes'][i]['s']
            p_sent = sent_score[i][:sent_num].cpu()
            p_sent = torch.sigmoid(p_sent)
            label = data['s_label'][i][:sent_num].cpu()
            if self.m == 0:
                # 无限制预测值个数
                prediction = p_sent.max(1)[1]
                pred_idx = torch.arange(sent_num)[prediction != 0]
            else:
                if blocking:
                    pred_idx = self.ngram_blocking(original_article_sents, p_sent, config.blocking_window,
                                                   min(self.m, sent_num))
                else:
                    topk, pred_idx = torch.topk(p_sent, min(self.m, sent_num))
                    pred_idx = pred_idx[topk > threshold] if len(pred_idx[topk > threshold]) > 0 else pred_idx[:1]

                prediction = torch.zeros(sent_num, device=label.device)
                prediction[pred_idx] = 1
            self.extracts.append(pred_idx.tolist())

            self.pred += prediction.sum()
            self.true += label.sum()

            self.match_true += ((prediction == label) & (prediction == 1)).sum()
            self.match += (prediction == label).sum()
            self.total_sentence_num += sent_num
            self.example_num += 1
            hyps = "\n".join(original_article_sents[id] for id in pred_idx)

            self._hyps.append(hyps)
            self._refer.append(refer)

        for i in range(len(img_score)):
            hash = origin[i]['hash']
            imgref = self.hash2imgref.get(hash, [])
            if len(imgref):
                self.ip_count += 1
                img_num = data['num_nodes'][i]['i']
                topk, pred_idx = torch.topk(img_score[i][:img_num], 1)
                if pred_idx + 1 in imgref:
                    self.ip_match += 1

    def getMetric(self):
        logger.info("Validset match_true %d, pred %d, true %d, total %d, match %d",
                    self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        self._accu, self._precision, self._recall, self._F = eval_label(
            self.match_true, self.pred, self.true, self.total_sentence_num, self.match)
        logger.info(
            "The size of totalset is %d, sent_number is %d, accu is %f, precision is %f, recall is %f, F is %f",
            self.example_num, self.total_sentence_num, self._accu, self._precision, self._recall, self._F)
        if self.ip_count:
            self.IP = self.ip_match / self.ip_count
            logger.info("IP is %f", self.IP)

    def ngram_blocking(self, sents, p_sent, window_size, k):
        """
        :param p_sent: [sent_num, 1]
        :return: 
        """

        def _get_ngrams(n, text):
            ngrams = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngrams.add(tuple(text[i:i + n]))
            return ngrams

        ngram_set = set()
        _, sorted_idx = p_sent.sort(descending=True)
        result = []
        for idx in sorted_idx:
            sent = sents[idx]
            pieces = sent.split()
            sent_ngram = _get_ngrams(window_size, pieces)
            if not ngram_set.intersection(sent_ngram):
                result.append(idx)
                ngram_set = ngram_set.union(sent_ngram)

            if len(result) >= k:
                break

        return torch.LongTensor(result)

    @property
    def labelMetric(self):
        return self._F

    @property
    def running_avg_loss(self):
        return self.running_loss / self.batch_number

    @property
    def rougePairNum(self):
        return len(self._hyps)

    @property
    def hyps(self):
        if self.limited:
            hlist = []
            for i in range(self.rougePairNum):
                k = len(self._refer[i].split(" "))
                lh = " ".join(self._hyps[i].split(" ")[:k])
                hlist.append(lh)
            return hlist
        else:
            return self._hyps

    @property
    def refer(self):
        return self._refer

    @property
    def extractLabel(self):
        return self.extracts

    def SaveDecodeFile(self):
        import datetime
        nowTime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')  # 现在
        log_dir = os.path.join(self.test_dir, nowTime)
        with open(log_dir, "wb") as resfile:
            for i in range(self.rougePairNum):
                resfile.write(b"[Reference]\t")
                resfile.write(self._refer[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"[Hypothesis]\t")
                resfile.write(self._hyps[i].encode('utf-8'))
                resfile.write(b"\n")
                resfile.write(b"\n")
                resfile.write(b"\n")


def eval_label(match_true, pred, true, total, match):
    match_true, pred, true, match = match_true.float(), pred.float(), true.float(), match.float()
    try:
        accu = match / total
        precision = match_true / pred
        recall = match_true / true
        F = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        accu, precision, recall, F = 0.0, 0.0, 0.0, 0.0
        logger.error("float division by zero")
    return accu, precision, recall, F


def batch_tile(batch_tensor, repeat):
    output = []
    for tensor in batch_tensor:
        output.append(tensor.tile((repeat,) + (1,) * len(tensor.shape)))
    return torch.stack(output)


def batch_index_select(batch_tensor, indices):
    batch_size = batch_tensor.size(0)
    beam_size = batch_tensor.size(1)
    batch_offset = torch.arange(batch_size, dtype=torch.long, device=config.device) * beam_size
    output = batch_tensor.view(batch_size * beam_size, -1).index_select(0, (indices+batch_offset.unsqueeze(-1)).view(-1))
    return output.view(batch_tensor.shape)
