from typing import Dict, List
from const import START_TAG, STOP_TAG
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import util


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Model(nn.Module):
    """BiGRU + CRF"""

    def __init__(self, vocab_size, tag_to_ix: Dict, embed_dim=100, hid_dim=100, layer_num=3, cnn_dim=100):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tags_size = len(tag_to_ix)
        self.layer_num = layer_num

        # 这里将来用预训练的embedding
        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        # util.init_embedding(self.word_embeds)

        # 多一层embed
        self.cnn_dim = cnn_dim
        kernel_num = 1
        self.char_cnn = nn.Conv2d(in_channels=1, out_channels=cnn_dim, kernel_size=(kernel_num, embed_dim), padding=(2, 0))

        # 可以把cnn的输出作为一个embed来用
        self.rnn = nn.LSTM(embed_dim + cnn_dim,
                           hid_dim // 2,
                           num_layers=self.layer_num,
                           bidirectional=True, dropout=0.3)
        util.init_rnn(self.rnn)

        self.hid2tag = nn.Linear(hid_dim, self.tags_size)
        util.init_linear(self.hid2tag)

        self.transitions = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.tags_size, self.tags_size)))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.dropout = nn.Dropout(0.3)

    def init_hidden(self):
        return (nn.init.xavier_uniform_(torch.empty(2 * self.layer_num, 1, self.hid_dim // 2), gain=nn.init.calculate_gain('relu')),    # h_{t-1} 因为是前后双向，所以是两个神经元 2
                nn.init.xavier_uniform_(torch.empty(2 * self.layer_num, 1, self.hid_dim // 2), gain=nn.init.calculate_gain('relu')))    # c_{t-1}

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tags_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tags_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tags_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        cnn_out = self.char_cnn3(self.char_embeds(sentence).unsqueeze(1))
        cnn_embeds = nn.functional.max_pool2d(cnn_out,
                                              kernel_size=(cnn_out.size(2), 1)
                                              ).view(cnn_out.size(0),
                                                     self.cnn_dim)

        embeds = self.word_embeds(sentence).view(
            len(sentence), -1)

        embeds = torch.cat((embeds, chars_embeds), 1)
        embeds = embeds.unsqueeze(1)  # 在这里加上一维

        embeds = self.dropout(embeds)

        lstm_out, _ = self.rnn(embeds, self.init_hidden())
        lstm_out = lstm_out.view(len(sentence), self.hid_dim)
        lstm_feats = self.hid2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tags_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step
            for next_tag in range(self.tags_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                # 这里求出的是，如果next_tag 为某个标签的时候，当前最佳标签是哪个
                bptrs_t.append(best_tag_id)

                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        # 最后backpointers的size为 len(sentences) by tagset_size

        # Transition to STOP_TAG
        terminal_var = forward_var + \
            self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签到STOP_TAG的转移概率
        # 最后一定是转移到STOP_TAG

        # 最后一个词的最佳tag标签
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            # 后一个最佳tag是best_tag_id，那么当前这个best_tag_id 是谁
            best_tag_id = bptrs_t[best_tag_id]
            # 把这个加入最佳路径
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def focal_loss(self, sentence, tags, gamma=2):
        nil = self.neg_log_likelihood(sentence, tags)
        return (1 - exp(-nil))**gamma * nil

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq  # score,
