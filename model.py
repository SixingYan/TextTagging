from typing import Dict, List
from const import START_TAG, STOP_TAG
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import util


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, long)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    '''
        input: 1 by tarset_size
        output: 一个tensor值
    '''
    max_score = vec[0, argmax(vec)]

    # vec.size()[1] : tarset_size
    # 把这个值拓成 1 by tarset_size
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 实现技巧，减去最大值，然后再去算exp
    # 因为已经减了，max_score那一项为0，所以要在前面加回来
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class Model(nn.Module):
    """BiGRU + CRF"""
    def __init__(self, vocab_size, tag_to_ix: Dict, embed_dim=10, hid_dim=20, layer_num=3):
        super(Model, self).__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tags_size = len(tag_to_ix)
        self.layer_num = layer_num
        
        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        # util.init_embedding(self.word_embeds)

        self.rnn = nn.LSTM(embed_dim, hid_dim // 2,
                           num_layers=self.layer_num, bidirectional=True, dropout=0.1)
        util.init_rnn(self.rnn)

        self.hid2tag = nn.Linear(hid_dim, self.tags_size)
        util.init_linear(self.hid2tag)

        self.transitions = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(self.tags_size, self.tags_size)))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self):
        return (nn.init.xavier_uniform_(torch.empty(2 * self.layer_num, 1, self.hid_dim // 2), gain=nn.init.calculate_gain('relu')),    # h_{t-1} 因为是前后双向，所以是两个神经元 2
                nn.init.xavier_uniform_(torch.empty(2 * self.layer_num, 1, self.hid_dim // 2), gain=nn.init.calculate_gain('relu')))    # c_{t-1}

    def _forward_alg(self, feats):
        '''
            input: len(sentence) by self.tagset_size
            output: 一个tensor值
        '''
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

        hidden = self.init_hidden()

        embeds = self.word_embeds(sentence).view(
            len(sentence), 1, -1)
        lstm_out, _ = self.rnn(embeds, hidden)
        lstm_out = lstm_out.view(len(sentence), self.hid_dim)
        lstm_feats = self.hid2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        '''
            input:
                feats: len(sentence) by self.tagset_size
                tags: 1 by tagset_size tensor
            output: 一个tensor值
        '''
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # 对score进行累加了
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

        return score

    def _viterbi_decode(self, feats):
        '''
            解码，得到预测的序列，以及预测序列的得分
            input: 
                feats: len(sentence) by self.tagset_size
            output:
        '''
        backpointers = []

        # Initialize the viterbi variables in log space
        # 1 by tagset_size
        init_vvars = torch.full((1, self.tags_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            # feat: 1 by self.tagset_size

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