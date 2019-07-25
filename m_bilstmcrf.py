import torch
from torch import nn

from const import tag_to_ix, START_TAG, STOP_TAG
from typing import Dict


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, embed_dim=10, hid_dim=20, layer_num=2, device='cpu',
                 pretrain_embed=None,
                 use_dropout=False, dropout=0.1,
                 use_subembed=True, subembed_dim=64, subhid_dim=64,
                 use_init=False, init_fns: Dict=None):
        super(BiLSTMCRF, self).__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.tags_size = len(tag_to_ix)
        self.layer_num = layer_num
        self.device = device
        

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pretrain_embed is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pretrain_embed))
            self.embedding.weight.requires_grad = False
        elif use_init:
            init_fns['embed'](self.embedding)

        self.use_subembed = use_subembed
        if use_subembed:
            self.sub_embed = nn.Embedding(vocab_size, subembed_dim)
            self.subnn = nn.GRU(subembed_dim, subhid_dim)
            embed_dim += subhid_dim

        self.rnn = nn.LSTM(embed_dim, hid_dim,
                           num_layers=self.layer_num,
                           bidirectional=True, dropout=dropout)
        if use_init:
            init_fns['rnn'](self.rnn, 'lstm')

        self.hid2tag = nn.Linear(hid_dim * 2, self.tags_size)
        if use_init:
            init_fns['linear'](self.hid2tag)

        self.transitions = nn.Parameter(
            torch.randn(self.tags_size, self.tags_size, device=device))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self, batch_size=1, init_fn=nn.init.xavier_uniform_):
        return (init_fn(torch.empty(2 * self.layer_num, batch_size, self.hid_dim, device=self.device)),
                init_fn(torch.empty(2 * self.layer_num, batch_size, self.hid_dim, device=self.device)))

    def _forward_alg(self, feats):
        init_alphas = torch.full(
            (1, self.tags_size), -10000., device=self.device)
        init_alphas[0][tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tags_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tags_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, x):
        hidden = self.init_hidden()
        embeds = self.embedding(x).view(
            len(x), 1, -1)

        if self.use_subembed:
            c_embed = self.sub_embed(x)
            sub_out, _ = self.subnn(c_embed.view(len(x), 1, -1))
            embeds = torch.cat([embeds, sub_out], dim=2)

        if self.use_dropout:
            self.dropout(embeds)

        lstm_out, _ = self.rnn(embeds, hidden)
        if self.use_dropout:
            self.dropout(lstm_out)

        lstm_out = lstm_out.view(len(x), self.hid_dim * 2)
        lstm_feats = self.hid2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=self.device)
        tags = torch.cat(
            [torch.tensor([tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tags_size), -
                                10000., device=self.device)
        init_vvars[0][tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tags_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + \
            self.transitions[tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return best_path

    def loss(self, x, y):
        feats = self._get_lstm_features(x)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, y)
        return forward_score - gold_score

    def forward(self, x):  # dont confuse this with _forward_alg above.
        lstm_feats = self._get_lstm_features(x)
        tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq
