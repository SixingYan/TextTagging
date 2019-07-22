import torch.nn.functional as F
import torch
from torch import nn

from typing import Dict


class BiLSTM(nn.Module):

    def __init__(self, vocab_size: int, embed_dim=128, hid_dim=128,
                 loss_fn=nn.NLLLoss(), num_layers=2,
                 use_dropout=False, dropout=0.1,
                 pre_word_embeds=None,
                 use_subembed=True, subembed_dim=64, subhid_dim=64,
                 use_init=False, init_fns: Dict):
        super(BiLSTM, self).__init__()

        self.use_dropout = use_dropout
        if use_dropout is True:
            self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        if pre_word_embeds is not None:
            self.embedding.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.embedding.weight.requires_grad = False
        elif use_init:
            init_fns['embed'](self.embedding)

        self.use_subembed = use_subembed
        if use_subembed:
            self.subembedding = nn.Embedding(vocab_size, subembed_dim)
            self.subrnn = nn.GRU(subembed_dim, subhid_dim)
            embed_dim += subhid_dim
            if use_init:
                init_fns['embed'](self.subembedding)

        self.rnn = nn.LSTM(embed_dim, hid_dim,
                           bidirectional=True,
                           num_layers=num_layers)
        if use_init:
            init_fns['rnn'](self.rnn, 'lstm')

        self.hid2tag = nn.Linear(hid_dim * 2, len(tag_to_ix))
        if use_init:
            init_fns['linear'](self.hid2tag)

        self.loss_fn = loss_fn

    def _forward(self, x):
        embeds = self.embedding(x).view(len(x), 1, -1)
        if self.use_subembed:
            _embed = self.subembedding(x)
            sub_out, _ = self.subrnn(_embed.view(len(x), 1, -1))
            embeds = torch.cat([embeds, sub_out], dim=2)
        if self.use_dropout:
            self.dropout(embeds)

        lstm_out, _ = self.rnn(embeds)
        if self.use_dropout:
            self.dropout(lstm_out)

        tag_space = self.hid2tag(lstm_out.view(len(x), -1))
        output = F.log_softmax(tag_space, dim=1)
        return output

    def forward(self, x):
        output = self._forward(x)
        tag_seq = torch.argmax(output, dim=1).cpu().numpy()
        return tag_seq

    def loss(self, x, y):
        output = self._forward(x)
        return self.loss_fn(output, y)
