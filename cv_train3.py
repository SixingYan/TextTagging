import util
from const import tag_to_ix, ix_to_tag, START_TAG, STOP_TAG
import const

import torch.nn.functional as F
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random
import os
import pickle
import copy
from sklearn.model_selection import train_test_split


class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim=10, hid_dim=10,
                 loss_fn=nn.NLLLoss(), num_layers=1, dropout=0.1, pre_word_embeds=None,
                 use_subembed=True):
        super(BiLSTM, self).__init__()
        self.hid_dim = hid_dim
        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        if pre_word_embeds is not None:
            self.word_embeds.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.word_embeds.weight.requires_grad = False
        self.use_subembed = use_subembed
        if use_subembed:
            self.sub_embed = nn.Embedding(vocab_size, embed_dim // 2)
            self.subrnn = nn.GRU(embed_dim // 2, hid_dim)
            self.hid_dim += hid_dim
        self.rnn = nn.LSTM(embed_dim, self.hid_dim, bidirectional=True,
                           num_layers=num_layers, dropout=dropout)
        #init_rnn(self.rnn, 'lstm')
        self.hid2tag = nn.Linear(hid_dim * 2, len(tag_to_ix))
        # init_linear(self.hid2tag)
        self.loss_fn = loss_fn

    def _forward(self, x):
        embeds = self.word_embeds(x).view(len(x), 1, -1)
        if self.use_subembed:
            c_embed = self.sub_embed(x)
            sub_out, _ = self.subrnn(c_embed.view(len(x), 1, -1))
            embeds = torch.cat([embeds, sub_out], dim=2)
            
        lstm_out, _ = self.rnn(embeds)
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


def evalinfo(y_preds, y_trues):
    size = len(y_preds)
    assert len(y_preds) == len(y_trues)
    stat = {'a': [0, 0], 'b': [0, 0], 'c': [0, 0]}
    tags = ['a', 'b', 'c']
    prec = {'a': [], 'b': [], 'c': []}
    recl = {'a': [], 'b': [], 'c': []}
    f1 = {'a': 0, 'b': 0, 'c': 0}
    for j in range(len(y_preds)):
        y_pred, y_true = y_preds[j], y_trues[j]
        prestat, recstat = copy.copy(stat), copy.copy(stat)
        for i in range(len(y_pred)):
            t = ix_to_tag[int(y_true[i])]
            p = ix_to_tag[int(y_pred[i])]
            if p in tags:
                prestat[p][1] += 1
            if p in tags and p == t:
                prestat[p][0] += 1
                recstat[t][0] += 1
            if t in tags:
                recstat[t][1] += 1
        for x in tags:
            if recstat[x][1] != 0:
                recl[x].append(recstat[x][0] / recstat[x][1])
            if prestat[x][1] != 0:
                prec[x].append(prestat[x][0] / prestat[x][1])
    for x in tags:
        prec[x] = 0 if len(prec[x]) == 0 else sum(prec[x]) / len(prec[x])
        recl[x] = 0 if len(recl[x]) == 0 else sum(recl[x]) / len(recl[x])
        f1[x] = (2 * prec[x] * recl[x]) / (prec[x] + recl[x] + 1e-8)

    for x in tags:
        print('TAG {} \t prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
            x, prec[x], recl[x], f1[x]))

    print('AVG prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
        sum(prec[x] for x in tags) / 3,
        sum(recl[x] for x in tags) / 3,
        sum(f1[x] for x in tags) / 3))


def test():
    trn_X, trn_y, tst, word_to_ix = util.load()
    # device =
    model = BiLSTM(len(word_to_ix), embed_dim=64, hid_dim=64).to(device)

    model.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'bilstm_4_3.pytorch'), map_location='cpu'))
    tsttag = []
    model.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        y_pred = model(x)
        tsttag.append([ix_to_tag[ix] for ix in y_pred])
    util.output(tsttag, 'bilstm43')


def judge():
    device = 'cpu'
    trn_X, trn_y, tst, word_to_ix = util.load()
    names = ['bilstm_{}_3.pytorch'.format(i) for i in range(5)]
    models = [None for _ in names]
    for i, n in enumerate(names):
        model = BiLSTM(len(word_to_ix), embed_dim=64, hid_dim=64).to(device)
        model.load_state_dict(torch.load(
            os.path.join(const.MODELPATH, n), map_location=device))
        models[i] = model

    for i, model in enumerate(models):
        print('This is ', names[i])
        y_preds, y_trues = [], []
        for i in tqdm(np.random.permutation(len(trn_X))):
            x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
            y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
            y_pred = model(x)
            y_preds.append(y_pred[:])
            y_trues.append(trn_y[i])
        evalinfo(y_preds, y_trues)
        print()


'''
def main(epoch_num=4, split_num=5):
    device = 'cpu'
    teacher_forcing_ratio = 0.3
    trn_X, trn_y, tst, word_to_ix = util.load()
    max_length = max(len(x) for x in trn_X)
    dropout = 0.5
    layers = 2
    direction = 2
    en_indim = 128
    en_outdim = 128
    de_indim = en_outdim * direction

    # train
    for sid in range(split_num):
        trn_ixs, vld_ixs = train_test_split(list(range(len(trn_X))),
                                            test_size=1 / split_num,
                                            shuffle=True,
                                            random_state=sid)
        # init
        print('SPLIT {} --------------------------------------------------------'.format(sid + 1))

        model = EncoderAttDecoder(len(word_to_ix), in_hdim=en_indim, out_hdim=en_outdim,
                                  de_hdim=de_indim, max_length=max_length,
                                  bi=direction, dropout=dropout, num_layers=layers,
                                  teacher_forcing_ratio=teacher_forcing_ratio)

        optimizer = optim.Adam(
            model.parameters(), lr=0.01, weight_decay=1e-6)
        # epoch train
        for epoch in range(epoch_num):
            # train ++++++++++++++++++++
            model.train()
            for i in tqdm(np.random.permutation(trn_ixs)):
                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                optimizer.zero_grad()
                loss = model.loss(x, y)
                loss.backward()
                optimizer.step()

'''
if __name__ == '__main__':
    judge()
