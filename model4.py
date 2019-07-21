import pprint
import util

import fasttext.FastText as fasttext
import torch.nn.functional as F
import torch
from torch import nn, optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random
import os
import copy

local = True
if local:
    DATAPATH = '/Users/alfonso/workplace/datagrad/Data'
    MODELPATH = '/Users/alfonso/workplace/datagrad/Model'

else:
    PATH = '/content/drive/'
    DATAPATH = os.path.join(PATH, 'Colab/datagrad/data/')
    MODELPATH = os.path.join(PATH, 'Colab/datagrad/model/')

START_TAG = "<START>"
STOP_TAG = "<STOP>"
tag_to_ix = {"a": 0, "b": 1, "c": 2, "o": 3,
             START_TAG: 4, STOP_TAG: 5}
ix_to_tag = {tp[1]: tp[0] for tp in tag_to_ix.items()}
ix_to_tag[5] = "o"
ix_to_tag[4] = "o"

trn_X, trn_y, tst, word_to_ix = util.load()
max_length = max(len(x) for x in trn_X)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained(name, word_to_ix, embed_size=128):
    model = fasttext.load_model(os.path.join(
        MODELPATH, '{}.bin'.format(name)))
    embedding_matrix = np.zeros((len(word_to_ix), embed_size))
    for w, ix in word_to_ix.items():
        embedding_matrix[ix] = model[w]
    return embedding_matrix


def main():
    epoch_num = 1
    pretrianed_embedding = load_pretrained('cbow', word_to_ix)
    trn_ixs, vld_ixs = train_test_split(
        list(range(len(trn_X))), test_size=1 / 2,
        shuffle=True, random_state=10)
    model = NeuralNet(
        len(word_to_ix),
        loss_fn=nn.BCEWithLogitsLoss(reduction="sum"),
        pre_word_embeds=pretrianed_embedding).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=0.01, weight_decay=1e-6)
    for epoch in range(epoch_num):
        model.train()
        model, avg_loss = train(trn_ixs, model, optimizer)
        model.eval()
        vld_loss, y_preds, y_trues = evaluate(vld_ixs, model)
        print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
            epoch + 1, epoch_num, avg_loss, vld_loss))
        evalinfo(y_preds, y_trues)


def evalinfo(y_preds, y_trues):
    print('****************************')
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


def train(trn_ixs, model, optimizer):
    global device
    avg_loss = 0
    tsize = len(trn_ixs)
    for i in tqdm(np.random.permutation(trn_ixs)):
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = model.loss(x, y)
        avg_loss += loss.item() / tsize
        loss.backward()
        optimizer.step()
    return model, avg_loss


def evaluate(vld_ixs, model):
    global device
    vld_loss = 0
    vsize = len(vld_ixs)
    y_preds, y_trues = [], []
    for i in tqdm(np.random.permutation(vld_ixs)):
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        y_pred = model(x)
        loss = model.loss(x, y)
        vld_loss += loss.item() / vsize
        y_preds.append(y_pred[:])
        y_trues.append(trn_y[i])
    return vld_loss, y_preds, y_trues


class Attention(nn.Module):

    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        eij = torch.mm(
            x.contiguous().view(-1, feature_dim),
            self.weight
        ).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class NeuralNet(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hid_dim=128, linear_dim=256,
                 pre_word_embeds=None, dropout=0.2, max_len=556, tag_size=4, loss_fn=nn.NLLLoss()):
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight = nn.Parameter(
            torch.tensor(pre_word_embeds, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(dropout)
        self.lstm = nn.LSTM(embed_dim, hid_dim,
                            bidirectional=True, batch_first=True)
        self.gru = nn.GRU(hid_dim * 2, hid_dim,
                          bidirectional=True, batch_first=True)
        self.lstm_attention = Attention(hid_dim * 2, max_len)
        self.gru_attention = Attention(hid_dim * 2, max_len)
        self.linear = nn.Linear(hid_dim * 4 * 2, linear_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.hid2tag = nn.Linear(linear_dim, tag_size)

    def _forward(self, x):
        h_embedding = self.embedding(x.unsqueeze(1))
        #h_embedding = torch.squeeze(
        #    self.embedding_dropout(torch.unsqueeze(h_embedding, 0)))

        h_lstm, _ = self.lstm(h_embedding)
        h_gru, _ = self.gru(h_lstm)
        h_lstm_atten = self.lstm_attention(h_lstm)
        h_gru_atten = self.gru_attention(h_gru)
        avg_pool = torch.mean(h_gru, 1)
        max_pool, _ = torch.max(h_gru, 1)
        conc = torch.cat((h_lstm_atten, h_gru_atten, avg_pool, max_pool), 1)
        conc = self.relu(self.linear(conc))
        conc = self.dropout(conc)
        out = self.hid2tag(conc)
        return out

    def loss(self, x, y):
        output = self._forward(x)
        return self.loss_fn(out, y)

    def forward(self, x):
        output = self._forward(x)
        tag_seq = torch.argmax(output, dim=1).cpu().numpy()
        return tag_seq

if __name__ == '__main__':
    main()
