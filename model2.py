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
    direction = 1
    en_indim = 128
    en_outdim = 128
    pretrianed_embedding = load_pretrained('cbow', word_to_ix)
    trn_ixs, vld_ixs = train_test_split(
        list(range(len(trn_X))), test_size=1 / 2,
        shuffle=True, random_state=10)
    model = BiLSTM(len(word_to_ix),pre_word_embeds=pretrianed_embedding)
    '''
    model = EncoderAttDecoder(len(word_to_ix), in_hdim=en_indim, out_hdim=en_outdim,
                              de_hdim=en_outdim * direction, max_length=max_length,
                              bi=direction, dropout=0.3, num_layers=1,
                              teacher_forcing_ratio=0.3, use_crf=False, device=device,
                              pre_word_embeds=pretrianed_embedding).to(device)
    '''
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


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


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

class BiLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hid_dim=128,
                 loss_fn=nn.NLLLoss(), num_layers=2, dropout=0.1, pre_word_embeds=None,
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
            embed_dim += hid_dim
        self.rnn = nn.LSTM(embed_dim, hid_dim, bidirectional=True,
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

class EncoderAttDecoder(nn.Module):

    def __init__(self, vocab_size, in_hdim=128, out_hdim=128, de_hdim: int=128, loss_fn=nn.NLLLoss(),
                 use_crf=True, max_length=1000,
                 bi=2, device='cpu', dropout=0.4, num_layers=2,
                 teacher_forcing_ratio=0.3, pre_word_embeds=None):
        super(EncoderAttDecoder, self).__init__()
        self.bi = bi
        self.device = device
        self.max_length = max_length
        self.loss_fn = loss_fn if use_crf is False else None
        self.num_layers = num_layers
        self.en_in_hdim = in_hdim
        self.en_out_hdim = out_hdim
        self.tags_size = len(tag_to_ix)
        self.de_hdim = de_hdim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_crf = use_crf
        self.dropout = nn.Dropout(dropout)

        self.en_wrd_embed = nn.Embedding(vocab_size, self.en_in_hdim)
        if pre_word_embeds is not None:
            self.en_wrd_embed.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.en_wrd_embed.weight.requires_grad = False

        self.enrnn = nn.GRU(self.en_in_hdim, self.en_out_hdim, num_layers=num_layers,
                            bidirectional=True if bi == 2 else False,)
        #init_rnn(self.enrnn, 'GRU')
        self.de_embed = nn.Embedding(self.tags_size, self.de_hdim)
        # init_embedding(self.de_embed)
        self.attn = nn.Linear(self.de_hdim * 2, self.max_length)
        # init_linear(self.attn)
        self.attn_combine = nn.Linear(self.de_hdim * 2, self.de_hdim)
        # init_linear(self.attn_combine)
        self.dernn = nn.GRU(self.de_hdim, self.de_hdim)  # , dropout=0.3)
        #init_rnn(self.dernn, 'GRU')
        self.hid2tag = nn.Linear(self.de_hdim, self.tags_size)
        # init_linear(self.hid2tag)
        self.transitions = nn.Parameter(torch.nn.init.uniform_(
            torch.empty(self.tags_size, self.tags_size, device=self.device)))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _encoder(self, x):
        eoutputs = torch.zeros(
            self.max_length, self.en_out_hdim * self.bi, device=self.device)
        ehidden = nn.init.xavier_uniform_(
            torch.zeros(self.bi * self.num_layers, 1,
                        self.en_out_hdim, device=self.device),
            gain=nn.init.calculate_gain('relu'))
        for ei in range(x.size(0)):
            eoutput, ehidden = self._encoder_net(
                x[ei], ehidden)
            eoutputs[ei] = eoutput[0, 0]

        return eoutputs, ehidden

    def _encoder_net(self, sentence, hidden):
        embed = self.en_wrd_embed(sentence).view(1, 1, -1)
        output, hidden = self.enrnn(self.dropout(embed), hidden)
        output = self.dropout(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden

    def _decoder(self, eoutputs, ehidden, y=None):
        dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=self.device)
        if self.bi == 2:
            dhidden = torch.cat(
                [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        else:
            dhidden = ehidden

        doutputs = torch.zeros(
            self.length, self.de_hdim, device=self.device)
        if y is not None:
            for di in range(self.length):
                doutput, dhidden = self._decoder_net(
                    dinput, dhidden, eoutputs)
                dinput = y[di]  # Teacher forcing
                doutputs[di] = doutput[0, 0]
        else:
            for di in range(self.length):
                doutput, decoder_hidden = self._decoder_net(
                    dinput, dhidden, eoutputs)
                topv, topi = doutput.topk(1)
                dinput = topi.squeeze().detach()
                doutputs[di] = doutput[0, 0]
                if dinput.item() == STOP_TAG:
                    break
        return doutputs

    def _decoder_net(self, input, hidden, encoder_outputs):
        embedded = self.de_embed(input).view(1, 1, -1)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.dernn(output, hidden)
        output = self.dropout(output)
        output = F.log_softmax(self.hid2tag(output[0]), dim=1)
        return output, hidden

    def _crf(self, feats):
        return self._viterbi_decode(feats)

    def _forward_alg(self, feats):
        init_alphas = torch.full(
            (1, self.tags_size), -10000., device=self.device)
        init_alphas[0][tag_to_ix[START_TAG]] = 0.
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
        terminal_var = forward_var + self.transitions[tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=self.device)
        tags = torch.cat(
            [torch.tensor([tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

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
        assert start == tag_to_ix[START_TAG]
        best_path.reverse()
        return best_path

    def _get_feat(self, x, y=None, use_tf=False):
        self.length = x.size(0)
        eoutputs, ehidden = self._encoder(x)
        if use_tf and random.random() < self.teacher_forcing_ratio:
            doutputs = self._decoder(eoutputs, ehidden, y)
        else:
            doutputs = self._decoder(eoutputs, ehidden)
        output = self.hid2tag(doutputs)
        return output

    def _nll_loss(self, output, y):
        loss = 0
        for i in range(y.size(0)):
            loss += self.loss_fn(output[i].unsqueeze(0), y[i].unsqueeze(0))
        return loss

    def loss(self, x, y):
        output = self._get_feat(x, y, use_tf=True)
        if self.use_crf:
            return self._neg_log_likelihood(output, y)
        else:
            return self._nll_loss(output, y)

    def forward(self, x):
        output = self._get_feat(x)
        if self.use_crf:
            tag_seq = self._crf(output)
        else:
            tag_seq = torch.argmax(output, dim=1).cpu().numpy()
        return tag_seq


if __name__ == '__main__':
    main()
