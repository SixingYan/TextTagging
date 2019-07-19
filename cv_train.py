import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import random

from model1 import Model, EncoderRNN, AttnDecoderRNN, FocalLoss

import util
from const import tag_to_ix, ix_to_tag, START_TAG, STOP_TAG
import const
import os
import copy

from sklearn.model_selection import train_test_split


def test():
    device = 'cpu'
    teacher_forcing_ratio = 0.3
    trn_X, trn_y, tst, word_to_ix = util.load()
    max_length = max(len(x) for x in trn_X)
    loss_fn = nn.NLLLoss()
    # train
    direction = 2
    en_indim = 100
    en_outdim = 100
    de_indim = en_outdim * direction
    encoder = EncoderRNN(
        len(word_to_ix), en_indim, en_outdim,
        bi=direction, num_layers=2).to(device)
    decoder = AttnDecoderRNN(
        de_indim, len(tag_to_ix), dropout=0.1,
        max_length=max_length).to(device)

    encoder.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'encoder719_0.pytorch')))
    decoder.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'decoder719_0.pytorch')))

    tsttag = []
    encoder.eval()
    decoder.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)

        eoutputs = torch.zeros(
            max_length, encoder.in_hdim * direction, device=device)
        for ei in range(len(x)):
            eoutput, ehidden = encoder(x[ei], encoder.init_hidden())
            ehidden = torch.cat([ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
            eoutputs[ei] += eoutputs[0, 0]

        dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)
        dhidden = ehidden

        y_pred = []
        for di in range(len(x)):
            doutput, dhidden = decoder(
                dinput, dhidden, eoutputs)
            topv, topi = doutput.data.topk(1)
            y_pred.append(topi.item())
            dinput = topi.squeeze().detach()
        tsttag.append([ix_to_tag[ix] for ix in y_pred])

    util.output(tsttag)


def eval(y_preds, y_trues):
    """
    micro 计算单次后做平均
    """
    size = len(y_preds)
    assert len(y_preds) == len(y_trues)
    stat = {'a': [0, 0], 'b': [0, 0], 'c': [0, 0]}
    tags = ['a', 'b', 'c']
    prec = {'a': 0, 'b': 0, 'c': 0}
    recl = {'a': 0, 'b': 0, 'c': 0}
    f1 = {'a': 0, 'b': 0, 'c': 0}
    for y_pred, y_true in zip(y_preds, y_trues):
        prestat, recstat = copy.copy(stat), copy.copy(stat)
        for i, p in enumerate(y_pred):
            t = ix_to_tag[y_true[i]]
            if p in tags:
                prestat[p][1] += 1
            if p in tags and p == t:
                prestat[p][0] += 1
                recstat[t][0] += 1
            if t in tags:
                recstat[t][1] += 1
        for x in tags:
            prec[x] += prestat[x][0] / (prestat[x][1] + 1e-8) / size
            recl[x] += recstat[x][0] / (recstat[x][1] + 1e-8) / size

    for x in tags:
        f1[x] = (2 * prec[x] * recl[x]) / (prec[x] + recl[x] + 1e-8)

    for x in tags:
        print('TAG {} \t prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
            x, prec[x], recl[x], f1[x]))

    print('AVG prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
        sum(prec[x] for x in tags) / 3,
        sum(recl[x] for x in tags) / 3,
        sum(f1[x] for x in tags) / 3))


def main(epoch_num=2, split_num=5):
    ix_to_tag[tag_to_ix[START_TAG]] = 'o'
    ix_to_tag[tag_to_ix[STOP_TAG]] = 'o'

    trn_X, trn_y, tst, word_to_ix = util.load()
    # train
    for sid in range(split_num):
        trn_ixs, vld_ixs = train_test_split(list(range(len(trn_X))),
                                            test_size=1 / split_num,
                                            shuffle=True,
                                            random_state=sid)
        # init
        print('SPLIT {} --------------------------------------------------------'.format(sid + 1))
        model = Model(len(word_to_ix), tag_to_ix, 100, 200)
        optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
        tsize, vsize = len(trn_ixs), len(vld_ixs)
        # epoch train
        for epoch in range(epoch_num):
            # train ++++++++++++++++++++
            model.train()
            avg_loss = 0
            for i in tqdm(np.random.permutation(trn_ixs)):

                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                model.zero_grad()
                optimizer.zero_grad()
                # 因为使用了特殊的模型，所以loss fn 需要自己写
                loss = model.loss_fn(x, y)
                avg_loss += loss.item() / tsize
                loss.backward()
                optimizer.step()

            # eval ++++++++++++++++++++
            model.eval()
            vld_loss = 0
            y_preds, y_trues = [], []
            for i in tqdm(np.random.permutation(vld_ixs)):
                y_p = model(torch.tensor(trn_X[i], dtype=torch.long))
                vld_loss += model.loss_fn(x, y).item() / vsize
                y_preds.append([ix_to_tag[t] for t in y_p])
                y_trues.append(trn_y[i])

            print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
                epoch + 1, epoch_num, avg_loss, vld_loss))
            eval(y_preds, y_trues)
            print()
        util.savemodel(model, 'bigrucrf717_{}.pytorch'.format(sid))


def main1(epoch_num=2, split_num=5):
    device = 'cpu'
    teacher_forcing_ratio = 0.5
    trn_X, trn_y, tst, word_to_ix = util.load()
    max_length = max(len(x) for x in trn_X)
    loss_fn = nn.NLLLoss()
    # train
    for sid in range(split_num):
        trn_ixs, vld_ixs = train_test_split(list(range(len(trn_X))),
                                            test_size=1 / split_num,
                                            shuffle=True,
                                            random_state=sid)
        # init
        print('SPLIT {} --------------------------------------------------------'.format(sid + 1))
        direction = 2
        en_indim = 100
        en_outdim = 100
        de_indim = en_outdim * direction
        encoder = EncoderRNN(len(word_to_ix), en_indim, en_outdim,
                             bi=direction, num_layers=2).to(device)
        decoder = AttnDecoderRNN(
            de_indim, len(tag_to_ix), dropout=0.2, max_length=max_length).to(device)
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01, weight_decay=1e-4)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01, weight_decay=1e-4)

        tsize, vsize = len(trn_ixs), len(vld_ixs)
        # loss_fn = FocalLoss()
        # epoch train
        for epoch in range(epoch_num):
            # train ++++++++++++++++++++
            encoder.train()
            decoder.train()
            avg_loss = 0
            for i in tqdm(np.random.permutation(trn_ixs)):
                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                length = len(trn_X[i])

                eoutputs = torch.zeros(
                    max_length, encoder.in_hdim * direction, device=device)
                loss = 0
                for ei in range(len(x)):
                    eoutput, ehidden = encoder(
                        x[ei], encoder.init_hidden())
                    ehidden = torch.cat([ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
                    eoutputs[ei] = eoutput[0, 0]

                dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)
                dhidden = ehidden

                # doutputs = torch.zeros(
                #    length, decoder.hidden_size, device=device)

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    for di in range(length):
                        doutput, dhidden = decoder(
                            dinput, dhidden, eoutputs)
                        loss += loss_fn(doutput, y[di].unsqueeze(0))
                        dinput = y[di]  # Teacher forcing
                        #doutputs[di] = doutput[0, 0]
                else:
                    for di in range(length):
                        doutput, decoder_hidden = decoder(
                            dinput, dhidden, eoutputs)
                        topv, topi = doutput.topk(1)
                        dinput = topi.squeeze().detach()  # detach from history as input
                        loss += loss_fn(doutput, y[di].unsqueeze(0))
                        #doutputs[di] = doutput[0, 0]
                        if dinput.item() == STOP_TAG:
                            break

                #loss = loss_fn(doutputs, y)

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                avg_loss += loss.item() / tsize

            # eval ++++++++++++++++++++
            encoder.eval()
            decoder.eval()
            vld_loss = 0
            y_preds, y_trues = [], []
            for i in tqdm(np.random.permutation(vld_ixs)):
                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                y_pred, loss = evalone(max_length, encoder, decoder, x, direction=direction, loss_fn=None, y=y)
                if loss != 0:
                    vld_loss += loss.item() / vsize
                y_preds.append([ix_to_tag[t] for t in y_pred])
                y_trues.append(trn_y[i])

            print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
                epoch + 1, epoch_num, avg_loss, vld_loss))
            eval(y_preds, y_trues)
            print()

        util.savemodel(encoder, 'encoder719_{}.pytorch'.format(sid))
        util.savemodel(decoder, 'decoder719_{}.pytorch'.format(sid))


def evalone(max_length, encoder, decoder, x, direction=1,
            loss_fn=None, y=None, device='cpu'):
    eoutputs = torch.zeros(
        max_length, encoder.in_hdim * direction, device=device)
    for ei in range(len(x)):
        eoutput, ehidden = encoder(x[ei], encoder.init_hidden())
        ehidden = torch.cat([ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        eoutputs[ei] += eoutputs[0, 0]

    dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)  # SOS
    dhidden = ehidden
    loss = 0
    y_pred = []
    for di in range(len(x)):
        doutput, dhidden = decoder(
            dinput, dhidden, eoutputs)
        if loss_fn is not None and y is not None:
            loss += loss_fn(doutput, y[di].unsqueeze(0))

        topv, topi = doutput.data.topk(1)
        if topi.item() == STOP_TAG:
            break
        else:
            y_pred.append(topi.item())
        dinput = topi.squeeze().detach()
    return y_pred, loss


if __name__ == '__main__':
    main1()
