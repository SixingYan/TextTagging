#from cv_train import eval
from model1 import EncoderRNN, AttnDecoderRNN, CRF

import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
import numpy as np
import random

import util
from const import tag_to_ix, ix_to_tag, START_TAG, STOP_TAG
import const
import os
import copy

from sklearn.model_selection import train_test_split


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


def testing(encoder, decoder, crf, tst, max_length, direction, device='cpu', name='',ngram_size=20000):
    tsttag = []
    encoder.eval()
    decoder.eval()
    crf.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        ngram = [hash((0, tst[i][0])) % ngram_size] + [hash((tst[i][j], tst[i][j + 1])) %
                                                       ngram_size for j in range(len(tst[i]) - 1)]
        g = torch.tensor(ngram, dtype=torch.long)

        eoutputs = torch.zeros(
            max_length, encoder.out_hdim * direction, device=device)
        ehidden = encoder.initHidden()
        for ei in range(len(x)):
            eoutput, ehidden = encoder(x[ei], g[ei], ehidden)
            eoutputs[ei] += eoutputs[0, 0]

        dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)
        dhidden = torch.cat(
            [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        doutputs = torch.zeros(
            len(x), decoder.hidden_size, device=device)

        y_pred = []
        for di in range(len(x)):
            doutput, dhidden = decoder(
                dinput, dhidden, eoutputs)
            topv, topi = doutput.data.topk(1)
            doutputs[di] = doutput[0, 0]
            dinput = topi.squeeze().detach()

        y_pred = crf(doutputs)

        tsttag.append([ix_to_tag[ix] for ix in y_pred])

    util.output(tsttag, name)


def test():
    device = 'cpu'
    ngram_size=20000
    teacher_forcing_ratio = 0.3
    trn_X, trn_y, tst, word_to_ix = util.load()
    max_length = max(len(x) for x in trn_X)
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
    crf = CRF(decoder.hidden_size, tag_to_ix)

    encoder.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'encoder-720_0.pytorch')))
    decoder.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'attdecoder-720_0.pytorch')))
    crf.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'crf-720_0.pytorch')))

    tsttag = []
    encoder.eval()
    decoder.eval()
    crf.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        ngram = [hash((0, tst[i][0])) % ngram_size] + [hash((tst[i][j], tst[i][j + 1])) %
                                                       ngram_size for j in range(len(tst[i]) - 1)]
        g = torch.tensor(ngram, dtype=torch.long)

        eoutputs = torch.zeros(
            max_length, encoder.out_hdim * direction, device=device)
        ehidden = encoder.initHidden()
        for ei in range(len(x)):
            eoutput, ehidden = encoder(x[ei], g[ei], ehidden)
            eoutputs[ei] += eoutputs[0, 0]

        dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)
        dhidden = torch.cat(
            [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        doutputs = torch.zeros(
            len(x), decoder.hidden_size, device=device)

        y_pred = []
        for di in range(len(x)):
            doutput, dhidden = decoder(
                dinput, dhidden, eoutputs)
            topv, topi = doutput.data.topk(1)
            doutputs[di] = doutput[0, 0]
            dinput = topi.squeeze().detach()

        y_pred = crf(doutputs)

        tsttag.append([ix_to_tag[ix] for ix in y_pred])

    util.output(tsttag)


def evalone(max_length, encoder, decoder, crf, x, direction=1, y=None, device='cpu', ngram_size=20000):

    ngram = [hash((0, x[0])) % ngram_size] + [hash((x[j], x[j + 1])) %
                                              ngram_size for j in range(len(x) - 1)]
    g = torch.tensor(ngram, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    eoutputs = torch.zeros(
        max_length, encoder.out_hdim * direction, device=device)
    ehidden = encoder.initHidden()

    for ei in range(len(x)):
        eoutput, ehidden = encoder(x[ei], g[ei], ehidden)
        eoutputs[ei] += eoutputs[0, 0]

    dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)  # SOS
    dhidden = torch.cat(
        [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
    doutputs = torch.zeros(
        len(x), decoder.hidden_size, device=device)

    loss = 0
    y_pred = []
    for di in range(len(x)):
        doutput, dhidden = decoder(
            dinput, dhidden, eoutputs)
        doutputs[di] = doutput[0, 0]
        topv, topi = doutput.topk(1)
        dinput = topi.squeeze().detach()
    y_pred = crf(doutputs)
    loss = crf.neg_log_likelihood(doutputs, y)
    return y_pred, loss


def main(epoch_num=4, split_num=5):
    device = 'cpu'
    teacher_forcing_ratio = 0.3
    trn_X, trn_y, tst, word_to_ix = util.load()
    max_length = max(len(x) for x in trn_X)

    # train
    for sid in range(split_num):
        trn_ixs, vld_ixs = train_test_split(list(range(len(trn_X))),
                                            test_size=1 / split_num,
                                            shuffle=True,
                                            random_state=sid)
        # init
        print('SPLIT {} --------------------------------------------------------'.format(sid + 1))
        use_ngram = True
        ngram_size = 20000
        layers = 2
        direction = 2
        en_indim = 128
        en_outdim = 128
        de_indim = en_outdim * direction
        encoder = EncoderRNN(len(word_to_ix), en_indim, en_outdim,
                             bi=direction, num_layers=layers,
                             use_ngram=True, ngram_size=ngram_size).to(device)
        decoder = AttnDecoderRNN(
            de_indim, len(tag_to_ix), dropout=0.4, max_length=max_length).to(device)
        encoder_optimizer = optim.Adam(
            encoder.parameters(), lr=0.01, weight_decay=1e-6)
        decoder_optimizer = optim.Adam(
            decoder.parameters(), lr=0.01, weight_decay=1e-6)

        crf = CRF(decoder.hidden_size, tag_to_ix)

        tsize, vsize = len(trn_ixs), len(vld_ixs)
        # epoch train
        for epoch in range(epoch_num):
            # train ++++++++++++++++++++
            encoder.train()
            decoder.train()
            crf.train()
            avg_loss = 0
            losslist = []
            loss_every = 2  # len(trn_ixs) // 200
            for i in tqdm(np.random.permutation(trn_ixs)):
                ngram = [hash((0, trn_X[i][0])) % ngram_size] + [hash((trn_X[i][j], trn_X[i][j + 1])) %
                                                                 ngram_size for j in range(len(trn_X[i]) - 1)]
                g = torch.tensor(ngram, dtype=torch.long)
                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                length = len(trn_X[i])
                eoutputs = torch.zeros(
                    max_length, encoder.out_hdim * direction, device=device)
                loss = 0
                ehidden = encoder.initHidden()
                # print(ehidden.size())
                for ei in range(len(x)):
                    eoutput, ehidden = encoder(
                        x[ei], g[ei], ehidden)
                    # print('---')
                    # print(ehidden.size())
                    eoutputs[ei] = eoutput[0, 0]

                dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)
                dhidden = torch.cat(
                    [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
                doutputs = torch.zeros(
                    length, decoder.hidden_size, device=device)

                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    for di in range(length):
                        doutput, dhidden = decoder(
                            dinput, dhidden, eoutputs)
                        dinput = y[di]  # Teacher forcing
                        doutputs[di] = doutput[0, 0]
                else:
                    for di in range(length):
                        doutput, decoder_hidden = decoder(
                            dinput, dhidden, eoutputs)
                        topv, topi = doutput.topk(1)
                        dinput = topi.squeeze().detach()
                        doutputs[di] = doutput[0, 0]
                        if dinput.item() == STOP_TAG:
                            break

                loss = crf.neg_log_likelihood(doutputs, y)

                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                avg_loss += loss.item() / tsize

                if i % loss_every == 0:
                    losslist.append(loss.item())

            util.toPickle(os.path.join(const.DATAPATH,
                                       'loss_split_{}_epoch_{}.pk'.format(sid, epoch)), losslist)
            # eval ++++++++++++++++++++
            encoder.eval()
            decoder.eval()
            crf.eval()
            vld_loss = 0
            y_preds, y_trues = [], []
            for i in tqdm(np.random.permutation(vld_ixs)):
                #x = torch.tensor(trn_X[i], dtype=torch.long)
                #y = torch.tensor(trn_y[i], dtype=torch.long)

                y_pred, loss = evalone(max_length, encoder, decoder, crf, trn_X[i],
                                       direction=direction, y=trn_y[i])
                if loss != 0:
                    vld_loss += loss.item() / vsize
                y_preds.append(y_pred[:])
                y_trues.append(trn_y[i])

            print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
                epoch + 1, epoch_num, avg_loss, vld_loss))
            eval(y_preds, y_trues)
            print()

            # test ++++++++++++++++++++
            testing(encoder, decoder, crf, tst, max_length, direction,
                    name='split-{}-epoch-{}'.format(sid, epoch))

        util.savemodel(encoder, 'encoder-720_{}.pytorch'.format(sid))
        util.savemodel(decoder, 'attdecoder-720_{}.pytorch'.format(sid))
        util.savemodel(crf, 'crf-720_{}.pytorch'.format(sid))


if __name__ == '__main__':
    main()
