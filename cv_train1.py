from cv_train import eval
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


def test():
    device = 'cpu'
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
        os.path.join(const.MODELPATH, 'encoder-719_0.pytorch')))
    decoder.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'attdecoder-719_0.pytorch')))
    crf.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'crf-719_0.pytorch')))

    tsttag = []
    encoder.eval()
    decoder.eval()
    crf.eval()
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


def evalone(max_length, encoder, decoder, crf, x, direction=1, y=None, device='cpu'):
    eoutputs = torch.zeros(
        max_length, encoder.in_hdim * direction, device=device)
    for ei in range(len(x)):
        eoutput, ehidden = encoder(x[ei], encoder.init_hidden())
        ehidden = torch.cat([ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        eoutputs[ei] += eoutputs[0, 0]

    dinput = torch.tensor([[tag_to_ix[START_TAG]]], device=device)  # SOS
    dhidden = ehidden
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


def main(epoch_num=2, split_num=5):
    device = 'cpu'
    teacher_forcing_ratio = 0.5
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
        direction = 2
        en_indim = 128
        en_outdim = 128
        de_indim = en_outdim * direction
        encoder = EncoderRNN(len(word_to_ix), en_indim, en_outdim,
                             bi=direction, num_layers=2).to(device)
        decoder = AttnDecoderRNN(
            de_indim, len(tag_to_ix), dropout=0.5, max_length=max_length).to(device)
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-8)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001, weight_decay=1e-8)

        crf = CRF(decoder.hidden_size, tag_to_ix)

        tsize, vsize = len(trn_ixs), len(vld_ixs)
        # epoch train
        for epoch in range(epoch_num):
            # train ++++++++++++++++++++
            encoder.train()
            decoder.train()
            crf.train()
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

            # eval ++++++++++++++++++++
            encoder.eval()
            decoder.eval()
            crf.eval()
            vld_loss = 0
            y_preds, y_trues = [], []
            for i in tqdm(np.random.permutation(vld_ixs)):
                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)

                y_pred, loss = evalone(max_length, encoder, decoder, crf, x, direction=direction, y=y)
                if loss != 0:
                    vld_loss += loss.item() / vsize
                y_preds.append(y_pred[:])
                y_trues.append(trn_y[i])

            print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
                epoch + 1, epoch_num, avg_loss, vld_loss))
            eval(y_preds, y_trues)
            print()

        util.savemodel(encoder, 'encoder-719_{}.pytorch'.format(sid))
        util.savemodel(decoder, 'attdecoder-719_{}.pytorch'.format(sid))
        util.savemodel(crf, 'crf-719_{}.pytorch'.format(sid))


if __name__ == '__main__':
    main()
