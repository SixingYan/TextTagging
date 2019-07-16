import torch
import torch.utils.data
from torch import nn, optim
from tqdm import tqdm
# tqdm.pandas(desc='Progress')
import numpy as np
#from Modeling import model_v0, model_v01
from model import Model

from training import train, test
import util
from const import tag_to_ix, ix_to_tag
import const
import os
#from sklearn.model_selection import train_test_split


def retest():
    trn_X, trn_y, tst, word_to_ix = util.load()
    model = model_v01.Model(len(word_to_ix), tag_to_ix, 300, 200)
    model.load_state_dict(torch.load(
        os.path.join(const.MODELPATH, 'bigrucrf.pytorch')))

    tsttag = []
    model.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        tag = model(x)
        tsttag.append([ix_to_tag[t] for t in tag])

    util.output(tsttag)


def main(epoch_num=2):
    trn_X, trn_y, tst, word_to_ix = util.load()

    splits = list(StratifiedKFold(n_splits=split_num, shuffle=True, random_state=SEED).split(trn_X, trn_y))

    # train
    # CV 并且加上阈值搜索的方式
    for i, (trn_ixs, vld_ixs) in enumerate(splits):
        # init

        # epoch train
        for epoch in range(epoch_num):
            # train
            model.train()
            for i in tqdm(np.random.permutation(trn_ixs)):

                x = torch.tensor(trn_X[i], dtype=torch.long)
                y = torch.tensor(trn_y[i], dtype=torch.long)
                model.zero_grad()
                optimizer.zero_grad()

                feats = model._get_lstm_features(x)
                forward_score = model._forward_alg(feats)
                gold_score = model._score_sentence(feats, y)
                loss = forward_score - gold_score

                loss.backward()
                optimizer.step()

            # eval
            model.eval()
            y_true
            
            for i in tqdm(np.random.permutation(vld_ixs)):
                tag = model(torch.tensor(trn_X[i], dtype=torch.long))

                tsttag.append([ix_to_tag[t] for t in tag])





    model = Model(len(word_to_ix), tag_to_ix, 200, 200, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

    model.train()
    for epoch in range(epoch_num):
        for i in tqdm(np.random.permutation(range(len(trn_X)))):
            x = torch.tensor(trn_X[i], dtype=torch.long)
            y = torch.tensor(trn_y[i], dtype=torch.long)

            model.zero_grad()
            optimizer.zero_grad()
            loss = model.neg_log_likelihood(x, y)
            loss.backward()
            optimizer.step()

    util.savemodel(model, 'bigrucrf716.pytorch')

    # test
    tsttag = []
    model.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        tag = model(x)
        tsttag.append([ix_to_tag[t] for t in tag])

    util.output(tsttag)


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in tqdm([i * 0.01 for i in range(100)]):
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    return search_result
search_result = threshold_search(train_y, train_preds)
search_result


if __name__ == '__main__':
    model01()
