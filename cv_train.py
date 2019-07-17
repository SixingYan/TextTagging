import torch
import torch.utils.data
from torch import optim
from tqdm import tqdm
# tqdm.pandas(desc='Progress')
import numpy as np
#from Modeling import model_v0, model_v01
from model1 import Model

#from training import train, test
import util
from const import tag_to_ix, ix_to_tag
import const
import os
import copy

from sklearn.model_selection import train_test_split  # StratifiedKFold  # ,
SEED = 10


def main(epoch_num=2, split_num=5):
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
            t = y_true[i]
            if p in tags:
                prestat[p][1] += 1
            if p == t:
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

if __name__ == '__main__':
    main()
