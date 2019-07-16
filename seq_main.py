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

"""
def model0(batch_size=1):

    # init

    # prepare data

    trn_X, trn_y, tst, word_to_ix = util.load()

    trn_X = torch.tensor(trn_X, dtype=torch.long)
    trn_y = torch.tensor(trn_y, dtype=torch.long)
    trn_loader = torch.utils.data.TensorDataset(trn_X, trn_y)
    '''
    trn_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trn_X, trn_y),
        batch_size=batch_size, shuffle=True)
    
    train_X, train_y, valid_X, valid_y = train_test_split(
        trn_X, trn_y, test_size=0.1, shuffle=True)
    
    trn_loader, vld_loader = _pack_loader(
        x_train_fold, y_train_fold, x_val_fold, y_val_fold)
    '''
    tst_tensor = torch.tensor(tst, dtype=torch.long)

    # train
    model = model_v0.Model(len(word_to_ix), len(tag_to_ix))
    lossfunc = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
    model = train(trn_loader, model, lossfunc, optimizer)

    # test
    tsttag = test(tst_tensor, model)
    util.output(tsttag)

"""
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


def model01(epoch_num=2):
    trn_X, trn_y, tst, word_to_ix = util.load()
    # train
    model = Model(len(word_to_ix), tag_to_ix, 200, 200,2)
    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    model.train()
    for epoch in range(epoch_num):
        for i in tqdm(np.random.permutation(range(len(trn_X)))):
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

    util.savemodel(model, 'bigrucrf716.pytorch')

    # test
    tsttag = []
    model.eval()
    for i in tqdm(range(len(tst))):
        x = torch.tensor(tst[i], dtype=torch.long)
        tag = model(x)
        tsttag.append([ix_to_tag[t] for t in tag])

    util.output(tsttag)


'''
def multitime():

    from sklearn.model_selection import StratifiedKFold
    splits = list(StratifiedKFold(n_splits=split_num, shuffle=True, random_state=SEED).split(trn_X, trn_y))

    trn_X, trn_y, tst, vocab_size, tags_size = load()

    for i, (train_idx, valid_idx) in enumerate(splits):

        x_train_fold = torch.tensor(train_X[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(train_y[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold = torch.tensor(train_X[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(train_y[valid_idx, np.newaxis], dtype=torch.float32).cuda()

        trn_loader, vld_loader = _pack_loader(x_train_fold, y_train_fold, x_val_fold, y_val_fold)



def _pack_loader(x_train_fold, y_train_fold, x_val_fold, y_val_fold, batch_size=1):
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
'''

if __name__ == '__main__':
    model01()
