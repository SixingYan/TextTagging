import torch
import torch.utils.data
from torch import nn, optim


from Modeling.model_v0 import Model
from training import train
from loading import load


def onetime():
    from sklearn.model_selection import train_test_split

    # init
    model = Model()
    lossfunc = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # prepare data
    trn_X, trn_y, tst, word_to_ix = load()
    train_X, train_y, valid_X, valid_y = train_test_split(trn_X, trn_y, test_size=0.2, shuffle=True)
    trn_loader, vld_loader = _pack_loader(x_train_fold, y_train_fold, x_val_fold, y_val_fold)
    tst_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(
            test_X, dtype=torch.long)), batch_size=batch_size, shuffle=False)
    
    # train
    model = train(trn_loader, vld_loader, model, lossfunc, optimizer)

    # test
    test(tst_loader, model)

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
'''


def _pack_loader(x_train_fold, y_train_fold, x_val_fold, y_val_fold):
    train = torch.utils.data.TensorDataset(x_train_fold, y_train_fold)
    valid = torch.utils.data.TensorDataset(x_val_fold, y_val_fold)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
