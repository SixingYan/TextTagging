from typing import List
import const
import os
import pickle
import torch


def savemodel(model, name):
    torch.save(model.state_dict(), os.path.join(const.MODELPATH, name))


def output(tsttag: List):
    """

    """
    with open(os.path.join(const.DATAPATH, 'test.txt'), 'r') as f:
        lines = f.readlines()  # [:10]
        chars = [l.split('_') for l in lines]
    assert len(chars) == len(tsttag)

    lines = []
    for i in range(len(chars)):
        assert len(chars[i]) <= len(tsttag[i])
        l = _merge(chars[i], tsttag[i][:len(chars[i])])
        lines.append(l + '\n')

    with open(os.path.join(const.DATAPATH, 'submit.txt'), 'w') as f:
        f.writelines(lines)


def _merge(chars: List, tags: List):
    """

    """
    gchar, gtag = [], []
    curc, curt = [chars[0]], tags[0]
    for i in range(1, len(chars)):
        if tags[i] == tags[i - 1]:
            curc.append(chars[i])
        else:
            gchar.append(curc[:])
            gtag.append(curt)
            curc, curt = [chars[i]], tags[i]
        if i == len(chars) - 1:
            gchar.append(curc[:])
            gtag.append(curt)

    line = ''
    for i in range(len(gtag)):
        line += ('_'.join(gchar[i]) + '/' + gtag[i] + ' ')
    return line


def load():
    """"""
    trn_X = fromPickle(os.path.join(
        const.DATAPATH, 'trn_X_token.pickle'))#[:10]
    trn_y = fromPickle(os.path.join(const.DATAPATH, 'trn_y_token.pickle'))#[:10]
    tst = fromPickle(os.path.join(const.DATAPATH, 'tst_X_token.pickle'))#[:10]
    word_to_ix = fromPickle(os.path.join(
        const.DATAPATH, 'word_to_ix.pickle'))

    return trn_X, trn_y, tst, word_to_ix


def fromPickle(path):
    """"""
    var = None
    with open(path, 'rb') as f:
        var = pickle.load(f)
    return var


def toPickle(path, var):
    """"""
    with open(path, 'wb') as f:
        pickle.dump(var, f)
