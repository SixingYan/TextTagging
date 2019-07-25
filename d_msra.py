
import os
import util

BATHPATH = os.getcwd()
DATAPATH = os.path.join(BATHPATH, 'Data', 'msra')


def preprocess():
    print('start...')
    trn_X = []
    trn_y = []
    seqs = util.load_pk(DATAPATH, '.pickle')
    labels = util.load_pk(DATAPATH, '.pickle')
    word_to_ix = {'<PAD>': 0}
    for seq in seqs:
        for c in seq:
            if c not in word_to_ix:
                word_to_ix[c] = len(word_to_ix)
    for seq in seqs:
        trn_X.append([word_to_ix[c] for c in seq])

    tag_to_ix = {}
    for label in labels:
        for c in label:
            if c not in tag_to_ix:
                tag_to_ix[c] = len(tag_to_ix)
    for label in labels:
        trn_y.append([tag_to_ix[c] for c in label])

    util.toPickle(os.path.join(DATAPATH, 'Xtoken.pickle'), trn_X)
    util.toPickle(os.path.join(DATAPATH, 'Ytoken.pickle'), trn_y)
    util.toPickle(os.path.join(DATAPATH, 'tag_to_ix.pickle'), tag_to_ix)
    util.toPickle(os.path.join(DATAPATH, 'word_to_ix.pickle'), word_to_ix)


if __name__ == '__main__':
    preprocess()
