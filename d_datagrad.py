
import os
import const
from const import tag_to_ix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import util


BATHPATH = os.getcwd()
DATAPATH = os.path.join(BATHPATH, 'Data', 'datagrad')


def preprocess():

    print('start...')
    trn_X = []
    trn_y = []
    with open(os.path.join(const.DATAPATH, 'train.txt'), 'r') as f:
        for line in f:
            words, tags = _parse_str_tag(line)
            trn_X.append(' '.join(words))
            trn_y.append(' '.join(tags))

    with open(os.path.join(const.DATAPATH, 'test.txt'), 'r') as f:
        tst_X = [line.replace('_', ' ') for line in f.readlines()]

    with open(os.path.join(const.DATAPATH, 'corpus.txt'), 'r') as f:
        corpus = [line.replace('_', ' ') for line in f.readlines()]

    print('load complete ')

    # 这里再生成词表
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(trn_X + tst_X + corpus)  # two list combine
    trn_X_token = tokenizer.texts_to_sequences(trn_X)  # word to id

    tst_X_token = tokenizer.texts_to_sequences(tst_X)
    trn_y_token = []
    for i in range(len(trn_y)):
        tmp = [tag_to_ix[y] for y in trn_y[i].split()]
        trn_y_token.append(tmp[:])

    word_to_ix = tokenizer.word_index
    word_to_ix[const.PAD_TAG] = 0
    util.toPickle(os.path.join(DATAPATH, 'trn_X_token.pickle'), trn_X_token)
    util.toPickle(os.path.join(DATAPATH, 'trn_y_token.pickle'), trn_y_token)
    util.toPickle(os.path.join(DATAPATH, 'tst_X_token.pickle'), tst_X_token)
    util.toPickle(os.path.join(DATAPATH, 'corpus.pickle'), corpus)
    util.toPickle(os.path.join(DATAPATH, 'word_to_ix.pickle'), word_to_ix)


if __name__ == '__main__':
    preprocess()
