
import os
import const

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


def _parse_str_tag(string):
    paras = string.split()
    words = []
    tags = []
    for p in paras:
        word, tag = p.split('/')
        chars = list(word.split('_'))
        words += chars
        tags += [tag] * len(chars)
    return words, tags


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
    MAX_LEN = max([len(line) - line.count(' ') for line in tst_X])

    # 这里再生成词表
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(trn_X + tst_X + corpus)  # two list combine
    trn_X_token = tokenizer.texts_to_sequences(trn_X)  # word to id
    trn_X_token = pad_sequences(trn_X_token, maxlen=MAX_LEN, padding='post', value=0)

    tst_X_token = tokenizer.texts_to_sequences(tst_X)
    tst_X_token = pad_sequences(tst_X_token, maxlen=MAX_LEN, padding='post', value=0)

    word_to_ix = tokenizer.word_index
    word_to_ix[const.PAD_TAG] = 0

    #ix_to_word = {ix: w for w, ix in tokenizer.word_index.items()}
    
    toPickle(os.path.join(const.DATAPATH, 'trn_X_token.pickle'), trn_X_token)
    toPickle(os.path.join(const.DATAPATH, 'tst_X.pickle'), tst_X)
    toPickle(os.path.join(const.DATAPATH, 'tst_X_token.pickle'), tst_X_token)
    toPickle(os.path.join(const.DATAPATH, 'corpus.pickle'), corpus)
    toPickle(os.path.join(const.DATAPATH, 'word_to_ix.pickle'), word_to_ix)


def toPickle(path, var):
    with open(path, 'wb') as f:
        pickle.dump(var,f)


def fromPickle():
    pass


if __name__ == '__main__':
    preprocess()
