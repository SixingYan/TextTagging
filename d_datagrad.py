
import os
import const
from const import tag_to_ix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import util


PATH = os.path.join(const.DATAPATH, 'datagrad')

def _parse_str_tag(line):
    f"""
    7212_17592_21182/c  8487_8217_14790_19215_4216_17186_6036_18097_8197_11743_18102_5797_6102_15111_2819_10925_15274/o
    """
    words,tags = [],[]
    for chuck in line.strip().split():
        c, t =chuck.split('/')
        words += c.split('_')
        tags += [t]*len(c.split('_'))
    return words, tags

def preprocess():
    """"""
    print('start...')
    trn_X = []
    trn_y = []
    with open(os.path.join(PATH, 'train.txt'), 'r') as f:
        for line in f:
            words, tags = _parse_str_tag(line)
            trn_X.append(' '.join(words))
            trn_y.append(' '.join(tags))

    with open(os.path.join(PATH, 'test.txt'), 'r') as f:
        tst_X = [line.replace('_', ' ') for line in f.readlines()]

    with open(os.path.join(PATH, 'corpus.txt'), 'r') as f:
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

    corpus_token = tokenizer.texts_to_sequences(corpus)

    word_to_ix = tokenizer.word_index
    word_to_ix[const.PAD_TAG] = 0
    util.toPickle(os.path.join(PATH, 'trn_X_token.pickle'), trn_X_token)
    util.toPickle(os.path.join(PATH, 'trn_y_token.pickle'), trn_y_token)
    util.toPickle(os.path.join(PATH, 'tst_X_token.pickle'), tst_X_token)
    util.toPickle(os.path.join(PATH, 'corpus_token.pickle'), corpus_token)
    util.toPickle(os.path.join(PATH, 'word_to_ix.pickle'), word_to_ix)


if __name__ == '__main__':
    preprocess()
    '''
    s = '7212_17592_21182/c  8487_8217_14790_19215_4216_17186_6036_18097_8197_11743_18102_5797_6102_15111_2819_10925_15274/o'
    w,t=_parse_str_tag(s)
    print(w)
    print(t)
    '''