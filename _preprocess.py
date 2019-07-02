

在这里就处理成ix了

每一行数据 1 2 3 4 5 / t a a b b o


需要统计最大长度


def preprocess():
    先转化成 空格 分割

    trn_X
    trn_y

    tst_X

    with open('') as f:
        sent = f.readlines()
    corpus = DataFrame({'sent': sent}).process_apply(lambda x: ' '.join(x.split('_'))).values.tolist()

    # 这里再生成词表
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(trn_X + tst_X + corpus)  # two list combine
    trn_X_token = tokenizer.texts_to_sequences(trn)
    trn_X_token = pad_sequences(trn_token, maxlen=MAX_LEN, padding='post')

    tst_X_token = tokenizer.texts_to_sequences(tst)
    tst_X_token = pad_sequences(trn_token, maxlen=MAX_LEN, padding='post')

    toPickle(os.path.join(const.DATAPATH, ''), word_to_ix)

    word_to_ix = tokenizer.word_index
    word_to_ix[PAD_TAG] = 0

    ix_to_word = {ix: w for w, ix in tokenizer.word_index.items()}


def toPickle(path, var):
    pass


def fromPickle():
    pass


if __name__ == '__main__':
    main()
