model = fasttext.train_unsupervised(os.path.join(const.DATAPATH, 'corp.txt'),
                                    model='skipgram',
                                    dim=128,
                                    epoch=10,
                                    ws=5,
                                    minCount=1,
                                    loss='hs',
                                    wordNgrams=2)
model.save_model(os.path.join(const.MODELPATH, 'skipgram.bin'))


model = fasttext.train_unsupervised(os.path.join(const.DATAPATH, 'corp.txt'),
                                    model='cbow',
                                    dim=128,
                                    epoch=10,
                                    ws=5,
                                    minCount=1,
                                    loss='hs',
                                    wordNgrams=2)
model.save_model(os.path.join(const.MODELPATH, 'cbow.bin'))
