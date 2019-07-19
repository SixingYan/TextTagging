"""
这里用于预训练词向量
"""
import fasttext.FastText as fasttext
import os
import const
"""
with open(os.path.join(const.DATAPATH, 'corpus.txt'), 'r') as f:
    lines = []
    for line in f:
        lines.append(' '.join(line.strip().split('_')) + '\n')

with open(os.path.join(const.DATAPATH, 'corp.txt'), 'w') as f:
    f.writelines(lines)
"""
#fasttext.train_unsupervised('data.txt', model='cbow')

model = fasttext.train_unsupervised(os.path.join(const.DATAPATH, 'corp.txt'),
                                    model='skipgram',
                                    dim=200,
                                    ws=10,
                                    minCount=2,
                                    neg=10,
                                    wordNgrams=3)
model.save_model(os.path.join(const.MODELPATH, 'skipgram.bin'))
