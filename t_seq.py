import os

from torch import nn, optim
from sklearn.model_selection import train_test_split

import util

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
seed = 10
use_pre = False
pre_embed_name = 'cbow'
vld_size = 0.1
source = 'datagrad'  # 'msra'  # 'boson'  #
BATHPATH = os.getcwd()
DATAPATH = os.path.join(BATHPATH, 'Data', source)
lr = 0.01
epoch_num = 4
use_alpha = False
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

X = util.fromPickle(os.path.join(DATAPATH, 'Xtoken.pickle'))[:10]
y = util.fromPickle(os.path.join(DATAPATH, 'Ytoken.pickle'))[:10]
word_to_ix = util.fromPickle(os.path.join(DATAPATH, 'word_to_ix.pickle'))
#tag_to_ix = util.load_pk(DATAPATH, 't2x.pickle')
tag_to_ix = {"a": 0, "b": 1, "c": 2, "o": 3, }
ix_to_tag = {tp[1]: tp[0] for tp in tag_to_ix.items()}

trn_ixs, vld_ixs = train_test_split(
    list(range(len(X))), test_size=vld_size,
    shuffle=True, random_state=seed)

if use_pre:
    pretrianed_embedding = util.load_pretrained(pre_embed_name, word_to_ix)
else:
    pretrianed_embedding = None

if use_alpha:
    tagix_to_frq = {0: 0, 1: 0, 2: 0, 3: 0, }
    for ix in trn_ixs:
        for t in y[ix]:
            tagix_to_frq[t] += 1
    frqs = [tagix_to_frq[t] for t in [0, 1, 2, 3]]
    alpha = [f / sum(frqs) for f in frqs]
else:
    alpha = None

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

init_fns = {'rnn': util.init_rnn,
            'embed': util.init_embedding,
            'linear': util.init_linear}

from m_focalloss import FocalLoss
loss_fn = FocalLoss(len(tag_to_ix), alpha=alpha, size_average=False)


from m_bilstm import BiLSTM
model = BiLSTM(vocab_size=len(word_to_ix), tag_size=len(tag_to_ix),
               embed_dim=128, hid_dim=128, num_layers=2,
               loss_fn=loss_fn,
               use_dropout=True, dropout=0.1,
               pre_word_embeds=pretrianed_embedding,
               use_subembed=True, subembed_dim=64, subhid_dim=64,
               use_init=True, init_fns=init_fns)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
for epoch in range(epoch_num):
    model.train()
    model, avg_loss = util.train(X, y, trn_ixs, model, optimizer)

    model.eval()
    vld_loss, y_preds, y_trues = util.evaluate(X, y, vld_ixs, model)

    print('Epoch {}/{} \t avg_loss {:.4f} \t vld_loss {:.4f} \t'.format(
        epoch + 1, epoch_num, avg_loss, vld_loss))
    util.eval_info(y_preds, y_trues, ix_to_tag)
