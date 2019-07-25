from typing import List
import const
import os
import pickle
import torch
from torch import nn
import numpy as np
import copy


def load_pretrained(name, word_to_ix, embed_size=128):
    import fasttext.FastText as fasttext
    model = fasttext.load_model(os.path.join(
        MODELPATH, '{}.bin'.format(name)))
    embedding_matrix = np.zeros((len(word_to_ix), embed_size))
    for w, ix in word_to_ix.items():
        embedding_matrix[ix] = model[w]
    return embedding_matrix


def train(trn_X, trn_y, trn_ixs, model, optimizer, device='cpu'):
    avg_loss = 0
    tsize = len(trn_ixs)
    for i in np.random.permutation(trn_ixs):  # tqdm(np.random.permutation(trn_ixs)):#
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        optimizer.zero_grad()
        loss = model.loss(x, y)
        avg_loss += loss.item() / tsize
        loss.backward()
        optimizer.step()
    return model, avg_loss


def evaluate(trn_X, trn_y, vld_ixs, model, device='cpu'):
    vld_loss = 0
    vsize = len(vld_ixs)
    y_preds, y_trues = [], []
    for i in np.random.permutation(vld_ixs):  # tqdm(np.random.permutation(vld_ixs)):#
        x = torch.tensor(trn_X[i], dtype=torch.long, device=device)
        y = torch.tensor(trn_y[i], dtype=torch.long, device=device)
        y_pred = model(x)
        loss = model.loss(x, y)
        vld_loss += loss.item() / vsize
        y_preds.append(y_pred[:])
        y_trues.append(trn_y[i])
    return vld_loss, y_preds, y_trues


def eval_info_bs():
    pass


def eval_info(y_preds, y_trues, ix_to_tag):
    assert len(y_preds) == len(y_trues)
    stat = {'a': [0, 0], 'b': [0, 0], 'c': [0, 0]}
    tags = ['a', 'b', 'c']
    prec = {'a': [], 'b': [], 'c': []}
    recl = {'a': [], 'b': [], 'c': []}
    f1 = {'a': 0, 'b': 0, 'c': 0}
    for j in range(len(y_preds)):
        y_pred, y_true = y_preds[j], y_trues[j]
        prestat, recstat = copy.copy(stat), copy.copy(stat)
        for i in range(len(y_pred)):
            t = ix_to_tag[int(y_true[i])]
            p = ix_to_tag[int(y_pred[i])]
            if p in tags:
                prestat[p][1] += 1
            if p in tags and p == t:
                prestat[p][0] += 1
                recstat[t][0] += 1
            if t in tags:
                recstat[t][1] += 1
        for x in tags:
            if recstat[x][1] != 0:
                recl[x].append(recstat[x][0] / recstat[x][1])
            if prestat[x][1] != 0:
                prec[x].append(prestat[x][0] / prestat[x][1])
    for x in tags:
        prec[x] = 0 if len(prec[x]) == 0 else sum(prec[x]) / len(prec[x])
        recl[x] = 0 if len(recl[x]) == 0 else sum(recl[x]) / len(recl[x])
        f1[x] = (2 * prec[x] * recl[x]) / (prec[x] + recl[x] + 1e-8)

    for x in tags:
        print('TAG {} \t prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
            x, prec[x], recl[x], f1[x]))

    print('AVG prec {:.4f} \t recl {:.4f} \t f1 {:.4f}'.format(
        sum(prec[x] for x in tags) / 3,
        sum(recl[x] for x in tags) / 3,
        sum(f1[x] for x in tags) / 3))


def savemodel(model, name):
    torch.save(model.state_dict(), os.path.join(const.MODELPATH, name))


def output(tsttag: List, name=''):
    """ """
    with open(os.path.join(const.DATAPATH, 'test.txt'), 'r') as f:
        lines = f.readlines()[:10]
        chars = [l.strip().split('_') for l in lines]
    assert len(chars) == len(tsttag)

    lines = []
    for i in range(len(chars)):
        assert len(chars[i]) <= len(tsttag[i])
        l = _merge(chars[i], tsttag[i][:len(chars[i])])
        lines.append(l + '\n')

    with open(os.path.join(const.DATAPATH, 'submit{}.txt'.format(name)), 'w') as f:
        f.writelines(lines)


def _merge(chars: List, tags: List):
    """"""
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

    lines = []
    for i in range(len(gtag)):
        lines.append('_'.join(gchar[i]) + '/' + gtag[i])
    return '  '.join(lines)


def load():
    """"""
    trn_X = fromPickle(os.path.join(
        const.DATAPATH, 'trn_X_token.pickle'))[:10]
    trn_y = fromPickle(os.path.join(
        const.DATAPATH, 'trn_y_token.pickle'))[:10]
    tst = fromPickle(os.path.join(
        const.DATAPATH, 'tst_X_token.pickle'))[:10]
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


def init_embedding(embedding):
    bias = np.sqrt(3.0 / embedding.embedding_dim)
    nn.init.uniform_(embedding.weight, -bias, bias)


def init_linear(linear):
    bias = np.sqrt(6.0 / (linear.weight.size(0) +
                          linear.weight.size(1)))
    nn.init.uniform_(linear.weight, -bias, bias)
    if linear.bias is not None:
        linear.bias.data.zero_()


def init_rnn(input_rnn, rnn='lstm', init_fn=nn.init.xavier_normal_):
    """
        Initialize lstm 
        但是为什么只初始化了两个权重，weight_ih_l[k] 和 weight_hh_l[k]
        对于RNN确实只需要训练ih和hh的权重，而对于LSTM, 相当于把4个张量合成到一个里面了
        (W_ii|W_if|W_ig|W_io) -> (4*hidden_size x input_size)
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size x input_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size x hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
    """
    part_num = 4 if rnn == 'lstm' else 3
    # weight
    for ind in range(0, input_rnn.num_layers):
        # 动态代码，类似于 执行 weight = input_lstm.weight_ih_l1
        weight = eval('input_rnn.weight_ih_l' + str(ind))
        hid_size = weight.size(0) // part_num
        for i in range(part_num):
            init_fn(weight[hid_size * i:hid_size * (i + 1), :])

        weight = eval('input_rnn.weight_hh_l' + str(ind))
        for i in range(part_num):
            init_fn(weight[hid_size * i:hid_size * (i + 1), :])
    if input_rnn.bidirectional:
        for ind in range(0, input_rnn.num_layers):
            weight = eval('input_rnn.weight_ih_l' + str(ind) + '_reverse')
            for i in range(part_num):
                init_fn(weight[hid_size * i:hid_size * (i + 1), :])

            weight = eval('input_rnn.weight_hh_l' + str(ind) + '_reverse')
            for i in range(part_num):
                init_fn(weight[hid_size * i:hid_size * (i + 1), :])
    # bias
    if input_rnn.bias:
        # 这里已经使用了把bias设成1的trick
        for ind in range(0, input_rnn.num_layers):
            bias = eval('input_rnn.bias_ih_l' + str(ind))
            bias.data.zero_()
            bias.data[input_rnn.hidden_size: 2 * input_rnn.hidden_size] = 1
            bias = eval('input_rnn.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_rnn.hidden_size: 2 * input_rnn.hidden_size] = 1
        if input_rnn.bidirectional:
            for ind in range(0, input_rnn.num_layers):
                bias = eval('input_rnn.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_rnn.hidden_size: 2 *
                          input_rnn.hidden_size] = 1
                bias = eval('input_rnn.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_rnn.hidden_size: 2 *
                          input_rnn.hidden_size] = 1
