from typing import List
import const
import os
import pickle
import torch
from torch import nn
import numpy as np


def savemodel(model, name):
    torch.save(model.state_dict(), os.path.join(const.MODELPATH, name))


def retestmodel():
    pass


def output(tsttag: List, name=''):
    """

    """
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


def init_embedding(input_embedding):
    """
    Initialize embedding
    """
    bias = np.sqrt(3.0 / input_embedding.embedding_dim)
    nn.init.uniform_(input_embedding.weight, -bias, bias)


def init_linear(input_linear):
    """
    Initialize linear transformation
    这里weight基本都通过偏置来矫正了
    """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) +
                          input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)

    if input_linear.bias is not None:
        # data：存储了Tensor，是本体的数据
        # 将内置bias 初始化为0
        input_linear.bias.data.zero_()


def init_rnn(input_rnn, rnn='lstm'):
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
        # 这是动态代码
        # 类似于 执行 weight = input_lstm.weight_ih_l1
        weight = eval('input_rnn.weight_ih_l' + str(ind))
        hid_size = weight.size(0) // part_num

        for i in range(part_num):
            nn.init.xavier_normal_(weight[hid_size * i:hid_size * (i + 1), :])

        weight = eval('input_rnn.weight_hh_l' + str(ind))
        for i in range(part_num):
            nn.init.xavier_normal_(weight[hid_size * i:hid_size * (i + 1), :])
    if input_rnn.bidirectional:
        for ind in range(0, input_rnn.num_layers):
            weight = eval('input_rnn.weight_ih_l' + str(ind) + '_reverse')
            for i in range(part_num):
                nn.init.xavier_normal_(
                    weight[hid_size * i:hid_size * (i + 1), :])

            weight = eval('input_rnn.weight_hh_l' + str(ind) + '_reverse')
            for i in range(part_num):
                nn.init.xavier_normal_(
                    weight[hid_size * i:hid_size * (i + 1), :])
    # bias
    if input_rnn.bias:
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
