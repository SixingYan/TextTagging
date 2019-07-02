from typing import Dict
from const import START_TAG, STOP_TAG


class Model(nn.Module):
    """BiGRU + CRF"""

    def __init__(self, vocab_size, tag_to_ix: Dict, embed_dim=10, hid_dim=10):
        super(BiLSTM_CRF, self).__init__()
        self.embed_dim = embed_dim
        self.hid_dim = hid_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tags_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embed_dim)

        self.rnn = nn.GRU(embed_dim, hid_dim // 2,
                          num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        # Input: (N,∗,in_features) where *∗ means any number of additional dimensions
        # Output: (N,∗,out_features) 除最后一个维度外，所有维度的形状都与输入相同
        self.hid2tag = nn.Linear(hid_dim, self.tags_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # size 为 tagset_size by tagset_size
        self.transitions = nn.Parameter(
            torch.randn(self.tags_size, self.tags_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hid_dim // 2),    # h_{t-1} 因为是前后双向，所以是两个神经元 2
                torch.randn(2, 1, self.hid_dim // 2))    # c_{t-1}

    def _forward_alg(self, feats: TorchTensor):
        '''
            input: len(sentence) by self.tagset_size
            output: 一个tensor值
        '''
        init_alphas = torch.full((1, self.tags_size), -10000.)

        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
        # Iterate through the sentence
        for feat in feats:
            alphas_t = []
            for next_tag in range(self.tags_size):
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tags_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score

                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence: TorchTensor)->TorchTensor:

        self.hidden = self.init_hidden()

        embeds = self.word_embeds(sentence).view(
            len(sentence), 1, -1)  # len(sentence) by 1 by embed_dim

        # self.lstm(input, (h_0, c_0))
        # input of shape (seq_len, batch, input_size)
        # output of shape (seq_len, batch, num_directions * hidden_size):
        # output features (h_t) from the last layer of the LSTM
        # 这里num_directions=2 和input的size已经不一样了
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        # the directions can be separated using
        # output.view(seq_len, batch, num_directions, hidden_size), with
        # forward and backward being direction 0 and 1 respectively. Similarly,
        # the directions can be separated in the packed case.
        # 所以这里相当于把前后两个拼在一起了，所以这个时候才能用hidden_dim，因为输入的分别是hidden_dim // 2
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)

        # Input: (N,∗,in_features) where *∗ means any number of additional dimensions
        # Output: (N,∗,out_features) 除最后一个维度外，所有维度的形状都与输入相同
        # 这里N是句子长度len(sentence)
        # 输出是 len(sentence) by self.tagset_size
        lstm_feats = self.hidden2tag(lstm_out)

        return lstm_feats

    def _score_sentence(self, feats, tags):
        '''
            input:
                feats: len(sentence) by self.tagset_size
                tags: 1 by tagset_size tensor
            output: 一个tensor值

        '''
        score = torch.zeros(1)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # 对score进行累加了
        for i, feat in enumerate(feats):

            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]

        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]

        return score

    def _viterbi_decode(self, feats):
        '''
            解码，得到预测的序列，以及预测序列的得分
            input: 
                feats: len(sentence) by self.tagset_size
            output:
        '''
        backpointers = []

        # Initialize the viterbi variables in log space
        # 1 by tagset_size
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars

        for feat in feats:
            # feat: 1 by self.tagset_size

            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):

                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                # 这里求出的是，如果next_tag 为某个标签的时候，当前最佳标签是哪个
                bptrs_t.append(best_tag_id)

                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        # 最后backpointers的size为 len(sentences) by tagset_size

        # Transition to STOP_TAG
        terminal_var = forward_var + \
            self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签到STOP_TAG的转移概率
        # 最后一定是转移到STOP_TAG

        # 最后一个词的最佳tag标签
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            # 后一个最佳tag是best_tag_id，那么当前这个best_tag_id 是谁
            best_tag_id = bptrs_t[best_tag_id]
            # 把这个加入最佳路径
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def neg_log_likelihood(self, sentence: TorchTensor, tags: TorchTensor):

        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
