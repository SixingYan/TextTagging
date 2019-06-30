import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

# 用于注解
from typing import List, Dict, TypeVar
TorchTensor = TypeVar('TorchTensor')

#####################################################################
# Helper functions to make the code more readable.


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq: List[str], to_ix: Dict)->TorchTensor:
    '''
        input: sentence word list, dict
        output: 1 by len(sentence)
    '''
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec: TorchTensor)->TorchTensor:
    '''
        input: 1 by tarset_size
        output: 一个tensor值
    '''
    max_score = vec[0, argmax(vec)]

    # vec.size()[1] : tarset_size
    # 把这个值拓成 1 by tarset_size
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])

    # 实现技巧，减去最大值，然后再去算exp
    # 因为已经减了，max_score那一项为0，所以要在前面加回来
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

#####################################################################
# Create model


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        '''
        # 加载pretrain model
        if pre_word_embeds is not None:
            self.pre_word_embeds = True
            self.word_embeds.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
        else:
            self.pre_word_embeds = False
        '''
        '''可以替换成pretrain
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = nn.Dropout2d(0.1)

        # Word-embedding layer.
        self.word_vec_dim = pretrained_word_vecs.size()[1]
        self._freeze_embeddings = freeze_embeddings
        self.word_embedding = \
            nn.Embedding.from_pretrained(pretrained_word_vecs, freeze=freeze_embeddings)

        self.dropout = nn.Dropout(p=dropout) if dropout else None

        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True) # num_layers 应该不止为1
        '''

        # embedding_dim : input_size – The number of expected features in the input x
        # hidden_dim // 2 : hidden_size – The number of features in the hidden
        # state h
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        '''
        self.lstm = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=hidden_dim,
            num_layers=layers,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0,
            batch_first=True,
        )
        '''

        # Maps the output of the LSTM into tag space.
        # Input: (N,∗,in_features) where *∗ means any number of additional dimensions
        # Output: (N,∗,out_features) 除最后一个维度外，所有维度的形状都与输入相同
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # size 为 tagset_size by tagset_size
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # 标准正态分布randn size = (2, 1, self.hidden_dim // 2)
        # num_layers * num_directions, batch, hidden_size, If bidirectional,
        # num_directions should be 2, else 1
        # batch 1
        # hidden_size=self.hidden_dim // 2
        return (torch.randn(2, 1, self.hidden_dim // 2),    # h_{t-1} 因为是前后双向，所以是两个神经元 2
                torch.randn(2, 1, self.hidden_dim // 2))    # c_{t-1}

    def _forward_alg(self, feats: TorchTensor):
        '''
            input: len(sentence) by self.tagset_size
            output: 一个tensor值
        '''

        # Do the forward algorithm to compute the partition function
        # (1, self.tagset_size) the shape of the output tensor 1 by self.tagset_size
        # -10000. 初始化的值
        # output 一个 1 by self.tagset_size
        init_alphas = torch.full((1, self.tagset_size), -10000.)

        # START_TAG has all of the score.
        # START_TAG 标签的值为0
        # tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = autograd.Variable(init_alphas) #初始状态的forward_var，随着step t变化
        # 上面是在其他地方看到的写法，不知道有什么区别
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:  # 每一个词的tag向量

            alphas_t = []  # The forward tensors at this timestep

            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                # feat[next_tag]: 1 by tagset_size
                # 把它拓成（通过复制）1 by self.tagset_size
                # 感觉这里size并没有什么变化
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                # size 为 1 by self.tagset_size
                trans_score = self.transitions[next_tag].view(1, -1)

                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 直接求和
                # 计算概念是 (1)这个词前面所有词的标注情况+(2)转移到当前的概率+(3)当前对应某个标签的概率
                # 所以这里才是便利每一个标签（针对第三个变量）
                next_tag_var = forward_var + trans_score + emit_score

                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                # 加入一个 tensor值的结果 然后转为 1 by 1
                # 这里每次加入一个 当前的一种情况的数值，它将成为下次的“(1)前面所有的标注情况”
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            # alphas_t List where length is tagset_size
            # cat 拼接 alphas_t 默认dim=0
            # 例如 [3x2,3x2,3x2],dim=0 = [9X2] ; dim=1 : [3X6]
            # 原本是 [1x1,1x1,...]
            # torch.cat 再转成 1 by len(sentence)
            # 遍历后，它就成为“从开头到当前词的所有标注可能”了，
            # 它会成为下一轮的“(1)这个词前面所有词的标注情况”
            forward_var = torch.cat(alphas_t).view(1, -1)

        # 最后再加上一个stop_tag（-1000）的值，结果还是1 by len(sentence)
        # 最后只将最后一个单词的forward var与转移 stop tag的概率相加
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]

        # 一个tensor值
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _get_lstm_features(self, sentence: TorchTensor)->TorchTensor:
        '''
            这里是11x5的一个tensor，而这个11x5的tensor，就是发射矩阵！（emission matrix）
            input: 
                sentence: len(sentence) by embed_size
            output:len(sentence) by self.tagset_size
        '''

        self.hidden = self.init_hidden()
        # 查找这个句子所有词的词向量，相当于输入了batch word，输出还是len(sentence) by embed_dim
        # -1 这个位置由其他位置的数字来推断
        # (seq_len, batch, input_size), input_size就是embed_dim
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

        # Gives the score of a provided tag sequence
        # size (1)
        # score = autograd.Variable(torch.Tensor([0])) 其他地方看到的写法
        score = torch.zeros(1)

        # tags: 1 by tagset_size tensor
        # [tensor(-1000)] : tensor([self.tag_to_ix[START_TAG]], dtype=torch.long)
        # 这个操作相当于把tensor(-1000)这个值接在开头
        # tags 更新为 1 by (1+tagset_size)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        # 对score进行累加了
        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG,
            # END_TAG, 取对应标签的值

            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
            # +1 是因为已经在tags的头加上了一个[tensor(-1000)]

        # 最后的分数还要加上 最后一个词作为结尾 的转移分数
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
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.

                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                # 这里求出的是，如果next_tag 为某个标签的时候，当前最佳标签是哪个
                bptrs_t.append(best_tag_id)

                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            # 从step0到step(i-1)时5个序列中每个序列的最大viterbi variable
            # forward_var 为 viterbi variable + 当前的feature
            # 更新forward_var
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)

            # bptrs_t有５个元素
            # bptrs_t List which length is self.tagset_size
            # 对应于前面的每种标签，当前词的最佳标签是哪个
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
        '''
            input: 
                sentence: len(sentence) by embed_size
                tags: 1 by tagset_size tensor
            output: 一个tensor值
        '''
        # feats = TorchTensor(len(sentence) by self.tagset_size)
        feats = self._get_lstm_features(sentence)

        # forward_score 一个tensor值
        forward_score = self._forward_alg(feats)

        # gold_score 一个tensor值
        gold_score = self._score_sentence(feats, tags)

        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        '''
            把定义的网络模型model当作函数调用的时候就自动调用定义的网络模型的forward方法
            当执行model(x)的时候，底层自动调用forward方法计算结果
            input 就是调用model时实际调用函数的输入
        '''
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

#####################################################################
# Run training


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
'''
learning_rate = 0.015
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
'''
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor(
        [tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()

# Check predictions after training
with torch.no_grad():
    print('start to check')
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(precheck_sent)
    print(model(precheck_sent))
# We got it!


'''
# Check predictions after training
precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
print(model(precheck_sent)[0]) #得分
print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
print(model(precheck_sent)[1]) #tag sequence
'''
