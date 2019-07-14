from torch import nn
import torch.nn.functional as F


class Model(nn.Module):
    """docstring for Model"""
    def __init__(self, vocab_size, tags_size, embed_dim=10, hid_dim=10):
        super(Model, self).__init__()
        self.hid_dim = hid_dim
        self.word_embeds = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hid_dim, bidirectional=True)
        self.hid2tag = nn.Linear(hid_dim * 2, tags_size)

    def forward(self, sentence):
        embeds = self.word_embeds(sentence)
        print()
        lstm_out, _ = self.rnn(embeds.view(len(sentence), 1, -1))
        tag_space = self.hid2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
