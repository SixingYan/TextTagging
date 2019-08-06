import torch
from torch import nn
from typing import List


class ELMo(object):
    """docstring for ELMo"""
    def __init__(self, num_layer: int, embed_dim: int, hid_dim: int):
        super(ELMo, self).__init__()
        self.arg = arg

        self.embedding = nn.Embedding()

        for i in range(num_layer):
            nn.LSTM()  # forward
            nn.LSTM()  # backward
        for i in range(num_layer - 1):
            nn.Linear()
            nn.Linear()

    def _embed(self, x):
        pass
        return

    def _biLM(self, embed):
        pass
        return

    def _mixer(self, vectors: List):
        pass
        return

    def loss(self,):
        """
        这里直接输出output
        """
        pass

    def forward(self,):
        # 协同训练的时候
        """
        这里才加上mixer
        """
        pass
        x_embed = self._embed(x)

        return
