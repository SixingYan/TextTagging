import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List


class ELMoMixer(nn.Module):
    """docstring for Mixer"""

    def __init__(self, 
        elmo, 
        num_layer: int, 
        normal_fn=nn.init.xavier_uniform_):
        super(ELMoMixer, self).__init__()

        self.elmo = elmo
        self.normal_fn = normal_fn
        self.scalar_parameters = nn.ParameterList([
            nn.Parameter(torch.FloatTensor([1/(num_layer+1)]),requires_grad=True) for _ in range(num_layer+1)
        ])
        self.gamma = nn.Parameter(torch.FloatTensor([1.0]),requires_grad=True)

    def forward(self, x):
        """
        x: torch.tensor([1,2,3], dtype=torch.long, device=device)
        """
        # 1. each layer representation
        if len(x.size()) < 3:
            x_batch = x.unsqueeze(0)

        embed = self.elmo._embed(x_batch)
        _, features = self.elmo(x_batch)
        features = [torch.cat([torch.clone(embed),torch.clone(embed)],dim=2)]+features

        # 2. scalar_parameters normalization
        # concat to softmax
        norm_scalars = F.softmax(torch.cat(
            [p for p in self.scalar_parameters]), dim=0)
        # split to list
        norm_scalars = torch.split(norm_scalars, split_size_or_sections=1)

        # 3. weight sum
        pieces = []
        for feat, scalar in zip(features, norm_scalars):
            pieces.append(self.normal_fn(feat) * scalar)
        embedding= self.gamma * torch.sum(torch.cat(pieces,dim=0),dim=0)
        print(embedding.size())

        return embedding


class ELMoChar(nn.Module):
    """ELMoChar: 只针对char级别的训练任务
        所以取消了char-embedding的部分
    """

    def __init__(self, vocab_size:int, num_layer: int,
                 embed_dim: int, hid_dim: int,
                 loss_fn=nn.NLLLoss()):
        super(ELMoChar, self).__init__()
        self.loss_fn = loss_fn
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.num_layer = num_layer
        self.nets =  []
        这里可能得注册到module里面才行
        
        for i in range(num_layer):
            fnn = nn.LSTM(embed_dim, hid_dim, batch_first=True)  # forward
            bnn = nn.LSTM(embed_dim, hid_dim, batch_first=True)  # backward
            self.nets.append([fnn, bnn])
        
        self.linears = []
        for i in range(num_layer):
            fln = nn.Linear(hid_dim, embed_dim)
            bln = nn.Linear(hid_dim, embed_dim)
            self.linears.append([fln, bln])

        self.embed2out = nn.Linear(hid_dim*2, vocab_size)

    def _embed(self, x_batch: torch.Tensor):
        """
        x_batch: Batch(sentence length) * 1 (each char)
        embedding 层，这里只考虑字符集别
        """
        embed = self.embedding(x_batch)
        return embed  # batch_size * 1 * embed_dim

    def _biLM(self, fembed: torch.Tensor, bembed: torch.Tensor):
        """
        embed: batch_size * 1 * embed_dim
        """
        features = []
        for i in range(self.num_layer):
            foutput, _ = self.nets[i][0](fembed)
            fembed = self.linears[i][0](foutput)
            #print(fembed.device)

            boutput, _ = self.nets[i][1](bembed)
            bembed = self.linears[i][1](boutput)

            rembed = torch.cat([fembed, bembed],dim=2)
            features.append(rembed)

        return (foutput, boutput), features

    def _feature(self, x_batch: torch.Tensor):
        """
        x_batch: Batch(sentence length) * 1 (each char)
        """
        fembed = self._embed(x_batch)
        bembed = self._embed(torch.flip(x_batch, (0,)))
        
        #print(fembed.device)
        #print(bembed.device)
        output, features = self._biLM(fembed, bembed)
        return output, features

    def loss(self, x_batch: torch.Tensor):
        """
        x_batch: Batch(sentence length) * 1 (each char)
        这里直接输出output
        """
        (foutput, boutput), _ = self._feature(x_batch)

        #output = torch.mean([foutput,boutput],dim=1)
        output = torch.cat([foutput, boutput], dim=2).squeeze()
        #print(output.size())
        out = self.embed2out(output)
        out = F.log_softmax(out, dim=1)
        loss = self.loss_fn(out, x_batch.squeeze())
        return loss

    def forward(self, x: torch.Tensor):
        # 协同训练的时候
        """
        这里是外部调用 Batch * 1
        """
        return self._feature(x)


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    emlo = ELMoChar(10,2,3,5)
    emlo.train().to(device)

    optimizer = optim.Adam(emlo.parameters(), lr=0.1, weight_decay=1e-6)

    x = [i for i in range(10)]
    for epoch in range(1):
        x_ts = torch.tensor(x, dtype=torch.long, device=device)
        #print(x_ts.device)
        x_batch = torch.unsqueeze(x_ts,dim=1)
        loss = emlo.loss(x_batch)
        loss.backward()
        optimizer.step()
    '''
    print('complete')
    emlo.eval()
    elmovector = ELMoMixer(emlo,2)
    x_ts = torch.tensor(x, dtype=torch.long, device=device)
    vec = elmovector(x_ts)
    print(vec)
    '''

if __name__ == '__main__':
    test()