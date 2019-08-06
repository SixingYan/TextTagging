import torch
from torch import nn
from typing import List

class ELMoMixer(nn.Module):
    """docstring for Mixer"""
    def __init__(self, elmo, num_layer:int,normal_fn=nn.init.xavier_uniform_`):
        super(Mixer, self).__init__()

        self.elmo = elmo
        self.normal_fn = normal_fn
        self.scalar_parameters = ParametersList([
            ])

        self.gamma = Parameter()

    def forward(self,x_batch):
        """"""
        # each layer representation
        _,features=self.elmo(x)

        # scalar_parameters normalization
        # concat to softmax
        norm_scalars = F.softmax(torch.concat([p for p in self.scalar_parameters]),dim=0)
        # split to list
        norm_scalars = torch.split(norm_scalars,split_size_or_sections=1)

        # weight sum
        pieces = []
        for feat, scalar in zip(features, norm_scalars):
            pieces.append(self.normal_fn(feat) * scalar)

        return self.gamma * torch.sum(pieces)

class ELMoChar(nn.Module):
    """ELMoChar: 只针对char级别的训练任务
        所以取消了char-embedding的部分
    """
    def __init__(self, num_layer: int, 
        embed_dim: int, hid_dim: int,
        loss_fn=nn.NLLLoss()):
        super(ELMoChar, self).__init__()
        self.loss_fn = loss_fn
        self.embedding = nn.Embedding(vocab_size,embed_dim)
        
        self.nets = []
        for i in range(num_layer):
            fnn = nn.LSTM(embed_dim,hid_dim,batch_first=True)  # forward
            bnn = nn.LSTM(embed_dim,hid_dim,batch_first=True)  # backward
            self.nets.append([fnn,bnn])

        self.linears = []
        for i in range(num_layer):
            fln = nn.Linear(hid_dim,embed_dim)
            bln = nn.Linear(hid_dim,embed_dim)
            self.linears.append([fln,bln])

        self.embed2out=nn.Linear(embed,vocab_size)

    def _embed(self, x_batch:torch.Tensor):
        """"""
        embed =self.embedding(x_batch)
        return embed # batch_size * 1 * embed_dim

    def _biLM(self, fembed:torch.Tensor,bembed:torch.Tensor):
        """
        embed: batch_size * 1 * embed_dim
        """
        features = []
        for i in range(self.num_layer):
            foutput,_= self.nets[i][0](fembed)
            fembed = self.linears[i][0](foutput)

            boutput,_= self.nets[i][1](bembed)
            bembed = self.linears[i][1](boutput)

            rembed=torch.cat([fembed,bembed])
            features.append(rembed)

        return (foutput,boutput),features


    def _feature(self,x_batch:torch.Tensor):
        """
        x_batch: Batch(sentence length) * 1 (each char)
        """
        fembed = self._embed(x_batch)
        bembed = self._embed(torch.flip(x_batch, (0,)))
        output, features = self._biLM(fembed,bembed)
        return output, features

    def loss(self,x_batch:torch.Tensor):
        """
        x_batch: Batch(sentence length) * 1 (each char)
        这里直接输出output
        """
        (foutput,boutput), _ = self._feature(x_batch) 

        #output = torch.mean([foutput,boutput],dim=1)
        output = torch.concat([foutput,boutput],dim=1)
        out = self.embed2out(output)
        out = F.log_softmax(out, dim=1)
        loss=self.loss_fn(out,true)
        return loss

    def forward(self,x):
        # 协同训练的时候
        """
        这里是外部调用
        """
        if len(x.size()) < :
            x = x.unsqeence(0)
        batch_embed = self._embed(x.unsqeence(0))
        return self._feature(x_batch) 



emlo = ELMoChar()
emlo.eval()
elmovector = ELMoMixer(emlo,)

elmovector(x)





