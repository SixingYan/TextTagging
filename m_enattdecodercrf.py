import torch
from torch import nn
from const import START_TAG, STOP_TAG

class EncoderAttDecoder(nn.Module):

    def __init__(self, vocab_size, tag_to_ix,
                 in_hdim=128, out_hdim=128, de_hdim: int=128, num_layers=2,
                 loss_fn=nn.NLLLoss(),
                 use_crf=True, max_length=1000,
                 bi=2, device='cpu',
                 use_dropout=False, dropout=0.1,
                 teacher_forcing_ratio=0.3, pre_word_embeds=None,
                 use_init=False, init_fns: Dict=None):
        super(EncoderAttDecoder, self).__init__()
        self.bi = bi
        self.device = device
        self.max_length = max_length
        self.loss_fn = loss_fn if use_crf is False else None
        self.num_layers = num_layers
        self.en_in_hdim = in_hdim
        self.en_out_hdim = out_hdim
        self.tag_to_ix = tag_to_ix
        self.tags_size = len(tag_to_ix)
        self.de_hdim = de_hdim
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_crf = use_crf
        if use_dropout:
            self.dropout = nn.Dropout(dropout)

        self.en_wrd_embed = nn.Embedding(vocab_size, self.en_in_hdim)
        if pre_word_embeds is not None:
            self.en_wrd_embed.weight = nn.Parameter(
                torch.FloatTensor(pre_word_embeds))
            self.en_wrd_embed.weight.requires_grad = False
        elif use_init:
            init_fns['embed'](self.en_wrd_embed)
        
        self.use_subembed = use_subembed
        if use_subembed:
            self.sub_embed = nn.Embedding(vocab_size, subembed_dim)
            self.subnn = nn.GRU(subembed_dim, subhid_dim)
            embed_dim += subhid_dim

        self.enrnn = nn.GRU(self.en_in_hdim, self.en_out_hdim, num_layers=num_layers,
                            bidirectional=True if bi == 2 else False)
        if use_init:
            init_fns['rnn'](self.enrnn, 'gru')

        self.de_embed = nn.Embedding(self.tags_size, self.de_hdim)
        if use_init:
            init_fns['embed'](self.de_embed)

        self.attn = nn.Linear(self.de_hdim * 2, self.max_length)

        self.attn_combine = nn.Linear(self.de_hdim * 2, self.de_hdim)

        self.dernn = nn.GRU(self.de_hdim, self.de_hdim)  # , dropout=0.3)
        if use_init:
            init_fns['rnn'](self.rnn, 'gru')

        self.hid2tag = nn.Linear(self.de_hdim, self.tags_size)
        if use_init:
            init_fns['linear'](self.hid2tag)

        if use_init:
            tensor = torch.empty(self.tags_size, self.tags_size, device=self.device)
            init_fns['linear'](tensor)
        else:
            tensor = torch.rand(self.tags_size, self.tags_size, device=self.device)
        self.transitions = nn.Parameter(tensor)
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

    def _encoder(self, x):
        eoutputs = torch.zeros(
            self.max_length, self.en_out_hdim * self.bi, device=self.device)
        ehidden = nn.init.xavier_uniform_(
            torch.zeros(self.bi * self.num_layers, 1,
                        self.en_out_hdim, device=self.device),
            gain=nn.init.calculate_gain('relu'))
        for ei in range(x.size(0)):
            eoutput, ehidden = self._encoder_net(
                x[ei], ehidden)
            eoutputs[ei] = eoutput[0, 0]

        return eoutputs, ehidden

    def _encoder_net(self, sentence, hidden):
        embed = self.en_wrd_embed(sentence).view(1, 1, -1)
        if self.use_dropout:
            self.dropout(embed)
        output, hidden = self.enrnn(embed, hidden)
        if self.use_dropout:
            self.dropout(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden

    def _decoder(self, eoutputs, ehidden, y=None):
        dinput = torch.tensor([[self.tag_to_ix[START_TAG]]], device=self.device)
        if self.bi == 2:
            dhidden = torch.cat(
                [ehidden[0, :, :], ehidden[-1, :, :]], 1).unsqueeze(0)
        else:
            dhidden = ehidden
        doutputs = torch.zeros(
            self.length, self.de_hdim, device=self.device)
        if y is not None:
            for di in range(self.length):
                doutput, dhidden = self._decoder_net(
                    dinput, dhidden, eoutputs)
                dinput = y[di]  # Teacher forcing
                doutputs[di] = doutput[0, 0]
        else:
            for di in range(self.length):
                doutput, decoder_hidden = self._decoder_net(
                    dinput, dhidden, eoutputs)
                topv, topi = doutput.topk(1)
                dinput = topi.squeeze().detach()
                doutputs[di] = doutput[0, 0]
                if dinput.item() == STOP_TAG:
                    break
        return doutputs

    def _decoder_net(self, input, hidden, encoder_outputs):
        embedded = self.de_embed(input).view(1, 1, -1)
        if self.use_dropout:
            self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.dernn(output, hidden)
        if self.use_dropout:
            self.dropout(output)

        output = F.log_softmax(self.hid2tag(output[0]), dim=1)
        return output, hidden

    def _crf(self, feats):
        return self._viterbi_decode(feats)

    def _forward_alg(self, feats):
        init_alphas = torch.full((1, self.tags_size), -10000., device=self.device)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        forward_var = init_alphas
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

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1, device=self.device)
        tags = torch.cat(
            [torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long, device=self.device), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score += self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _viterbi_decode(self, feats):
        backpointers = []
        init_vvars = torch.full((1, self.tags_size), -10000., device=self.device)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []
            for next_tag in range(self.tags_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)
        terminal_var = forward_var + \
            self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return best_path

    def _get_feat(self, x, y=None, use_tf=False):
        self.length = x.size(0)
        eoutputs, ehidden = self._encoder(x)
        if use_tf and random.random() < self.teacher_forcing_ratio:
            doutputs = self._decoder(eoutputs, ehidden, y)
        else:
            doutputs = self._decoder(eoutputs, ehidden)
        output = self.hid2tag(doutputs)
        return output

    def _nll_loss(self, output, y):
        loss = 0
        for i in range(y.size(0)):
            loss += self.loss_fn(output[i].unsqueeze(0), y[i].unsqueeze(0))
        return loss

    def loss(self, x, y):
        output = self._get_feat(x, y, use_tf=True)
        if self.use_crf:
            return self._neg_log_likelihood(output, y)
        else:
            return self._nll_loss(output, y)

    def forward(self, x):
        output = self._get_feat(x)
        if self.use_crf:
            tag_seq = self._crf(output)
        else:
            tag_seq = torch.argmax(output, dim=1).cpu().numpy()
        return tag_seq
