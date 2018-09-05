# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import pdb
import random
from seq2seq.utils import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN

use_cuda = torch.cuda.is_available()


class DecoderRNN(nn.Module):
    def __init__(self, dim=0, emb_dim=0, enc_dim=0, n_words=0, n_layers=1, dropout=0.1,
                 encoder_is_bidirectional=False, vocab=None, rnn_type='gru'):

        super(DecoderRNN, self).__init__()

        assert n_words > 0, "n_words must be bigger than 0 (vocabulary size)"
        assert dim > 0, "dim must be bigger than 0 (hidden layer size)"
        assert emb_dim > 0, "emb_dim must be bigger than 0 (embedding size)"

        self.dim = dim
        self.emb_dim = emb_dim
        self.n_words = n_words
        self.n_layers = n_layers
        self.dropout = dropout
        self.encoder_is_bidirectional = encoder_is_bidirectional
        self.rnn_type = rnn_type
        self.enc_dim = enc_dim
        self.embedding = nn.Embedding(n_words, emb_dim, padding_idx=vocab.stoi[PAD_TOKEN])

        if rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim, dim, batch_first=True, dropout=dropout,
                              num_layers=n_layers)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim, dim, batch_first=True, dropout=dropout,
                              num_layers=n_layers)
        else:
            self.rnn = nn.LSTM(emb_dim, dim, batch_first=True, dropout=dropout,
                               num_layers=n_layers)

        # projected final encoder state
        self.enc_to_dec_h = []
        self.enc_to_dec_c = []

        for _ in range(n_layers):

            layer = nn.Linear(enc_dim, dim, bias=True)
            self.enc_to_dec_h.append(layer)

            if rnn_type == 'lstm':
                layer = nn.Linear(enc_dim, dim, bias=True)
                self.enc_to_dec_c.append(layer)

        self.enc_to_decs_h = nn.ModuleList(self.enc_to_dec_h)
        self.enc_to_decs_c = nn.ModuleList(self.enc_to_dec_c)

        # output layer from context vector and current decoder state to n_words
        self.output_layer = nn.Linear(dim, n_words, bias=False)
        self.dropout = nn.Dropout(self.dropout)

        self.pad_idx = vocab.stoi[PAD_TOKEN]
        self.unk_idx = vocab.stoi[UNK_TOKEN]
        self.eos_idx = vocab.stoi[EOS_TOKEN]

        self.criterion = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

    def init_hidden(self, encoder_final):
        """
        initialize hidden state [n_layers, B, D]
        """

        if isinstance(encoder_final, tuple):  # lstm
            hiddens = []
            cells = []
            encoder_final_h, encoder_final_c = encoder_final

            for h, layer in zip(encoder_final_h, self.enc_to_dec_h):
                h_init = torch.tanh(layer(h))
                hiddens.append(h_init)

            for c, layer in zip(encoder_final_c, self.enc_to_dec_c):
                c_init = torch.tanh(layer(c))
                cells.append(c_init)

            hiddens = torch.stack(hiddens, dim=0)
            cells = torch.stack(cells, dim=0)
            hidden = (hiddens, cells)

        else:  # gru or rnn
            hiddens = []
            for h, layer in zip(encoder_final, self.enc_to_dec_h):
                h_init = torch.tanh(layer(h))
                hiddens.append(h_init)

            hidden = torch.stack(hiddens, dim=0)

        return hidden

    def forward(self, encoder_final=None, max_length=0, trg_var=None, tf_ratio=0.5, encoder_outputs=None,
                encoder_mask=None, return_states=False, return_log_probs=False, **kwargs):
        """
        encoder_final : last encoder hidden states (fwd, bwd concat) (B, 1, 2D)
        max_length : maximum number of decoding steps
        trg_var : variable with target indexes for teacher forcing
        """

        assert max_length > 0, "provide max length somehow"
        batch_size = encoder_outputs.size(0)

        # we indicate start-of-sequence with an embedding of all zeros
        embedded = Variable(torch.zeros((batch_size, 1, self.emb_dim)))
        embedded = embedded.cuda() if use_cuda else embedded
        hidden = self.init_hidden(encoder_final)  # (n_layers, B, D)

        all_predictions = torch.zeros(max_length, batch_size).long()  # [T, B]
        all_predictions = all_predictions.cuda() if use_cuda else all_predictions

        all_log_probs = []

        # to store masks
        mask = torch.zeros(max_length, batch_size).byte()  # [T, B]
        mask = mask.cuda() if use_cuda else mask

        # mask everything after </s> was generated
        current_mask = torch.ones(batch_size).byte()
        current_mask = current_mask.cuda() if use_cuda else current_mask

        use_tf = True if random.random() < tf_ratio else False

        # apply GRU to the output so far
        for i in range(max_length):

            mask[i] = current_mask
            _, hidden = self.rnn(embedded, hidden)  # hidden (n_layers, B, D)

            # predict from (top) RNN hidden state directly
            to_predict = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, V]

            logits = self.output_layer(to_predict.unsqueeze(1))  # (B, 1, V)
            log_probs = F.log_softmax(logits, 2)

            all_log_probs.append(log_probs.squeeze(1))
            predictions = log_probs.max(2)[1]  # [B, 1]
            all_predictions[i] = predictions.squeeze(1).data

            current_mask = current_mask * (predictions != self.eos_idx).squeeze(1).data

            if trg_var is not None and use_tf:  # teacher forcing, feed true targets to next step
                targets_this_iter = trg_var[:, i, None]       # (B, 1)
                embedded = self.embedding(targets_this_iter)  # (B, 1, E)
                embedded = self.dropout(embedded)
            else:  # feed current predictions to next step
                embedded = self.embedding(predictions)   # (B, 1, E)
                embedded = self.dropout(embedded)

        all_log_probs = torch.stack(all_log_probs, dim=0)  # (T, B, V)
        all_predictions = all_predictions.transpose(0, 1).contiguous()  # (B, T)

        loss = None
        if trg_var is not None:
            loss = self.get_loss(log_probs=all_log_probs, trg_var=trg_var, mask=mask)
            loss = dict(loss=loss)

        if not return_log_probs:
            all_log_probs = None

        return dict(preds=all_predictions, log_probs=all_log_probs, mask=mask, loss=loss)

    def get_loss(self, log_probs=None, trg_var=None, mask=None):

        trg_var = trg_var.transpose(0, 1).contiguous()  # make time-major [T, B]

        trg_mask = (trg_var != self.pad_idx)
        mask = mask & trg_mask  # mask if predicted sequence longer than target sequence
        padding_mask = ~mask  # this is positive after </s> predicted

        time, batch_size, voc_size = log_probs.size()  # time-major!
        log_probs_2d = log_probs.contiguous().view(-1, voc_size)

        loss = self.criterion(log_probs_2d, trg_var.view(-1))
        loss = loss.contiguous().view([time, batch_size])
        loss = loss.masked_fill(padding_mask, 0.)
        loss = loss.sum() / batch_size

        return loss

