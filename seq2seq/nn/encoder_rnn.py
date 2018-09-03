# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from seq2seq.utils import PAD_TOKEN, UNK_TOKEN, EOS_TOKEN

use_cuda = torch.cuda.is_available()
logger = logging.getLogger(__name__)


class EncoderRNN(nn.Module):
    """
    Simple encoder module
    """
    def __init__(self, n_words=0, dim=0, emb_dim=0, n_layers=1, dropout=0., bidirectional=True,
                 emb_dim_tags=0, n_tags=0, factored_input=False, vocab=None, rnn_type=None):
        super(EncoderRNN, self).__init__()

        assert n_words > 0, "n_words must be bigger than 0 (vocabulary size)"
        assert dim > 0, "dim must be bigger than 0 (hidden layer size)"
        assert emb_dim > 0, "emb_dim must be bigger than 0 (embedding size)"

        self.n_layers = n_layers
        self.dim = dim
        self.emb_dim = emb_dim
        self.emb_dim_factor = emb_dim_tags
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.factored_input = factored_input
        self.rnn_type = rnn_type

        self.vocab = vocab
        self.pad_idx = vocab.stoi[PAD_TOKEN]
        self.unk_idx = vocab.stoi[UNK_TOKEN]
        self.eos_idx = vocab.stoi[EOS_TOKEN]

        self.embedding = nn.Embedding(n_words, emb_dim, padding_idx=vocab.stoi[PAD_TOKEN])

        if self.factored_input:
            self.embedding_tags = nn.Embedding(n_tags, emb_dim_tags)

        self.emb_dropout_layer = nn.Dropout(dropout)
        self.dropout_layer = nn.Dropout(dropout)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(emb_dim + emb_dim_tags, dim, batch_first=True,
                              bidirectional=bidirectional, dropout=dropout, num_layers=n_layers)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(emb_dim + emb_dim_tags, dim, batch_first=True, nonlinearity='tanh',
                              bidirectional=bidirectional, dropout=dropout, num_layers=n_layers)
        else:  # lstm
            self.rnn = nn.LSTM(emb_dim + emb_dim_tags, dim, batch_first=True,
                               bidirectional=bidirectional, dropout=dropout, num_layers=n_layers)

    def forward(self, x, input_lengths, tags=None):
        """
        x : inputs, shape: [B, T] (LongTensor)
        input_lengths : real lengths of input batch (list)
        """

        embedded = self.embedding(x)

        if self.factored_input:
            embedded_factor = self.embedding_tags(tags)
            embedded = torch.cat([embedded, embedded_factor], dim=2)

        embedded = self.emb_dropout_layer(embedded)

        packed = pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.rnn(packed)

        # outputs [B, T, 2D]
        outputs, output_lengths = pad_packed_sequence(outputs, batch_first=True)

        # concatenate final hidden states ([n_layers*n_directions], B, D])
        if self.bidirectional:
            if isinstance(hidden, tuple):  # lstm
                hidden_h = [torch.cat((hidden[0][i], hidden[0][i + 1]), dim=1) for i in
                            range(0, self.n_layers * 2, 2)]
                hidden_h = torch.stack(hidden_h, dim=0)
                hidden_c = [torch.cat((hidden[1][i], hidden[1][i + 1]), dim=1) for i in
                            range(0, self.n_layers * 2, 2)]
                hidden_c = torch.stack(hidden_c, dim=0)
                hidden = (hidden_h, hidden_c)
            else:
                hidden = [torch.cat((hidden[i], hidden[i + 1]), dim=1) for i in
                          range(0, self.n_layers * 2, 2)]
                hidden = torch.stack(hidden, dim=0)

        return outputs, hidden, embedded
