# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import logging
import torch.nn as nn
from seq2seq.nn.encoder_rnn import EncoderRNN
from seq2seq.nn.decoder_rnn_attention import DecoderRNNAttention
from seq2seq.utils import PAD_TOKEN, UNK_TOKEN

use_cuda = torch.cuda.is_available()
logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """
    Base model
    Has support for taggers on source and target side, regardless of the chosen enc/dec
    """

    def __init__(self, n_words_src=0, n_words_trg=0, dim=0, emb_dim=0,
                 n_enc_layers=1, n_dec_layers=1,
                 dropout=0.,
                 emb_dim_tags=0,
                 pass_hidden_state=True,
                 vocab_src=None, vocab_trg=None,
                 rnn_type='gru'):

        super(BaseModel, self).__init__()

        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.dim = dim
        self.emb_dim = emb_dim
        self.emb_dim_tags = emb_dim_tags  # for factored input

        self.n_words_src = n_words_src
        self.n_words_trg = n_words_trg

        self.dropout = dropout

        self.vocab_src = vocab_src
        self.vocab_trg = vocab_trg

        self.rnn_type = rnn_type
        self.pass_hidden_state = pass_hidden_state

        self.src_tagger = None
        self.trg_tagger = None

        self.src_pad_idx = vocab_src.stoi[PAD_TOKEN]
        self.src_unk_idx = vocab_src.stoi[UNK_TOKEN]
        self.trg_pad_idx = vocab_trg.stoi[PAD_TOKEN]
        self.trg_unk_idx = vocab_trg.stoi[UNK_TOKEN]

        self.criterion = nn.NLLLoss(reduce=False, size_average=False,
                                    ignore_index=self.trg_pad_idx)

    def forward(self):
        raise NotImplementedError('needs to be overridden')
