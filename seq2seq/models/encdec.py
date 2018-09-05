# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from seq2seq.nn.encoder_rnn import EncoderRNN
from seq2seq.nn.decoder_rnn import DecoderRNN
from seq2seq.nn.decoder_rnn_attention import DecoderRNNAttention

from seq2seq.models.base import BaseModel


use_cuda = torch.cuda.is_available()


class EncoderDecoder(BaseModel):
    """
    Encoder-Decoder with Attention
    """

    def __init__(self, n_words_src=0, n_words_trg=0, dim=0, emb_dim=0,
                 n_enc_layers=1, n_dec_layers=1, dropout=0.,
                 enc_type=None, dec_type=None,
                 pass_hidden_state=True,
                 vocab_src=None, vocab_trg=None,
                 rnn_type=None,
                 predict_from_emb=False, predict_from_ctx=False,
                 predict_from_dec=False,
                 dec_input_emb=False, dec_input_ctx=False,
                 ctx_dim=0,
                 tf_ratio=1.0,
                 **kwargs):
        """
        Initializer.

        Args:
            n_words_src:
            n_words_trg:
            dim:
            emb_dim:
            n_enc_layers:
            n_dec_layers:
            dropout:
            bidirectional:
        """

        super(EncoderDecoder, self).__init__(
            n_words_src=n_words_src, n_words_trg=n_words_trg, dim=dim,
            emb_dim=emb_dim,
            n_enc_layers=n_enc_layers,
            n_dec_layers=n_dec_layers, dropout=dropout,
            pass_hidden_state=pass_hidden_state,
            vocab_src=vocab_src,
            vocab_trg=vocab_trg)

        bidirectional = False
        enc_dim = dim

        if enc_type == 'rnn':
            self.encoder = EncoderRNN(
                n_words=n_words_src, dim=dim, emb_dim=emb_dim,
                n_layers=n_enc_layers, dropout=dropout,
                bidirectional=bidirectional, vocab=vocab_src, rnn_type=rnn_type)
        elif enc_type == 'birnn':
            bidirectional = True
            enc_dim = dim * 2
            self.encoder = EncoderRNN(
                n_words=n_words_src, dim=dim, emb_dim=emb_dim,
                n_layers=n_enc_layers, dropout=dropout,
                bidirectional=bidirectional, vocab=vocab_src, rnn_type=rnn_type)
        else:
            raise ValueError('unknown enc_type')

        if dec_type == 'rnn':
            self.decoder = DecoderRNN(
                n_words=n_words_trg, dim=dim, emb_dim=emb_dim,
                n_layers=n_dec_layers, dropout=dropout,
                vocab=vocab_trg, rnn_type=rnn_type, enc_dim=enc_dim)
        elif dec_type == 'rnn_att':
            self.decoder = DecoderRNNAttention(
                n_words=n_words_trg, dim=dim, emb_dim=emb_dim,
                n_layers=n_dec_layers, dropout=dropout,
                pass_hidden_state=pass_hidden_state, vocab=vocab_trg,
                rnn_type=rnn_type, ctx_dim=ctx_dim, enc_dim=enc_dim,
                predict_from_emb=predict_from_emb,
                predict_from_ctx=predict_from_ctx,
                predict_from_dec=predict_from_dec,
                dec_input_emb=dec_input_emb,
                dec_input_ctx=dec_input_ctx,
                tf_ratio=tf_ratio)

        else:
            raise ValueError('unknown dec_type')

    def forward(self, src_var=None, src_lengths=None, trg_var=None, trg_lengths=None, max_length=0,
                tf_ratio=1., src_tags_var=None, return_attention=False,
                predict_src_tags=False, predict_trg_tags=False, return_states=False,
                pointer=False, iter_i=1, trg_tags_var=None):
        """

        Args:
            src_var:
            src_lengths:
            trg_var:
            trg_lengths:
            max_length: required when trg_var is not given
            tf_ratio:
            src_tags_var: variable with source tags (must have same shape as src_var)
            return_attention: return attention scores
            predict_src_tags: also predict source-side tags (1 per encoder state)
            predict_trg_tags: also predict target-side tags (1 per decoder state)
            pointer: use pointer network (apply attention to source word embeddings)

        Returns:

        """

        trg_max_length = trg_var.size(1) if trg_var is not None else max_length
        assert trg_max_length > 0, "max length should be >0"

        input_mask = torch.ne(src_var, self.src_pad_idx)
        encoder_outputs, encoder_final, embedded = self.encoder(src_var, src_lengths, tags=src_tags_var)

        attention_values = embedded if pointer else None

        result = self.decoder(
            encoder_final=encoder_final, encoder_outputs=encoder_outputs,
            encoder_mask=input_mask,
            max_length=trg_max_length, trg_var=trg_var, tf_ratio=tf_ratio,
            return_attention=return_attention,
            attention_values=attention_values, return_states=return_states)

        return result
