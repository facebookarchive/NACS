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
from seq2seq.nn.encoder_rnn import EncoderRNN
from seq2seq.nn.decoder_rnn import DecoderRNN
from seq2seq.nn.decoder_rnn_attention_rl import DecoderRNNAttentionRL

from seq2seq.models.tagger import Tagger
from seq2seq.models.base import BaseModel
from seq2seq.utils import init_gru

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


class Model1(BaseModel):
    """
    Model 1
    Encoder-Decoder with RL in the decoder.
    Warning: the decoder is simpler than Bahdanau.

    Tags are sampled from the hidden state (only).
    Then a word is sampled from the tag + chosen context (e.g. context vector, among other options).

    """

    def __init__(self, n_words_src=0, n_words_trg=0, dim=0, emb_dim=0,
                 n_enc_layers=1, n_dec_layers=1, dropout=0.,
                 emb_dim_tags=0, n_tags_src=0, n_tags_trg=0,
                 predict_src_tags=False, predict_trg_tags=False,
                 enc_type=None, dec_type=None, pass_hidden_state=False,
                 factored_input=False,
                 coeff_ce=0.1, coeff_rl=1.0, coeff_rl_baseline=0.1, coeff_entropy=0.01,
                 n_symbols=20, reward_type='logprob',
                 ctx=True, context_start_iter=8000, vocab_src=None, vocab_trg=None,
                 vocab_tags_src=None, vocab_tags_trg=None,
                 ctx_dropout=0., ctx_dim=0, ctx_gate=False,
                 use_prev_word=True, rnn_type='gru',
                 predict_word_separately=False, use_gold_symbols=False,
                 freeze_symbol=False, entropy_decay=1., entropy_decay_steps=0, ctx_detach=False,
                 symbol_word_gate=False, input_feeding=True,
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

        super(Model1, self).__init__(n_words_src=n_words_src, n_words_trg=n_words_trg, dim=dim,
                                     emb_dim=emb_dim, emb_dim_tags=emb_dim_tags, n_enc_layers=n_enc_layers,
                                     n_dec_layers=n_dec_layers, dropout=dropout,
                                     n_tags_src=n_tags_src,
                                     n_tags_trg=n_tags_trg, predict_src_tags=predict_src_tags,
                                     predict_trg_tags=predict_trg_tags,
                                     pass_hidden_state=pass_hidden_state, vocab_src=vocab_src, vocab_trg=vocab_trg,
                                     vocab_tags_src=vocab_tags_src, vocab_tags_trg=vocab_tags_trg,
                                     rnn_type=rnn_type)

        if enc_type == 'rnn':
            bidirectional = False
            enc_dim = dim
            self.encoder = EncoderRNN(n_words=n_words_src, dim=dim, emb_dim=emb_dim, n_layers=n_enc_layers,
                                      dropout=dropout, bidirectional=bidirectional,
                                      emb_dim_tags=emb_dim_tags, rnn_type=rnn_type,
                                      factored_input=factored_input, n_tags=n_tags_src, vocab=vocab_src)
        elif enc_type == 'birnn':
            bidirectional = True
            enc_dim = dim * 2
            self.encoder = EncoderRNN(n_words=n_words_src, dim=dim, emb_dim=emb_dim, n_layers=n_enc_layers,
                                      dropout=dropout, bidirectional=bidirectional,
                                      emb_dim_tags=emb_dim_tags, rnn_type=rnn_type,
                                      factored_input=factored_input, n_tags=n_tags_src, vocab=vocab_src)
        else:
            raise ValueError('unknown enc_type')

        if dec_type == 'rnn_att':
            self.decoder = DecoderRNNAttentionRL(n_words=n_words_trg, dim=dim, emb_dim=emb_dim, enc_dim=enc_dim,
                                                 n_layers=n_dec_layers,
                                                 dropout=dropout, encoder_is_bidirectional=bidirectional,
                                                 pass_hidden_state=pass_hidden_state, n_symbols=n_symbols,
                                                 ctx=ctx, context_start_iter=context_start_iter,
                                                 vocab=vocab_trg, ctx_dropout=ctx_dropout, ctx_dim=ctx_dim,
                                                 ctx_gate=ctx_gate, n_tags=n_tags_trg,
                                                 use_prev_word=use_prev_word, rnn_type=rnn_type,
                                                 predict_word_separately=predict_word_separately,
                                                 use_gold_symbols=use_gold_symbols,
                                                 ctx_detach=ctx_detach, reward_type=reward_type,
                                                 symbol_word_gate=symbol_word_gate, input_feeding=input_feeding)
        else:
            raise ValueError('unknown dec_type')

        # coefficients for the loss
        self.coeff_ce = coeff_ce
        self.coeff_ce_second = coeff_ce  # FIXME own parameter
        self.coeff_rl = coeff_rl
        self.coeff_rl_baseline = coeff_rl_baseline
        self.coeff_entropy = coeff_entropy
        self.reward_type = reward_type  # prob, logprob, 01, mean_log_prob

        self.current_coeff_entropy = coeff_entropy
        self.entropy_decay = entropy_decay
        self.entropy_decay_steps = entropy_decay_steps

        self.ctx = ctx
        self.ctx_dropout = ctx_dropout
        self.ctx_dim = ctx_dim
        self.ctx_gate = ctx_gate
        self.use_prev_word = use_prev_word
        self.use_gold_symbols = use_gold_symbols
        self.freeze_symbol = freeze_symbol
        self.context_start_iter = context_start_iter

        self.predict_word_separately = predict_word_separately

        logger.warning("reward_type = %s" % self.reward_type)

        logger.info("loss coefficients: CE %f RL %f RL-baseline %f Entropy reg %f" % (
            self.coeff_ce, self.coeff_rl, self.coeff_rl_baseline, self.coeff_entropy))

        self.eval_random_sym = False
        self.eval_argmin_sym = False
        self.feed_original_sym = False

    def forward(self, src_var=None, src_lengths=None, trg_var=None, trg_lengths=None, max_length=0,
                tf_ratio=1., src_tags_var=None, trg_tags_var=None, return_attention=False, return_states=False,
                predict_src_tags=False, predict_trg_tags=False, return_log_probs=False,
                pointer=False, iter_i=1):
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

        input_mask = torch.ne(src_var.cuda(), self.src_pad_idx)
        encoder_outputs, encoder_final, embedded = self.encoder(src_var, src_lengths, tags=src_tags_var)

        attention_values = embedded if pointer else None

        result = self.decoder(
            encoder_final=encoder_final, encoder_outputs=encoder_outputs, encoder_mask=input_mask,
            max_length=trg_max_length, trg_var=trg_var, trg_tags_var=trg_tags_var, tf_ratio=tf_ratio,
            return_attention=return_attention, return_states=return_states, return_log_probs=return_log_probs,
            attention_values=attention_values, encoder_embeddings=embedded, iter_i=iter_i,
            eval_random_sym=self.eval_random_sym, eval_argmin_sym=self.eval_argmin_sym,
            feed_original_sym=self.feed_original_sym)

        # if predict_src_tags:
        #     src_tag_log_probs, src_tag_predictions = self.src_tagger(encoder_outputs)
            # result['src_tag_preds'] = src_tag_predictions
            # result['src_tag_log_probs'] = src_tag_log_probs

        # if predict_trg_tags:
        #     decoder_states = result['states']
        #     trg_tag_log_probs, trg_tag_predictions = self.trg_tagger(decoder_states)  # (B, 1, #Tags)
        #     result['trg_tag_preds'] = trg_tag_predictions
        #     result['trg_tag_log_probs'] = trg_tag_log_probs

        return result
