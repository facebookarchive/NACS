# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import logging
from seq2seq.nn.attention import BahdanauAttention
from seq2seq.utils import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


class DecoderRNNAttention(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, dim=0, emb_dim=0, enc_dim=0, n_words=0, n_layers=1, dropout=0.1, attention=None,
                 n_tags=0, pass_hidden_state=False, vocab=None,
                 rnn_type='gru', ctx_dim=0,
                 dec_input_emb=False, dec_input_ctx=False,
                 predict_from_emb=False, predict_from_ctx=False, predict_from_dec=False,
                 ctx_gate=False, tf_ratio=1.,
                 ):

        super(DecoderRNNAttention, self).__init__()

        assert n_words > 0, "n_words must be bigger than 0 (vocabulary size)"
        assert dim > 0, "dim must be bigger than 0 (hidden layer size)"
        assert emb_dim > 0, "emb_dim must be bigger than 0 (embedding size)"

        self.dim = dim
        self.emb_dim = emb_dim
        self.enc_dim = enc_dim
        self.n_words = n_words
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention = attention
        self.n_tags = n_tags
        self.pass_hidden_state = pass_hidden_state
        self.rnn_type = rnn_type
        self.tf_ratio = tf_ratio

        self.dec_input_emb = dec_input_emb
        self.dec_input_ctx = dec_input_ctx

        self.predict_from_emb = predict_from_emb
        self.predict_from_ctx = predict_from_ctx
        self.predict_from_dec = predict_from_dec

        self.ctx_gate = ctx_gate

        self.vocab = vocab
        self.pad_idx = vocab.stoi[PAD_TOKEN]
        self.unk_idx = vocab.stoi[UNK_TOKEN]
        self.eos_idx = vocab.stoi[EOS_TOKEN]
        self.ctx_dim = ctx_dim

        # need to predict from something
        assert predict_from_dec or predict_from_ctx or predict_from_emb, \
            "Must use at least one of 'predict_from_dec, 'predict_from_ctx', 'predict_from_emb'"

        # this will kill a combination we don't want early
        if predict_from_emb:
            assert dec_input_emb, "must also feed emb if predicting from emb (this is to limit combinations)"
            assert predict_from_dec, "must also predict from decoder (this is to limit combinations)"

        self.criterion = nn.NLLLoss(reduction='none', ignore_index=self.pad_idx)

        self.embedding = nn.Embedding(n_words, emb_dim, padding_idx=vocab.stoi[PAD_TOKEN])

        proj_ctx_dim = enc_dim if ctx_dim == 0 else ctx_dim

        rnn_input_dim = proj_ctx_dim if dec_input_ctx else 0
        if dec_input_emb:
            rnn_input_dim += emb_dim

        if rnn_type == 'gru':
            self.rnn = nn.GRU(rnn_input_dim, dim, batch_first=True, dropout=dropout, num_layers=n_layers)
        elif rnn_type == 'rnn':
            self.rnn = nn.RNN(rnn_input_dim, dim, batch_first=True, dropout=dropout, num_layers=n_layers,
                              nonlinearity='tanh')
        else:
            self.rnn = nn.LSTM(rnn_input_dim, dim, batch_first=True, dropout=dropout, num_layers=n_layers)

        self.attention = BahdanauAttention(query_dim=dim, memory_dim=enc_dim)

        # projected final encoder state
        self.enc_to_dec_h = []
        self.enc_to_dec_c = []

        if pass_hidden_state:
            for _ in range(n_layers):

                layer = nn.Linear(enc_dim, dim, bias=True)
                self.enc_to_dec_h.append(layer)

                if rnn_type == 'lstm':
                    layer = nn.Linear(enc_dim, dim, bias=True)
                    self.enc_to_dec_c.append(layer)

            self.enc_to_decs_h = nn.ModuleList(self.enc_to_dec_h)
            self.enc_to_decs_c = nn.ModuleList(self.enc_to_dec_c)

        pre_output_input_dim = 0

        if predict_from_emb:
            pre_output_input_dim += emb_dim

        if predict_from_ctx:
            pre_output_input_dim += proj_ctx_dim

        if predict_from_dec:
            pre_output_input_dim += dim

        self.ctx_proj = nn.Linear(enc_dim, ctx_dim, bias=False) if ctx_dim > 0 else None
        self.pre_output_layer = nn.Linear(pre_output_input_dim, dim, bias=False)

        self.output_layer = nn.Linear(dim, n_words, bias=False)

        self.emb_dropout = nn.Dropout(self.dropout)
        self.to_predict_dropout = nn.Dropout(self.dropout)

    def init_hidden(self, encoder_final):
        """
        initialize hidden state [n_layers, B, D]
        """

        if not self.pass_hidden_state:

            if isinstance(encoder_final, tuple):  # lstm
                batch_size = encoder_final[0][-1].size(0)
            else:
                batch_size = encoder_final[-1].size(0)

            hidden = Variable(torch.zeros(self.n_layers, batch_size, self.dim))
            hidden = hidden.cuda() if use_cuda else hidden

            if self.rnn_type == 'lstm':
                return hidden, hidden

            return hidden

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

            # copy initial state if we have more layers than the encoder
            if self.n_layers > encoder_final_h.size(0):
                hidden = (torch.cat([hiddens, hiddens], dim=0),
                          torch.cat([cells, cells], dim=0))

        else:  # gru or rnn
            hiddens = []
            for h, layer in zip (encoder_final, self.enc_to_dec_h):
                h_init = torch.tanh(layer(h))
                hiddens.append(h_init)

            hidden = torch.stack(hiddens, dim=0)

        return hidden

    def forward(self, encoder_outputs=None, encoder_final=None, encoder_mask=None, max_length=0, trg_var=None,
                tf_ratio=1., return_attention=False, attention_values=None, return_states=False,
                trg_tags_var=None, return_log_probs=False):
        """
        Forward decoding pass.

        Args:
            encoder_outputs: encoder hidden states (B, T, 2D)
            encoder_final: encoder final state (forward, backward concatenated)
            encoder_mask: mask of actual encoder positions (B, T)
            max_length: maximum number of decoding steps (for prediction)
            attention_values: values to apply attention to, e.g. encoder embeddings (defaults to encoder_outputs if None)
            trg_var: variable with target indexes for teacher forcing
            tf_ratio: teacher forcing ratio (1.0 = always teacher forcing)
            return_attention: return the attention scores
            return_states: return decoder states

        Returns:

        """
        assert max_length > 0, "provide max length somehow"
        batch_size, enc_time, ctx_dim = encoder_outputs.size()

        # pre-compute attention memory (projected encoder states)
        projected_memory = self.attention.project_memory(encoder_outputs)

        # set attention_values to the word embeddings to get a Pointer network (Vinyas, 2015)
        attention_values = encoder_outputs if attention_values is None else attention_values

        # we indicate start-of-sequence with an embedding of all zeros
        embedded = Variable(torch.zeros((batch_size, 1, self.emb_dim)))
        embedded = embedded.cuda() if use_cuda else embedded

        hidden = self.init_hidden(encoder_final)  # (n_layers, B, D) or (2, n_layers_, B, D) for lstm

        # to store log probs
        all_log_probs = []
        all_src_gates = []
        all_trg_gates = []

        # to store attention scores
        all_attention_scores = torch.zeros(max_length, batch_size, enc_time) if return_attention else None  # [T, B, T']
        if use_cuda and return_attention:
            all_attention_scores = all_attention_scores.cuda()

        # to store RNN states
        decoder_states = Variable(torch.zeros(max_length, batch_size, self.dim)) if return_states else None  # [T, B, D]
        if use_cuda and return_states:
            decoder_states = decoder_states.cuda()

        all_predictions = torch.zeros(max_length, batch_size).long()  # [T, B]
        all_predictions = all_predictions.cuda() if use_cuda else all_predictions

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

            # compute context vector using attention
            query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
            if len(query.size()) == 2:
                query = query.unsqueeze(0)  # [1, B, 2D]
            alpha = self.attention(query=query, projected_memory=projected_memory, mask=encoder_mask)  # (B, 1, T)
            if return_attention:
                all_attention_scores[i] = alpha.squeeze(1).data

            context = alpha.bmm(attention_values)  # (B, 1, 2D)
            prev_state = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
            prev_state = prev_state.unsqueeze(1)

            # downscale context to dim
            if self.ctx_dim > 0:
                context = self.ctx_proj(context)

            rnn_input_emb = embedded
            rnn_input_ctx = context

            if self.dec_input_emb:
                rnn_input = torch.cat((rnn_input_emb, rnn_input_ctx), 2)
            else:
                rnn_input = rnn_input_ctx

            _, hidden = self.rnn(rnn_input, hidden)  # hidden (n_layers, B, D)
            current_state = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]

            if return_states:
                decoder_states[i] = current_state

            pre_output_inputs = []

            if self.predict_from_emb:
                pre_output_inputs.append(embedded)

            if self.predict_from_dec:
                pre_output_inputs.append(current_state.unsqueeze(1))

            if self.predict_from_ctx:
                pre_output_inputs.append(context)

            to_predict = torch.cat(pre_output_inputs, 2)
            to_predict = self.to_predict_dropout(to_predict)
            to_predict = self.pre_output_layer(to_predict)  # (B, 1, D)

            logits = self.output_layer(to_predict)          # (B, 1, V)

            log_probs = F.log_softmax(logits, 2)
            all_log_probs.append(log_probs.squeeze(1))
            predictions = log_probs.max(2)[1]  # [B, 1]
            all_predictions[i] = predictions.squeeze(1).data

            current_mask = current_mask * (predictions != self.eos_idx).squeeze(1).data

            if trg_var is not None and use_tf:  # teacher forcing, feed true targets to next step
                targets_this_iter = trg_var[:, i, None]       # (B, 1)
                embedded = self.embedding(targets_this_iter)  # (B, 1, E)
                embedded = self.emb_dropout(embedded)
            else:  # feed current predictions to next step
                embedded = self.embedding(predictions)   # (B, 1, E)
                embedded = self.emb_dropout(embedded)

        # return everything batch-first
        all_log_probs = torch.stack(all_log_probs, dim=0)
        all_predictions = all_predictions.transpose(0, 1).contiguous()  # (B, T)

        if return_states:
            decoder_states = decoder_states.transpose(0, 1).contiguous()  # (T, B, D) -> (B, T, D)

        if return_attention:
            all_attention_scores = all_attention_scores.transpose(0, 1)  # (T, B, T') -> (B, T, T')

        loss = None
        if trg_var is not None:
            loss = self.get_loss(log_probs=all_log_probs,
                                 trg_var=trg_var, mask=mask)
            loss = dict(loss=loss)

        result = dict(preds=all_predictions, loss=loss,
                      att_scores=all_attention_scores,
                      states=decoder_states, mask=mask)
        return result

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
