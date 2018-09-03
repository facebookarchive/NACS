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
from torch.distributions import Categorical
import random
import logging
import math
from seq2seq.nn.attention import BahdanauAttention
from seq2seq.utils import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


class DecoderRNNAttentionRL(nn.Module):
    """
    Decoder with attention
    """
    def __init__(self, dim=0, emb_dim=0, enc_dim=0, n_words=0, n_layers=1, dropout=0.1, attention=None,
                 encoder_is_bidirectional=True, n_tags=0,
                 n_symbols=20, pass_hidden_state=False,
                 ctx=True, context_start_iter=8000, vocab=None, ctx_dropout=0., ctx_dim=0,
                 use_prev_word=True, predict_word_separately=False, rnn_type='gru',
                 use_gold_symbols=False, ctx_gate=False, ctx_detach=False,
                 reward_type='prob',
                 coeff_ce=0.1, coeff_rl=1.0, coeff_rl_baseline=0.1, coeff_entropy=0.01,
                 entropy_decay=1., entropy_decay_steps=100, symbol_word_gate=False,
                 input_feeding=True):

        super(DecoderRNNAttentionRL, self).__init__()

        assert n_words > 0, "n_words must be bigger than 0 (vocabulary size)"
        assert dim > 0, "dim must be bigger than 0 (hidden layer size)"
        assert emb_dim > 0, "emb_dim must be bigger than 0 (embedding size)"

        self.dim = dim
        self.emb_dim = emb_dim
        self.n_words = n_words
        self.n_layers = n_layers
        self.n_symbols = n_symbols
        self.dropout = dropout
        self.attention = attention
        self.n_tags = n_tags
        self.pass_hidden_state = pass_hidden_state
        self.rnn_type = rnn_type
        self.reward_type = reward_type
        self.input_feeding = input_feeding

        self.coeff_ce = coeff_ce
        self.coeff_rl = coeff_rl
        self.coeff_rl_baseline = coeff_rl_baseline
        self.coeff_entropy = coeff_entropy
        self.entropy_decay = entropy_decay
        self.entropy_decay_steps = entropy_decay_steps

        self.ctx = ctx
        self.ctx_dropout = ctx_dropout
        self.use_prev_word = use_prev_word
        self.ctx_dim = ctx_dim  # this is not the original context dim, but the downscaled dim
        self.ctx_gate = ctx_gate
        self.ctx_detach = ctx_detach
        self.vocab = vocab
        self.predict_word_separately = predict_word_separately
        self.use_gold_symbols = use_gold_symbols
        self.symbol_word_gate = symbol_word_gate

        self.pad_idx = vocab.stoi[PAD_TOKEN]
        self.unk_idx = vocab.stoi[UNK_TOKEN]
        self.eos_idx = vocab.stoi[EOS_TOKEN]

        if use_gold_symbols:
            assert n_tags == n_symbols, 'n_tags must equal n_symbols when using tags as gold symbols'

        self.context_start_iter = context_start_iter

        self.embedding = nn.Embedding(n_words, emb_dim)
        self.symbol_embedding = nn.Embedding(n_symbols, emb_dim)

        if rnn_type == 'gru':
            self.rnn = nn.GRU(enc_dim + emb_dim, dim, batch_first=True, dropout=dropout, num_layers=n_layers)
        else:
            self.rnn = nn.LSTM(enc_dim + emb_dim, dim, batch_first=True, dropout=dropout, num_layers=n_layers)

        self.attention = BahdanauAttention(query_dim=dim, memory_dim=enc_dim)

        if pass_hidden_state:
            for _ in range(n_layers):

                layer = nn.Linear(enc_dim, dim, bias=True)
                self.enc_to_dec_h.append(layer)

                if rnn_type == 'lstm':
                    layer = nn.Linear(enc_dim, dim, bias=True)
                    self.enc_to_dec_c.append(layer)

            self.enc_to_decs_h = nn.ModuleList(self.enc_to_dec_h)
            self.enc_to_decs_c = nn.ModuleList(self.enc_to_dec_c)

        self.symbol_output_layer = nn.Linear(dim, n_symbols)

        if self.ctx_gate:
            self.ctx_gate_layer = nn.Linear(dim, 1)

        if self.symbol_word_gate:
            self.symbol_word_gate_layer = nn.Linear(dim, 1)

        if ctx_dim > 0:
            self.ctx_to_pre_output = nn.Linear(enc_dim, ctx_dim, bias=False)  # context, sym -> output layer

        self.ctx_to_predict_dropout = nn.Dropout(ctx_dropout)

        if ctx:
            logger.warning("predicting from symbols AND context vector")
            logger.warning("ctx: %s" % ctx)
            logger.warning("context_start_iter: %d" % context_start_iter)
            logger.warning('ctx_dropout %f' % ctx_dropout)
            logger.warning('ctx_dim %d (downscaled ctx to predict from)' % ctx_dim)
            logger.warning('use_prev_word %s' % use_prev_word)

            pre_output_input_dim = emb_dim + ctx_dim
            if use_prev_word:
                pre_output_input_dim += emb_dim
            if ctx_dim == 0:
                pre_output_input_dim += enc_dim

            self.word_output_layer = nn.Linear(emb_dim, n_words, bias=False)
            self.pre_output_layer = nn.Linear(pre_output_input_dim, emb_dim, bias=False)  # context, sym -> output layer

        else:
            logger.warning("predicting from symbol only")
            self.word_output_layer = nn.Linear(emb_dim, n_words)

        if predict_word_separately:
            self.sec_symbol_embedding = nn.Embedding(n_symbols, emb_dim)
            ctx_proj_dim = ctx_dim if ctx_dim > 0 else enc_dim
            self.sec_pre_output_layer = nn.Linear(ctx_proj_dim + emb_dim, dim, bias=False)
            self.sec_word_output_layer = nn.Linear(dim, n_words, bias=False)

        self.reinforce_fv_inner_layer = nn.Linear(dim, dim)
        self.reinforce_fv_layer = nn.Linear(dim, 1)

        self.emb_dropout = nn.Dropout(self.dropout)
        self.to_predict_dropout = nn.Dropout(self.dropout)

        self.criterion = nn.NLLLoss(reduce=False, size_average=False, ignore_index=self.pad_idx)

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
                h_init = F.tanh(layer(h))
                hiddens.append(h_init)

            for c, layer in zip(encoder_final_c, self.enc_to_dec_c):
                c_init = F.tanh(layer(c))
                cells.append(c_init)

            hiddens = torch.stack(hiddens, dim=0)
            cells = torch.stack(cells, dim=0)
            hidden = (hiddens, cells)

            # copy initial state if we have more layers than the encoder
            # TODO can handle this in a more principled way
            if self.n_layers > encoder_final_h.size(0):
                hidden = (torch.cat([hiddens, hiddens], dim=0), torch.cat([cells, cells], dim=0))

        else:  # gru or rnn
            hiddens = []
            for h, layer in zip (encoder_final, self.enc_to_dec_h):
                h_init = F.tanh(layer(h))
                hiddens.append(h_init)

            hidden = torch.stack(hiddens, dim=0)

        return hidden

    def forward(self, encoder_outputs=None, encoder_final=None, encoder_mask=None, max_length=0, trg_var=None,
                tf_ratio=1., return_attention=False, attention_values=None, return_states=False,
                encoder_embeddings=None, iter_i=1, eval_random_sym=False, eval_argmin_sym=False,
                feed_original_sym=False, trg_tags_var=None, return_log_probs=False):
        """
        Forward decoding pass.

        RL notes:
        - Predict a "symbol" from hidden state (only)
        - Predict a word from that symbol, and ...?

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
            encoder_embeddings:
            iter_i:
            eval_random_sym:
            eval_argmin_sym:
            feed_original_sym:

        Returns:

        """
        assert max_length > 0, "provide max length somehow"
        batch_size = encoder_outputs.size(0)

        if self.use_gold_symbols:
            max_length = trg_tags_var.size(1)

        use_tf = True if random.random() < tf_ratio else False

        # pre-compute attention memory (projected encoder states)
        projected_memory = self.attention.project_memory(encoder_outputs)

        # set attention_values to the word embeddings to get a Pointer network (Vinyals et al., 2015)
        attention_values = encoder_outputs if attention_values is None else attention_values

        embedded = Variable(torch.zeros((batch_size, 1, self.emb_dim)))
        embedded = embedded.cuda() if use_cuda else embedded

        # we indicate start-of-sequence with an embedding of all zeros
        embedded_symbols = Variable(torch.zeros((batch_size, 1, self.emb_dim)))
        embedded_symbols = embedded_symbols.cuda() if use_cuda else embedded_symbols

        hidden = self.init_hidden(encoder_final)  # (n_layers, B, D)

        all_predictions = []
        all_log_probs = []
        all_attention_scores = []
        all_decoder_states = []
        all_fvs = []
        all_symbols = []
        all_symbol_log_probs = []
        masks = []
        all_sec_log_probs = []
        all_sec_predictions = []
        all_gates = []
        all_symbol_word_gates = []

        # mask everything after </s> was generated
        mask = Variable(torch.ones([batch_size, 1]).byte())
        mask = mask.cuda() if use_cuda else mask

        # apply GRU to the output so far
        for i in range(max_length):

            masks.append(mask)

            # compute context vector using attention
            query = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]
            query = query.unsqueeze(0)  # [1, B, D]
            alpha = self.attention(query=query, projected_memory=projected_memory, mask=encoder_mask)  # (B, 1, T)
            if return_attention:
                all_attention_scores.append(alpha)
            context = alpha.bmm(attention_values)  # (B, 1, 2D)

            if self.input_feeding:
                rnn_input = torch.cat((embedded_symbols, context), 2)  # condition on prev. symbol, state
            else:
                rnn_input = embedded_symbols

            _, hidden = self.rnn(rnn_input, hidden)  # hidden (n_layers, B, D)

            current_state = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]  # [B, D]

            if return_states:
                all_decoder_states.append(current_state.unsqueeze(0))

            # use GOLD "symbols" (using trg_tags_var) instead of whatever we would have predicted
            if self.use_gold_symbols:
                symbols = trg_tags_var[:, i].unsqueeze(1)  # (B, 1)
                all_symbols.append(symbols)
                embedded_symbols = self.symbol_embedding(symbols)
            else:
                symbol_logits = self.symbol_output_layer(current_state)  # (B, n_symbols)
                symbol_log_probs = F.log_softmax(symbol_logits, -1)      # (B, n_symbols)

                # value prediction
                fv = self.reinforce_fv_layer(
                    F.relu(self.reinforce_fv_inner_layer(current_state.detach())))
                all_fvs.append(fv)

                m = Categorical(probs=torch.exp(symbol_log_probs))  # (B, 1) TODO in pytorch 0.4 directly use logits
                if self.training:
                    symbols = m.sample()                       # (B) "RL action"
                else:
                    _, symbols = m.probs.max(-1)

                symbols = symbols.unsqueeze(1)                 # (B, 1)

                all_symbols.append(symbols)
                all_symbol_log_probs.append(symbol_log_probs.unsqueeze(1))  # add (B, 1, n_symbols)

                # now predict a word from these symbols
                # We use cross-entropy loss as the reward.
                embedded_symbols = self.symbol_embedding(symbols)  # (B, 1, E)

            # predict current word from the sampled symbol and the context vector
            # there are many more options here

            if eval_random_sym:
                predict_symbols = Variable(torch.LongTensor(batch_size).random_(self.n_symbols).unsqueeze(1))
                predict_symbols = predict_symbols.cuda() if use_cuda else predict_symbols
                embedded_symbols_predict = self.symbol_embedding(predict_symbols)  # (B, 1, E)
                embedded_symbols_predict = self.emb_dropout(embedded_symbols_predict)

                if not feed_original_sym:
                    embedded_symbols = embedded_symbols_predict

            elif eval_argmin_sym:
                _, predict_symbols = m.probs.min(-1)
                predict_symbols = predict_symbols.unsqueeze(1)
                embedded_symbols_predict = self.symbol_embedding(predict_symbols)  # (B, 1, E)
                embedded_symbols_predict = self.emb_dropout(embedded_symbols_predict)

                if not feed_original_sym:
                    embedded_symbols = embedded_symbols_predict

            else:
                predict_symbols = symbols
                embedded_symbols_predict = self.emb_dropout(embedded_symbols)

            # originally we predict from embedded_symbols and context vector
            if self.ctx:

                assert not (eval_random_sym and eval_argmin_sym), "only 1 of random or argmin can be true"

                context = context.detach() if self.ctx_detach else context
                ctx_proj = self.ctx_to_pre_output(context) if self.ctx_dim > 0 else context
                ctx_proj = self.ctx_to_predict_dropout(ctx_proj)

                if self.ctx_gate:
                    gate = F.sigmoid(self.ctx_gate_layer(current_state))
                    all_gates.append(gate)
                    gate = gate.unsqueeze(2)
                    embedded_symbols_predict = embedded_symbols_predict * gate
                    ctx_proj = ctx_proj * (1.-gate)

                if self.use_prev_word:
                    to_predict = torch.cat([embedded_symbols_predict, embedded, ctx_proj], dim=2)
                else:
                    to_predict = torch.cat([embedded_symbols_predict, ctx_proj], dim=2)

                to_predict = self.pre_output_layer(to_predict)
            else:
                to_predict = embedded_symbols_predict

            logits = self.word_output_layer(to_predict)
            log_probs = F.log_softmax(logits, dim=2)

            all_log_probs.append(log_probs)
            predictions = log_probs.max(2)[1]  # [B, 1]

            if trg_var is not None and use_tf:  # teacher forcing, feed true targets to next step
                targets_this_iter = trg_var[:, i, None]       # (B, 1)
                embedded = self.embedding(targets_this_iter)  # (B, 1, E)
                embedded = self.emb_dropout(embedded)
            else:  # feed current predictions to next step
                embedded = self.embedding(predictions)   # (B, 1, E)
                embedded = self.emb_dropout(embedded)

            # symbol / word mix
            if self.symbol_word_gate:

                symbol_word_gate = self.symbol_word_gate_layer(current_state)

                if self.training:
                    noise = torch.Tensor(symbol_word_gate.size()).normal_(mean=0, std=0.1)
                    noise = noise.cuda() if use_cuda else noise
                else:
                    noise = 0.

                symbol_word_gate = F.sigmoid(symbol_word_gate + noise)
                all_symbol_word_gates.append(symbol_word_gate)
                symbol_word_gate = symbol_word_gate.unsqueeze(2)
                symbol_word_mix = symbol_word_gate * embedded_symbols_predict + (1-symbol_word_gate) * embedded
                embedded_symbols = symbol_word_mix  # feed as input to next time step

            mask = (predictions != self.eos_idx) * mask
            all_predictions.append(predictions)

            # predict word also with additional network
            # embedded_symbols are (B, 1, E)
            # context is (B, 1, 2D)
            if self.predict_word_separately:

                sec_embedded_symbols = self.sec_symbol_embedding(predict_symbols)  # (B, 1, E)

                ctx_proj = self.ctx_to_pre_output(context.detach()) if self.ctx_dim > 0 else context.detach()
                ctx_proj = self.ctx_to_predict_dropout(ctx_proj)

                if self.ctx_gate:
                    gate = F.sigmoid(self.ctx_gate_layer(current_state))
                    all_gates.append(gate)
                    gate = gate.unsqueeze(2)
                    sec_embedded_symbols = sec_embedded_symbols * gate
                    ctx_proj = ctx_proj * (1.-gate)

                sec_context = self.ctx_to_predict_dropout(ctx_proj)
                sec_pre_output = self.sec_pre_output_layer(torch.cat([sec_context, sec_embedded_symbols], dim=2))  # [B, 1, D]
                sec_pre_output = F.relu(sec_pre_output)
                sec_word_logits = self.sec_word_output_layer(sec_pre_output)
                sec_word_log_probs = F.log_softmax(sec_word_logits, dim=-1)
                all_sec_log_probs.append(sec_word_log_probs)
                sec_predictions = sec_word_log_probs.max(2)[1]  # [B, 1]
                all_sec_predictions.append(sec_predictions)

        predictions = torch.cat(all_predictions, 1).data  # (B, T)
        log_probs = torch.cat(all_log_probs, 1)      # (B, T, V)
        symbols = torch.cat(all_symbols, 1)          # (B, T)
        mask = torch.cat(masks, 1)

        fvs = None
        sec_log_probs = None
        att_scores = None
        decoder_states = None
        sec_predictions = None
        symbol_log_probs = None
        gates = None
        symbol_word_gates = None

        if len(all_fvs) > 0:
            fvs = torch.cat(all_fvs, 1)

        if len(all_symbol_log_probs) > 0:
            symbol_log_probs = torch.cat(all_symbol_log_probs, 1)

        if self.predict_word_separately:
            sec_log_probs = torch.cat(all_sec_log_probs, 1)      # second network word log probs
            sec_predictions = torch.cat(all_sec_predictions, 1)  # second network word predictions
            predictions = sec_predictions                        # override predictions

        if return_states:
            decoder_states = torch.cat(all_decoder_states, 0)      # (T, B, D)
            decoder_states = decoder_states.transpose(0, 1)    # (T, B, D) -> (B, T, D)

        if return_attention:
            att_scores = torch.cat(all_attention_scores, 1).data  # (B, T', T)

        if self.ctx_gate:
            gates = torch.cat(all_gates, 1)

        if self.symbol_word_gate:
            symbol_word_gates = torch.cat(all_symbol_word_gates, 1)

        loss = None
        if trg_var is not None:
            loss = self.get_loss(log_probs=log_probs, trg_var=trg_var, mask=mask,
                                 fv=fvs, symbols=symbols, symbol_log_probs=symbol_log_probs,
                                 sec_log_probs=sec_log_probs, iter_i=iter_i, symbol_word_gates=symbol_word_gates)

        if not return_log_probs:
            log_probs = None

        result = dict(preds=predictions, loss=loss,
                      att_scores=att_scores,
                      states=decoder_states, mask=mask,
                      symbols=symbols, symbol_log_probs=symbol_log_probs,
                      sec_predictions=sec_predictions, gates=gates, symbol_word_gates=symbol_word_gates,
                      log_probs=log_probs)

        return result

    def get_loss(self, log_probs=None, fv=None, symbols=None, symbol_log_probs=None,
                 trg_var=None, mask=None, iter_i=0, gates=None, sec_log_probs=None, symbol_word_gates=None):

        """For this model loss consists of CE loss, RL loss and RL baseline loss"""

        batch_size = trg_var.size(0)

        trg_mask = (trg_var != self.pad_idx)  # mask is positive for padding locations (BASED ON GOLD TARGETS/REFERENCE)
        trg_lengths = trg_mask.float().sum(dim=1)
        padding_mask = (trg_var == self.pad_idx)

        # mask = mask * trg_mask  # mask if longer than target sequence
        # padding_mask = 1-mask  # this is positive after </s> predicted

        predicted_lengths = mask.float().sum(dim=1)
        predicted_lengths = torch.min(predicted_lengths, trg_lengths)

        # cross entropy loss
        time = log_probs.size(1)
        voc_size = log_probs.size(2)

        log_probs_2d = log_probs.view(-1, voc_size)
        ce_loss = self.criterion(log_probs_2d, trg_var.view(-1))
        ce_loss = ce_loss.view([batch_size, time])
        ce_loss = ce_loss.masked_fill(padding_mask, 0.)
        ce_loss = ce_loss.sum() / batch_size

        if self.use_gold_symbols:
            # only use cross-entropy loss
            return dict(loss=ce_loss)

        # collect rewards

        if symbol_log_probs is not None:
            used_symbol_logprobs = torch.gather(symbol_log_probs, 2, symbols.unsqueeze(2))  # (B, T, 1)

        fv = fv.unsqueeze(2)  # (B, T, 1)

        # choose RL reward
        if self.reward_type == 'prob':
            reward_nograd = torch.gather(log_probs.detach(), 2, trg_var.unsqueeze(2))
            reward_nograd = torch.exp(reward_nograd)
            reward_nograd = reward_nograd.masked_fill(padding_mask.unsqueeze(2), 0.)

        elif self.reward_type == 'logprob':
            reward_nograd = torch.gather(log_probs.detach(), 2, trg_var.unsqueeze(2))
            reward_nograd = reward_nograd.masked_fill(padding_mask.unsqueeze(2), 0.)

        elif self.reward_type == 'mean_log_prob':
            reward_nograd = torch.gather(log_probs.detach(), 2, trg_var.unsqueeze(2)).squeeze(2)
            reward_nograd = reward_nograd.sum(dim=1) / predicted_lengths
            reward_nograd = reward_nograd.unsqueeze(1).expand([batch_size, time]).unsqueeze(2)

        elif self.reward_type == '01':
            reward = (symbols == trg_var).float()  # 1 for correct, 0 for incorrect
            reward = reward.masked_fill(padding_mask, 0.)
            reward = reward.unsqueeze(2)
            reward_nograd = reward
        else:
            raise ValueError("unknown reward type")

        # FIXME try without this
        # reward_nograd = 0. * reward_nograd + ((reward_nograd.sum(1) / predicted_lengths.unsqueeze(1)).unsqueeze(1).expand([batch_size, time, 1]))

        # reinforce loss for sampled actions/symbols
        reinforce_loss = -(reward_nograd - fv.detach())
        reinforce_loss_ = reinforce_loss / (torch.std(reinforce_loss, dim=1, keepdim=True).clamp(min=-1., max=1.))
        reward_min_baseline_var = (reinforce_loss ** 2)

        reinforce_loss = reinforce_loss_ * used_symbol_logprobs
        reinforce_loss = reinforce_loss.squeeze(2).masked_fill(padding_mask, 0.)
        reinforce_loss = reinforce_loss.sum() / batch_size

        # monitor variance in reward-baseline
        reward_min_baseline_var = reward_min_baseline_var.squeeze(2).masked_fill(padding_mask, 0.)
        reward_min_baseline_var = reward_min_baseline_var.sum() / predicted_lengths.sum()

        # baseline loss (to train value function fv)
        baseline_loss = torch.pow(reward_nograd - fv, 2)
        baseline_loss = baseline_loss.squeeze(2).masked_fill(padding_mask, 0.)
        baseline_loss = baseline_loss.sum() / batch_size

        # entropy regularization
        symbol_entropy_reg = symbol_log_probs * torch.exp(symbol_log_probs)
        symbol_entropy_reg = symbol_entropy_reg.masked_fill(padding_mask.unsqueeze(2), 0.)
        symbol_entropy_reg = symbol_entropy_reg.sum()

        mean_reward = reward_nograd.sum() / predicted_lengths.sum()

        coeff_entropy = self.coeff_entropy * (self.entropy_decay ** (iter_i/self.entropy_decay_steps))
        self.current_coeff_entropy = coeff_entropy
        coeff_entropy = Variable(torch.FloatTensor([coeff_entropy]))[0]
        coeff_entropy = coeff_entropy.cuda() if use_cuda else coeff_entropy

        loss = self.coeff_ce * ce_loss + self.coeff_rl * reinforce_loss + self.coeff_rl_baseline * baseline_loss + \
            coeff_entropy * symbol_entropy_reg

        losses = dict(loss=loss, ce_loss=ce_loss, reinforce_loss=reinforce_loss,
                      reinforce_baseline_loss=baseline_loss,
                      first_reward=reward_nograd[0][0][0], symbol_entropy_reg=symbol_entropy_reg,
                      reward_min_baseline_var=reward_min_baseline_var, mean_reward=mean_reward,
                      coeff_entropy=coeff_entropy)

        # additional loss for second word prediction
        if self.predict_word_separately:
            if iter_i > self.context_start_iter:

                sec_log_probs_2d = sec_log_probs.view(-1, voc_size)
                sec_ce_loss = self.criterion(sec_log_probs_2d, trg_var.view(-1))
                sec_ce_loss = sec_ce_loss.view([batch_size, time])
                sec_ce_loss = sec_ce_loss.masked_fill(padding_mask, 0.)
                sec_ce_loss = sec_ce_loss.sum() / batch_size

                if self.freeze_symbol:
                    loss = self.coeff_ce_second * sec_ce_loss  # this is the only loss
                    losses['loss'] = loss
                    losses = dict(loss=loss)
                else:
                    loss = loss + self.coeff_ce_second * sec_ce_loss  # train both parts
                    losses['loss'] = loss

                losses['sec_ce_loss'] = sec_ce_loss

        if self.ctx_gate:
            mean_gate = gates.masked_fill(padding_mask, 0.)
            mean_gate = mean_gate.sum() / predicted_lengths.sum()
            losses['mean_gate'] = mean_gate

        if self.symbol_word_gate:
            mean_symbol_word_gate = symbol_word_gates.masked_fill(padding_mask, 0.)
            mean_symbol_word_gate = mean_symbol_word_gate.sum() / predicted_lengths.sum()
            losses['mean_symbol_word_gate'] = mean_symbol_word_gate

        return losses
