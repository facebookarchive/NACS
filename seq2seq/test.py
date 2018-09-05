# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import logging
import torch
import subprocess
import re
import math
import pdb
import numpy as np
from torch import nn
from torchtext import data, datasets

from seq2seq.utils import EOS_TOKEN, PAD_TOKEN, get_fields, build_model
from seq2seq.nn.translationdataset import FactoredTranslationDataset
from seq2seq.evaluation import evaluate, evaluate_bleu, evaluate_exact_match, evaluate_all

logger = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()


def predict_and_save(path=None, **kwargs):
    """
    Predict and save predictions to a file
    """
    with open(path, mode='w', encoding='utf-8') as f:
        for words in predict(**kwargs):
            line = " ".join(words) + "\n"
            f.write(line)


def predict(model=None, batch_iter=None, src_vocab=None, trg_vocab=None, max_length=0):
    """
    Predict (without knowing the targets)
    """
    assert max_length > 0, "max_length should be bigger than zero"

    model.eval()
    for batch in iter(batch_iter):

        src_var, src_lengths = batch.src
        src_tags_var = batch.src_tags if hasattr(batch, 'src_tags') else None

        src_lengths = src_lengths.view(-1).tolist()

        result = model(src_var=src_var, src_lengths=src_lengths, max_length=max_length, src_tags_var=src_tags_var)
        predictions = result['preds']

        eos_index = trg_vocab.stoi[EOS_TOKEN]

        for sentence in predictions.cpu().numpy():
            eos_idx = np.where(sentence == eos_index)[0]
            cutoff = eos_idx[0] if len(eos_idx) > 0 else len(sentence)
            words = [trg_vocab.itos[i] for i in sentence[:cutoff]]
            yield words


def predict_single_batch(model=None, batch=None, max_length=0, return_attention=False,
                         predict_src_tags=False, predict_trg_tags=False):
    """
    Predict a single batch (the first of the iterator) and return translations
    """
    assert max_length > 0, "max_length should be bigger than zero"

    model.eval()

    src_var, src_lengths = batch.src
    src_tags_var = batch.src_tags if hasattr(batch, 'src_tags') else None
    trg_tags_var = batch.trg_tags if hasattr(batch, 'trg_tags') else None

    src_lengths = src_lengths.view(-1).tolist()

    result = model(src_var=src_var, src_lengths=src_lengths, max_length=max_length,
                   src_tags_var=src_tags_var, trg_tags_var=trg_tags_var,
                   return_attention=return_attention,
                   predict_src_tags=predict_src_tags, predict_trg_tags=predict_trg_tags)

    return result


def test(model_type=None, enc_type=None, dec_type=None, src=None, trg=None, root=None, train=None, validation=None,
         test=None, src_tags='', trg_tags='', workdir=None, emb_dim=0, dim=0, dropout=0., word_dropout=0.,
         weight_decay=0., learning_rate=0., batch_size=1, n_iters=10000, save_every=0, print_every=0,
         plot_every=0, eval_every=0, tf_ratio=1., resume="", max_length=0, seed=0, clip=5., metric="",
         emb_dim_tags=0, optimizer='adam', n_enc_layers=1, n_dec_layers=1, n_val_examples=5,
         use_visdom=False, mtl=False, coeff_ce=0., coeff_rl=0., coeff_rl_baseline=0.,
         coeff_entropy=0., n_symbols=0, reward_type='logprob', ctx=True, context_start_iter=8000, save_heatmaps=False,
         save_heatmap_animations=False, unk_src=True, unk_trg=True, ckpt='', src_vocab=None, trg_vocab=None,
         src_tags_vocab=None, trg_tags_vocab=None,
         pass_hidden_state=True,
         predict_from_emb=False, predict_from_ctx=False, predict_from_dec=False,
         dec_input_emb=False, dec_input_ctx=False,
         **kwargs):
    """
    Load a model and predict
    """
    print(ckpt)
    print(workdir)

    cfg = locals().copy()

    assert ckpt is not None, "ckpt must be given"
    assert src_vocab is not None, "src_vocab must be given"
    assert trg_vocab is not None, "trg_vocab must be given"

    assert os.path.isfile(os.path.join(workdir, ckpt)), "ckpt %s not found in workdir %s" % (ckpt, workdir)
    assert os.path.isfile(os.path.join(workdir, src_vocab)), "src_vocab %s not found in workdir" % src_vocab
    assert os.path.isfile(os.path.join(workdir, trg_vocab)), "src_vocab %s not found in workdir" % src_vocab

    if src_tags:
        assert src_tags_vocab is not None, "src_tags_vocab must be given"
        assert os.path.isfile(os.path.join(workdir, src_tags_vocab)), \
            "src_tags_vocab %s not found in workdir" % src_tags_vocab

    factored_input = True if src_tags and emb_dim_tags > 0 else False
    predict_src_tags = True if mtl and src_tags and emb_dim_tags == 0 else False
    predict_trg_tags = True if mtl and trg_tags and model_type == "encdec" else False

    # if not os.path.exists(workdir):
    #     os.makedirs(workdir)

    use_src_tags = True if src_tags or emb_dim_tags > 0 else False
    use_trg_tags = True if trg_tags else False
    task_title = "_".join(root.split("/")[-2:])

    fields, exts = get_fields(src=src, trg=trg, src_tags=src_tags, trg_tags=trg_tags, unk_src=unk_src, unk_trg=unk_trg)

    src_field = fields[0][1]
    trg_field = fields[1][1]
    src_tags_field = fields[2][1] if use_src_tags else None
    trg_tags_field = fields[-1][1] if use_trg_tags else None

    assert test is not None, "must provide test data set"

    test_data = FactoredTranslationDataset(os.path.join(root, test), exts=exts, fields=fields)
    test_iter = data.BucketIterator(test_data, batch_size=batch_size, train=False, sort_within_batch=True,
                                    sort=True, shuffle=False, sort_key=lambda x: len(x.src))

    logger.info("Test data size: %d" % len(test_data))

    logger.info("Loading checkpoint..")
    checkpoint = torch.load(os.path.join(workdir, ckpt))

    src_field.vocab = torch.load(os.path.join(workdir, src_vocab))
    trg_field.vocab = torch.load(os.path.join(workdir, trg_vocab))

    if use_src_tags:
        src_tags_field.vocab = torch.load(os.path.join(workdir, src_tags_vocab))
        n_tags_src = len(src_tags_field.vocab)
    else:
        n_tags_src = None

    if use_trg_tags:
        trg_tags_field.vocab = torch.load(os.path.join(workdir, trg_tags_vocab))

    n_tags_trg = len(trg_tags_field.vocab) if use_trg_tags else 0

    # model creation
    # TODO could get config from checkpoint['cfg'] instead of commandline
    # TODO this code is WIP
    model = build_model(n_words_src=len(src_field.vocab), n_words_trg=len(trg_field.vocab),
                        predict_src_tags=predict_src_tags, predict_trg_tags=predict_trg_tags,
                        factored_input=factored_input,
                        vocab_src=src_field.vocab, vocab_trg=trg_field.vocab,
                        n_tags_src=n_tags_src, n_tags_trg=n_tags_trg,
                        vocab_tags_src=src_tags_field.vocab if src_tags_field is not None else None,
                        vocab_tags_trg=trg_tags_field.vocab if trg_tags_field is not None else None,
                        **cfg)
    # load parameters
    model.load_state_dict(checkpoint['state_dict'])
    logger.info("Loaded checkpoint '{}' (iter {})".format(ckpt, checkpoint['iter']))

    # internal evaluation: validation perpexity, accuracy and exact match
    acc, ppx, exact_match, bleu = evaluate_all(
        model=model, batch_iter=test_iter,
        src_vocab=src_field.vocab, trg_vocab=trg_field.vocab,
        src_vocab_tags=src_tags_field.vocab if src_tags_field is not None else None,
        trg_vocab_tags=trg_tags_field.vocab if trg_tags_field is not None else None,
        max_length=max_length)

    logger.info({'test_acc': acc, 'test_ppx': ppx, 'test_exact_match': exact_match, 'test_bleu': bleu})
