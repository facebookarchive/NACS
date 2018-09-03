# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np


class Statistics:

    def __init__(self, name="train", metric='bleu'):

        self.name = name
        self.metric = metric

        self.best = np.inf if metric == 'perplexity' else -1.
        self.best_iter = 0
        self.best_idx = 0

        self.best_ppx = 0.
        self.best_acc = 0.
        self.best_bleu = 0.
        self.best_em = 0.

        self.acc_loss = 0.  # accumulated loss in between plots

        self.loss = []
        self.perplexity = []
        self.accuracy = []
        self.bleu = []
        self.exact_match = []

        self.eval_iters = []

    def add(self, acc, ppx, em, bleu, iter_i):
        self.accuracy.append(acc)
        self.perplexity.append(ppx)
        self.exact_match.append(em)
        self.bleu.append(bleu)

        # set eval_score
        if self.metric == "perplexity":
            eval_score = ppx
            minimize_metric = True
        elif self.metric == "accuracy":
            eval_score = acc
            minimize_metric = False
        elif self.metric == "exact_match":
            eval_score = em
            minimize_metric = False
        elif self.metric == "bleu":
            eval_score = bleu
            minimize_metric = False
        else:
            raise ValueError("Unknown evaluation metric.")

        # save best model
        comp = np.less if minimize_metric else np.greater
        if comp(eval_score, self.best):
            self.best = eval_score
            self.best_iter = iter_i
            self.best_idx = len(self.perplexity) - 1
            self.best_ppx = ppx
            self.best_bleu = bleu
            self.best_em = em
            self.best_acc = acc
            return True
        else:
            return False
