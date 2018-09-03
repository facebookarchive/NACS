# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


use_cuda = torch.cuda.is_available()


class Tagger(nn.Module):
    """
    A (very) simple tagger
    """

    def __init__(self, dim=0, n_tags=0):
        super(Tagger, self).__init__()
        self.tag_layer = nn.Linear(dim, n_tags, bias=False)

    def forward(self, hidden):
        """
        Args:
            hidden: [X x dim] or [X x Y x dim]

        Returns:
            log_probs: [X x (Y x) n_tags]
            predictions: [X x (Yx) 1]
        """
        logits = self.tag_layer(hidden)  # (B, 1, #Tags)
        log_probs = F.log_softmax(logits, dim=-1)
        predictions = log_probs.max(-1)[1]

        return log_probs, predictions
