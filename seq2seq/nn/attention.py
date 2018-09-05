# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pdb


class BahdanauAttention(nn.Module):
    """
    Computes Bahdanau attention between a memory (e.g. encoder states)
    and a query (e.g. a decoder state)
    """

    def __init__(self, query_dim=0, memory_dim=0):
        """
        Initialize the attention mechanism.

        Args:
            query_dim:  dimensionality of the query
            memory_dim: dimensionality of the memory (e.g. encoder states)
        """
        super(BahdanauAttention, self).__init__()
        self.query_dim = query_dim
        self.memory_dim = memory_dim
        self.memory_layer = nn.Linear(memory_dim, query_dim, bias=False)
        self.query_layer = nn.Linear(query_dim, query_dim, bias=False)
        self.energy_layer = nn.Linear(query_dim, 1, bias=False)

    def project_memory(self, memory):
        """
        Simply applies a learned transformation over the given memory tensor
        Args:
            memory:

        Returns:
            projected memory

        """
        return self.memory_layer(memory)

    def forward(self, query=None, projected_memory=None, mask=None):
        proj_query = self.query_layer(query)            # (1, B, D)
        proj_query = proj_query.transpose(0, 1)         # (B, 1, D)

        # this broadcasts the query over the projected memory
        energy = torch.tanh(proj_query + projected_memory)  # (B, T, D)
        energy = self.energy_layer(energy).squeeze(2)   # (B, T)

        # mask illegal attention values
        pad_mask = (mask == 0)
        energy = energy.masked_fill(pad_mask, -1e3)
        energy = F.softmax(energy, 1)    # (B, T)
        energy = energy.unsqueeze(1)      # (B, 1, T)

        return energy
