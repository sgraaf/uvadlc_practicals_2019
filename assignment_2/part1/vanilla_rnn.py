################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        mu = 0
        sigma = 1e-4

        # initialize the weights
        self.Whx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden).normal_(mean=mu, std=sigma).to(self.device))
        self.Whh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden).normal_(mean=mu, std=sigma).to(self.device))
        self.Wph = nn.Parameter(torch.Tensor(self.num_hidden, self.num_classes).normal(mean=mu, std=sigma).to(self.device))

        # initialize the biases
        self.bh = nn.Parameter(torch.zeros(self.num_hidden).to(self.device))
        self.bp = nn.Parameter(torch.zeros(self.num_classes).to(self.device))

        # initialize the tanh activation function
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Implementation here ...
        # initialize the hidden state
        h = torch.zeros(self.num_hidden).to(self.device)
        
        # compute the hidden state
        for i in range(self.seq_length):
            h = self.tanh(
                x[:, i] @ self.Whx +
                h @ self.Whh +
                self.bh
            )

        # compute the output
        p = h @ self.Wph + self.bp

        return p
        
