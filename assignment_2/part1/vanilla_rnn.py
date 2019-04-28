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
        sigma = 1e-2

        # initialize the weights
        self.Whx = nn.Parameter(mu + sigma * torch.randn(self.input_dim, self.num_hidden))
        self.Whh = nn.Parameter(mu + sigma * torch.randn(self.num_hidden, self.num_hidden))
        self.Wph = nn.Parameter(mu + sigma * torch.randn(self.num_hidden, self.num_classes))

        # initialize the biases
        self.bh = nn.Parameter(torch.zeros((self.num_hidden, 1)))
        self.bp = nn.Parameter(torch.zeros((self.num_classes, 1)))

        # initialize the tanh activation function
        self.tanh = nn.Tanh()


    def forward(self, x):
        # Implementation here ...
        # initialize the hidden state
        h = torch.zeros(self.batch_size, self.num_hidden, device=self.device)
        
        # compute the hidden state
        for i in range(self.seq_length):
            xi = x[:, i].view(-1, self.input_dim)
            
            h = self.tanh(
                xi @ self.Whx +
                h @ self.Whh +
                self.bh
            )

        # compute the output
        p = h @ self.Wph + self.bp

        return p
        
