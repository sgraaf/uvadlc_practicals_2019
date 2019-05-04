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


class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(LSTM, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        # initialize the weights and biases for the input modulation gate
        self.Wgx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden))
        self.Wgh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden))
        self.bg = nn.Parameter(torch.zeros(self.num_hidden))

        # initialize the weights and biases for the input gate
        self.Wix = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden))
        self.Wih = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden))
        self.bi = nn.Parameter(torch.zeros(self.num_hidden))

        # initialize the weights and biases for the forget gate
        self.Wfx = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden))
        self.Wfh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden))
        self.bf = nn.Parameter(torch.zeros(self.num_hidden))

        # initialize the weights and biases for the output gate
        self.Wox = nn.Parameter(torch.Tensor(self.input_dim, self.num_hidden))
        self.Woh = nn.Parameter(torch.Tensor(self.num_hidden, self.num_hidden))
        self.bo = nn.Parameter(torch.zeros(self.num_hidden))

        # intialize the weights and biases for the output
        self.Wph = nn.Parameter(torch.Tensor(self.num_hidden, self.num_classes))
        self.bp = nn.Parameter(torch.zeros(self.num_classes))

        # actually initialize the weights
        for parameter in [self.Wgx, self.Wgh, self.Wix, self.Wih, self.Wfx, self.Wfh, self.Wox, self.Woh, self.Wph]:
            torch.nn.init.xavier_normal_(parameter)

        # initialize the activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # send to device
        self.to(device)

    def forward(self, x):
        # Implementation here ...
        # initialize the hidden state(s)
        h = torch.zeros((self.batch_size, self.num_hidden), device=self.device)
        c = torch.zeros((self.batch_size, self.num_hidden), device=self.device)

        # compute the hidden state
        for i in range(self.seq_length):
            xi = x[:, i].view(self.batch_size, self.input_dim)

            g = self.tanh(
                xi @ self.Wgx +
                h @ self.Wgh +
                self.bg
            )
            i = self.sigmoid(
                xi @ self.Wix +
                h @ self.Wih +
                self.bi
            )
            f = self.sigmoid(
                xi @ self.Wfx +
                h @ self.Wfh +
                self.bf
            )
            o = self.sigmoid(
                xi @ self.Wox +
                h @ self.Woh +
                self.bo
            )

            c = g * i + c * f
            h = self.tanh(c) * o

        # compute the output
        p = h @ self.Wph + self.bp

        return p
