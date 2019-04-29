# MIT License
#
# Copyright (c) 2017 Tom Runia
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

from __future__ import absolute_import, division, print_function

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel
from utils import accuracy, one_hot

################################################################################

# make the results directory (if it doesn't exist)
RESULTS_DIR = Path.cwd() / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def train(config):
    # Initialize the device which to run the model on
    if config.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device(config.device)
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(
        filename=config.txt_file,
        seq_length=config.seq_length
    )
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        vocabulary_size=dataset.vocab_size,
        lstm_num_hidden=config.lstm_num_hidden,
        lstm_num_layers=config.lstm_num_layers,
        device=device
    )
    model.to(device)

    # Setup the loss and optimizer
    criterion =nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(
        model.parameters(), 
        config.learning_rate
    )

    results = {
        'step': [],
        'accuracy': [],
        'loss': []
    }

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        #######################################################
        # Add more code here ...
        #######################################################
        # put the model in training model
        model.train()

        # convert the data and send to device
        X = torch.stack(batch_inputs, dim=1)
        X = one_hot(X, dataset.vocab_size)
        X = X.to(device)
        
        Y = torch.stack(batch_targets, dim=1)
        Y = Y.to(device)

        # (re)set the optimizer gradient to 0
        optimizer.zero_grad()

        # forward pass the mini-batch
        Y_pred, _ = model(X)

        # compute the accuracy and the loss and append them
        accuracy = accuracy(Y_pred, Y)
        loss = criterion.forward(Y_pred.transpose(2, 1), Y)

        results['step'].append(step)
        results['accuracy'].append(accuracy)
        results['loss'].append(loss.item())
        
        # backwards propogate the loss
        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % config.print_every == 0:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M")}] Train Step {step:04d}/{config.train_steps:04d}, Batch Size = {config.batch_size}, Examples/Sec = {examples_per_second:.2f}, Accuracy = {accuracy:.2f}, Loss = {loss:.3f}')

        if step == config.sample_every:
            # Generate some sentences by sampling from the model
            pass

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (ie. "cuda" or "cpu")')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')

    config = parser.parse_args()

    # Train the model
    train(config)
