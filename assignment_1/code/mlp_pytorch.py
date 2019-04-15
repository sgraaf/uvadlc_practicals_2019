"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    TODONE:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(MLP, self).__init__()
    if len(n_hidden) > 0:
      # add modules for the first hidden layer (from x to h_1)
      self.modules = [
        nn.Linear(n_inputs, n_hidden[0]),
        nn.ReLU()
      ]

      # add modules for the intermediate hidden layers (from h_2 to h_n-1)
      for i in range(len(n_hidden) - 1):
        self.modules += [
          nn.Linear(n_hidden[i], n_hidden[i + 1]),
          nn.ReLU()
        ]

      # add modules for the last hidden layer (from h_n to y)
      self.modules += [
        nn.Linear(n_hidden[-1], n_classes),
      ]
    else:
      # no hidden units, so SLP w/ SoftMax instead of MLP
      self.modules = [
        nn.Linear(n_inputs, n_classes),
      ]
    self.model = nn.Sequential(*self.modules)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODONE:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.model(x)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
