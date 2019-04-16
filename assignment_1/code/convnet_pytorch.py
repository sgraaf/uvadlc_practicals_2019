"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from torch import nn

class ConvNet(nn.Module):
  """
  This class implements a Convolutional Neural Network in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an ConvNet object can perform forward.
  """

  def __init__(self, n_channels, n_classes):
    """
    Initializes ConvNet object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
                 
    
    TODONE:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    super(ConvNet, self).__init__()
    self.module_1 = nn.Sequential(
      nn.Conv2d(n_channels, 64, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.MaxPool2d((3, 3), stride=2, padding=1)
    )
    self.module_2 = nn.Sequential(
      nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.MaxPool2d((3, 3), stride=2, padding=1)
    )
    self.module_3 = nn.Sequential(
      nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(256),
      nn.MaxPool2d((3, 3), stride=2, padding=1)
    )
    self.module_4 = nn.Sequential(
      nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.MaxPool2d((3, 3), stride=2, padding=1)
    )
    self.module_5 = nn.Sequential(
      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
      nn.BatchNorm2d(512),
      nn.MaxPool2d((3, 3), stride=2, padding=1)
    )
    self.module_6 = nn.Sequential(
      nn.AvgPool2d((1, 1), stride=1, padding=0)
    )
    self.module_7 = nn.Sequential(
      nn.Linear(512, n_classes)
    )


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
    out = self.module_1(x)
    out = self.module_2(out)
    out = self.module_3(out)
    out = self.module_4(out)
    out = self.module_5(out)
    out = self.module_6(out)
    out = self.module_7(out.squeeze())
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
