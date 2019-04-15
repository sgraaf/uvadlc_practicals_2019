"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    if len(n_hidden) > 0:
      # add modules for the first hidden layer (from x to h_1)
      self.modules = [
        LinearModule(n_inputs, n_hidden[0]),
        ReLUModule()
      ]

      # add modules for the intermediate hidden layers (from h_2 to h_n-1)
      for i in range(len(n_hidden) - 1):
        self.modules += [
          LinearModule(n_hidden[i], n_hidden[i + 1]),
          ReLUModule()
        ]

      # add modules for the last hidden layer (from h_n to y)
      self.modules += [
        LinearModule(n_hidden[-1], n_classes),
        SoftMaxModule()
      ]
    else:
      # no hidden units, so SLP w/ SoftMax instead of MLP
      self.modules = [
        LinearModule(n_inputs, n_classes),
        SoftMaxModule()
      ]
    
    # for module in self.modules:
    #   print(module.__class__.__name__)
    #   if isinstance(module, LinearModule):
    #     params = module.params
    #     keys = params.keys()
    #     for key in keys:
    #       print(f'  {key}: {params[key].shape}')
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
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # pass the input forwards through the modules
    for module in self.modules:
      x = module.forward(x)
    
    # the output is that of the final module
    out = x
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODONE:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # pass the loss gradients backwards through the modules
    for module in reversed(self.modules):
      dout = module.backward(dout)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return
