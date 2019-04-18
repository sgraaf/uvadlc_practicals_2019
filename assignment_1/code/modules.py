"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODONE:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.in_features = in_features
    self.out_features = out_features

    self.params = {
      'weight': np.random.normal(0, 0.0001, (out_features, in_features)), 
      'bias': np.zeros((out_features, 1))
    }
    self.grads = {
      'weight': np.zeros((out_features, in_features)), 
      'bias': np.zeros((out_features, 1))
    }
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODONE:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = self.params['weight'] @ x.T + self.params['bias']
    self.x = x
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out.T

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODONE:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # compute dx
    dx = dout @ self.params['weight']

    # update the weight and bias gradients
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = dout.sum(axis=0).reshape(self.grads['bias'].shape)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODONE:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.maximum(0, x)
    self.x = x
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODONE:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # compute dx
    dx = dout * (self.x > 0)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODONE:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    out =  np.exp(x - x.max(axis=1, keepdims=True)) / np.exp(x - x.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)
    self.out = out
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module
    
    TODONE:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # get the softmax dimensions
    batch_size, features_dimension = self.out.shape

    # create a diagonal tensor with the elements of self.out
    diag_tensor = np.zeros((batch_size, features_dimension, features_dimension))
    idxs = np.arange(features_dimension)
    diag_tensor[:, idxs, idxs] = self.out

    # compute the partial derivative
    dx_dx_t = diag_tensor - np.einsum('ij, ik -> ijk', self.out, self.out)

    # compute the gradients
    dx = np.einsum('ij, ijk -> ik', dout, dx_dx_t)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """
  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    TODONE:
    Implement forward pass of the module. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.eps = 1e-10  # introduce very small epsilon to deal with zeroes in x
    out = - np.sum(y * np.log(x + self.eps), axis=1).mean()
    # out = - np.log(x + eps)[np.argmax(y)]
    self.out = out
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    TODONE:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # compute the gradient
    dx = (- y / (x + self.eps)) / y.shape[0]
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx
