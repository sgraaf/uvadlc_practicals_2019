import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""

######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  The operations called in self.forward track the history if the input tensors have the
  flag requires_grad set to True. The backward pass does not need to be implemented, it
  is dealt with by the automatic differentiation provided by PyTorch.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormAutograd object. 
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODONE:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormAutograd, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # save parameters
    self.n_neurons = n_neurons
    self.eps = eps
    
    # initialize parameters gamma and beta via nn.Parameter
    self.gamma = nn.Parameter(torch.ones(self.n_neurons))
    self.beta = nn.Parameter(torch.zeros(self.n_neurons))
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODONE:
      Check for the correctness of the shape of the input tensor.
      Implement batch normalization forward pass as given in the assignment.
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    n_batch = n_neurons = input.shape

    # check for correctness of the shape of the input tensor
    assert n_neurons == self.n_neurons

    # implement batch normalization
    mu = input.mean(dim=0)
    var = input.var(dim=0, unbiased=False)
    x_hat = (input - mu) / (var + self.eps).sqrt()
    out = self.gamma * x_hat + self.beta
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out



######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
  """
  This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
  Using torch.autograd.Function allows you to write a custom backward function.
  The function will be called from the nn.Module CustomBatchNormManualModule
  Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
  pass is done via the backward method.
  The forward pass is not called directly but via the apply() method. This makes sure that the context objects
  are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
  """

  @staticmethod
  def forward(ctx, input, gamma, beta, eps=1e-5):
    """
    Compute the batch normalization
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
      input: input tensor of shape (n_batch, n_neurons)
      gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
      beta: mean bias tensor, applied per neuron, shpae (n_neurons)
      eps: small float added to the variance for stability
    Returns:
      out: batch-normalized tensor

    TODONE:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
      For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # similar as for 3.1 a
    mu = input.mean(dim=0)
    var = input.var(dim=0, unbiased=False)
    x_hat = (input - mu) / (var + eps).sqrt()
    out = gamma * x_hat + beta

    # store tensors for the backward pass
    ctx.save_for_backward(var, gamma, x_hat)

    # store constant non-tensor objects
    ctx.eps = eps
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out


  @staticmethod
  def backward(ctx, grad_output):
    """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments
    
    TODONE:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # retrieve the batch_size and n_neurons
    batch_size, n_neurons = grad_output.shape

    # retrieve stored tensors and constants
    var, gamma, x_hat = ctx.saved_tensors
    eps = ctx.eps

    # determine which inputs need a gradient computed
    (input_grad, gamma_grad, beta_grad) = ctx.needs_input_grad

    if input_grad:  # input gradient
      grad_x_hat = grad_output * gamma
      denominator = batch_size * (var + eps).sqrt
      grad_input_term_1 = batch_size * grad_x_hat
      grad_input_term_2 = - grad_x_hat.sum(dim=0)
      grad_input_term_3 = - x_hat * (grad_x_hat * x_hat).sum(dim=0)
      grad_input = (grad_input_term_1 + grad_input_term_2 + grad_input_term_3) / denominator
    else:
      grad_input = None
    
    if gamma_grad:  # gamma gradient
      grad_gamma = grad_output.mul(x_hat).sum(dir=0)
    else:
      grad_gamma = None
    
    if beta_grad:  # beta grad
      grad_beta = grad_output.sum(dir=0)
    else:
      grad_beta = None
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    # return gradients of the three tensor inputs and None for the constant eps
    return grad_input, grad_gamma, grad_beta, None



######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
  """
  This nn.module implements a custom version of the batch norm operation for MLPs.
  In self.forward the functional version CustomBatchNormManualFunction.forward is called.
  The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
  """

  def __init__(self, n_neurons, eps=1e-5):
    """
    Initializes CustomBatchNormManualModule object.
    
    Args:
      n_neurons: int specifying the number of neurons
      eps: small float to be added to the variance for stability
    
    TODONE:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
    super(CustomBatchNormManualModule, self).__init__()

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # save parameters
    self.n_neurons = n_neurons
    self.eps = eps
    
    # initialize parameters gamma and beta via nn.Parameter
    self.gamma = nn.Parameter(torch.ones(self.n_neurons))
    self.beta = nn.Parameter(torch.zeros(self.n_neurons))
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, input):
    """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODONE:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    n_batch = n_neurons = input.shape

    # check for correctness of the shape of the input tensor
    assert n_neurons == self.n_neurons

    # instantiate a CBNMF and call its .apply() me
    # 
    # thod
    CBNMF = CustomBatchNormManualFunction()
    out = CBNMF.apply(input, self.gamma, self.beta, self.eps)
    # raise NotImplementedError
    ########################
    # END OF YOUR CODE    #
    #######################

    return out
