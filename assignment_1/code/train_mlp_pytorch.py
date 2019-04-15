"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
from mlp_pytorch import MLP
import cifar10_utils

import torch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODONE:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = (predictions.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()
  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODONE:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  learning_rate = FLAGS.learning_rate
  max_steps     = FLAGS.max_steps
  batch_size    = FLAGS.batch_size
  eval_freq     = FLAGS.eval_freq
  data_dir      = FLAGS.data_dir

  # load the cifar10 data
  cifar10 = cifar10_utils.get_cifar10(data_dir)
  train = cifar10['train']
  test = cifar10['test']
  test_images, test_labels = test.images, test.labels
  
  # obtain the dimensions of the data
  n_test_images, depth, height, width = test_images.shape
  n_inputs = height * width * depth
  n_classes = test_labels.shape[1]

  # reshape the test images and send to device
  test_images = test_images.reshape((n_test_images, n_inputs))
  # test_images_torch = torch.from_numpy(test_images).to(device)
  test_images_torch = torch.tensor(test_images, dtype=torch.float, device=device)
  test_labels_torch = torch.tensor(test_labels, dtype=torch.long, device=device)

  # initialize the MLP, loss function and optimizer
  mlp = MLP(n_inputs, dnn_hidden_units, n_classes).to(device)
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(mlp.parameters(), lr=learning_rate)

  # initialize relevant metrics
  train_acc = []
  train_loss = []
  test_acc = []
  test_loss = []

  # train the MLP
  for step in range(max_steps):
    # obtain a new mini-batch and reshape the images and convert to float send to device
    train_images, train_labels = train.next_batch(batch_size)
    train_images = train_images.reshape((batch_size, n_inputs))
    train_images_torch = torch.tensor(train_images, dtype=torch.float, device=device)
    train_labels_torch = torch.tensor(train_labels, dtype=torch.long, device=device)

    # reset the optimizer gradient to zero
    optimizer.zero_grad()

    # forward pass the mini-batch
    predictions = mlp.forward(train_images_torch)
    loss = loss_function.forward(predictions, train_labels_torch.argmax(dim=1))
    
    # backwards propogate the loss
    loss.backward()
    optimizer.step()

    # evaluate the MLP
    if (step % eval_freq == 0) or (step == max_steps - 1):
      # append train data metrics
      train_acc.append(accuracy(predictions, train_labels_torch))
      train_loss.append(loss)

      # evaluate the MLP on the test data
      test_predictions = mlp.forward(test_images_torch)
      
      # append the test data metrics
      test_acc.append(accuracy(test_predictions, test_labels_torch))
      test_loss.append(loss_function.forward(test_predictions, test_labels_torch.argmax(dim=1)))

      print(f'Step {step:0{len(str(max_steps))}} / {max_steps}:')
      print(f' Performance on the training data (mini-batch):')
      print(f'  Accuracy: {train_acc[-1]}')
      print(f'  Loss: {train_loss[-1]}')
      print(f' Performance on the testing data (mini-batch):')
      print(f'  Accuracy: {test_acc[-1]}')
      print(f'  Loss: {test_loss[-1]}')

      # break if train loss has converged
      threshold = 1e-6
      if len(train_loss) > 20:
        previous_losses = train_loss[-20:-10]
        current_losses = train_loss[-10:]
        if (previous_losses - current_losses) < threshold:
          print(f'Loss has converged early in {step} steps')
          break

  # save the relevant metrics to disk
  print('Saving the metrics to disk...')
  output_dir = Path.cwd().parent / 'output'
  if not output_dir.exists():
    output_dir.mkdir(parents=True)

  combined_metrics = list(zip(train_acc, train_loss, test_acc, test_loss))
  metric_names = ['train_acc', 'train_loss', 'test_acc', 'test_loss']
  df = pd.DataFrame(combined_metrics, columns=metric_names)
  df.to_csv(output_dir / 'mlp_pytorch.csv')
  np.savetxt(output_dir / 'train_acc.csv', train_acc, delimiter=',')
  np.savetxt(output_dir / 'train_loss.csv', train_loss, delimiter=',')
  np.savetxt(output_dir / 'test_acc.csv', test_acc, delimiter=',')
  np.savetxt(output_dir / 'test_loss.csv', test_loss, delimiter=',')
  # raise NotImplementedError
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()