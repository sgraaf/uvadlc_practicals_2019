"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pathlib import Path
from convnet_pytorch import ConvNet
import cifar10_utils

import torch

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
  Performs training and evaluation of ConvNet model. 

  TODONE:
  Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

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

  # compute the number of test_evals
  test_evals = int(np.ceil(n_test_images / batch_size))

  # initialize the MLP, loss function and optimizer
  convnet = ConvNet(depth, n_classes).to(device)
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(convnet.parameters(), lr=learning_rate)

  # initialize empty results list
  results = []
  output_dir = Path.cwd().parent / 'output'
  output_dir.mkdir(parents=True, exist_ok=True)
  best_acc = 0.0

  # train the ConvNet
  for step in range(max_steps):
    # obtain a new mini-batch of training data and convert to float send to device
    train_images, train_labels = train.next_batch(batch_size)
    train_images_torch = torch.tensor(train_images, dtype=torch.float, device=device)
    train_labels_torch = torch.tensor(train_labels, dtype=torch.long, device=device)

    # reset the optimizer gradient to zero
    optimizer.zero_grad()

    # forward pass the mini-batch
    probabilities = convnet(train_images_torch)
    loss = loss_function.forward(probabilities, train_labels_torch.argmax(dim=1))
    
    # backwards propogate the loss
    loss.backward()
    optimizer.step()

    # evaluate the ConvNet
    if (step % eval_freq == 0) or (step == max_steps - 1):

      # compute the train data metrics
      train_acc = accuracy(probabilities, train_labels_torch)
      train_loss = loss.item()

      # detach probabilities to free up memory
      probabilities.detach()

      test_accs = []
      test_losses = []

      for test_eval in range(test_evals):
        # obtain a new mini-batch of training data and convert to float send to device
        test_images, test_labels = test.next_batch(batch_size)
        test_images_torch = torch.tensor(test_images, dtype=torch.float, device=device)
        test_labels_torch = torch.tensor(test_labels, dtype=torch.long, device=device)

        # evaluate the MLP on the test data
        test_probabilities = convnet(test_images_torch)
        
        # compute the test data metrics
        test_accs.append(accuracy(test_probabilities, test_labels_torch))
        test_losses.append(loss_function.forward(test_probabilities, test_labels_torch.argmax(dim=1)).item())

        # detach to free up memory
        test_probabilities.detach()
        test_images_torch.detach()
        test_labels_torch.detach()      

      # compute the average test accuracy and loss
      test_acc = np.mean(test_accs)
      test_loss = np.mean(test_losses)

      # append the results
      results.append([step + 1, train_acc, train_loss, test_acc, test_loss])

      print(f'Step {step + 1:0{len(str(max_steps))}}/{max_steps}:')
      print(f' Performance on the training data (mini-batch):')
      print(f'  Accuracy: {train_acc}, Loss: {train_loss}')
      print(f' Performance on the testing data (mini-batch):')
      print(f'  Accuracy: {test_acc}, Loss: {test_loss}')

      # break if train loss has converged
      threshold = 1e-6
      if len(results) > 20:
        previous_losses = results[-20:-10][2]
        current_losses = train_loss[-10:][2]
        if (previous_losses - current_losses) < threshold:
          print(f'Loss has converged early in {step} steps')
          break

  # save the relevant results to disk
  print('Saving the results to disk...')
  output_path = Path.cwd().parent / 'output' / 'convnet_pytorch.csv'
  output_path.parent.mkdir(parents=True, exist_ok=True)

  if output_path.exists():
    mode = 'a'
  else:
    mode = 'w'

  column_names = ['step', 'train_acc', 'train_loss', 'test_acc', 'test_loss']

  with open(output_path, mode) as csv_file:
    if mode == 'w':
      csv_file.write(','.join(column_names) + '\n')
    for i in range(len(results)):
      csv_file.write(f'{results[i][0]},{results[i][1]},{results[i][2]},{results[i][3]},{results[i][4]}' + '\n')
  
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