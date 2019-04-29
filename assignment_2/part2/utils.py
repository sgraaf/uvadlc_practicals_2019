from pathlib import Path

import torch
from pandas import DataFrame as df
from pandas import read_csv


def one_hot(batch, vocab_size, device='cpu'):
    X = torch.zeros(batch.shape, vocab_size, device=device)
    X.scatter_(2, batch.unsqueeze(dim=2), 1)

    return X


def accuracy(predicted_labels, true_labels):
    accuracy = (predicted_labels.argmax(dim=2) == true_labels).sum().float()
    accuracy /= (predicted_labels.shape[0] * predicted_labels.shape[1])

    return accuracy


def create_checkpoint(checkpoint_dir, step, model, optimizer, results, best_accuracy):
    """
    Creates a checkpoint for the current step

    :param pathlib.Path checkpoint_dir: the path of the directory to store the checkpoints in
    :param int step: the current step
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :param dict results: the results
    :param float best_accuracy: the best accuracy thus far
    """
    print('Creating checkpoint...', end=' ')
    checkpoint_path = checkpoint_dir / (f'{model.__class__.__name__}_checkpoint_{step}.pt')
    torch.save(
        {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'best_accuracy': best_accuracy
        },
        checkpoint_path
    )
    print('Done!')


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Loads a checkpoint

    :param pathlib.Path checkpoint_path: the path of the checkpoint
    :param nn.Module model: the model
    :param optim.Optimizer optimizer: the optimizer
    :returns: tuple of epoch, model, optimizer, results and best_accuracy of the checkpoint
    :rtype: tuple(int, nn.Module, optim.Optimizer, dict, float)
    """
    print('Loading checkpoint...', end=' ')
    checkpoint = torch.load(checkpoint_path)
    step = checkpoint['step'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    results = checkpoint['results']
    best_accuracy = checkpoint['best_accuracy']
    print('Done!')

    return step, results, best_accuracy


def save_model(model_dir, model):
    """
    Saves the model

    :param pathlib.Path model_dir: the path of the directory to save the models in
    :param nn.Module model: the model
    """
    print('Saving the model...', end=' ')
    model_path = model_dir / f'{model.__class__.__name__}_model.pt'
    torch.save(model.state_dict(), model_path)
    print('Done!')


def save_results(results_dir, results, model):
    """
    Saves the training results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{model.__class__.__name__}_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8')
    print('Done!')