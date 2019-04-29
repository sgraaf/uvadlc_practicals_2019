from pathlib import Path

import torch
from pandas import DataFrame as df
from pandas import read_csv


def one_hot(batch, vocab_size, device='cpu'):
    """
    Converts the batch inputs to one-hot vectors of length vocab_size.

    :param torch.Tensor batch: the batch inputs
    :param int vocab_size: the size of the vocabulary (length of one-hot vectors)
    :param str device: the device to assign the batch input to
    :returns: the one-hot vectors of the batch inputs
    :rtype: torch.Tensor
    """
    X = torch.zeros(batch.shape, vocab_size, device=device)
    X.scatter_(2, batch.unsqueeze(dim=2), 1)

    return X


def accuracy(predicted_labels, true_labels):
    """
    Computes the accuracy of the predictions made.

    :param torch.Tensor predicted_labels: the predicted labels
    :param torch.Tensor true_labels: the true labels
    :returns: the accuracy
    :rtype: float
    """
    accuracy = (predicted_labels.argmax(dim=2) == true_labels).sum().float()
    accuracy /= (predicted_labels.shape[0] * predicted_labels.shape[1])

    return accuracy


def sample_character(output, temperature=1.0):
    """
    Sample a character from a probability distribution

    :param torch.Tensor output: the output from the model forward pass
    :param float temperature: the temperature with which to scale the probability distribution
    :returns: the character
    :rtype: str
    """
    output = (output.squeeze() / temperature).softmax(dim=0)
    sample = output.multinomial(dim=1)

    return sample.item()


def sample_sequence(model, vocab_size, int2char, char2int, T=30, start_char=None, temperature=1.0, device='cpu'):
    """
    Sample a sequence of length T, given the previous character.

    :param nn.module model: the model to "sample" the sequence from
    :param int vocab_size: the size of the vocabulary
    :param dict int2char: the dictionary to convert ints to characters
    :param dixt char2int: the dictionary to convert characters to ints
    :param int T: the length of the sequence to generate
    :param str prev_char: the previous character
    :param str device: the device
    :returns: the sampled sequence
    :rtype: str
    """
    with torch.no_grad():
        # initialize sequence with random character
        if start_char is None:  # sample a random character and convert to one-hot
            start_char = torch.randint(vocab_size, (1, 1), dtype=torch.long, device=device)
        
        # convert the start_char to one-hot
        start_char = one_hot(start_char, vocab_size).to(device)
        
        # initialize the sequence to be sampled
        sequence = [start_char]
       
        # run the start_char through the model
        out, (h, c) = model(start_char)
        out = out[:, -1, :]
        
        # sample the next_char from the model
        next_char = sample_character(out, temperature)
        sequence.append(next_char)

        for t in range(T - 1):
            # convert the prev_char to one-hot
            prev_char = one_hot(next_char, vocab_size).to(device)

            # run the prev_char through the model
            out, (h, c) = model(prev_char, (h, c))
            out = out[:, -1, :]

            # sample the next_char from the model
            next_char = sample_character(out, temperature)
            sequence.append(next_char)

        return sequence


def create_checkpoint(checkpoint_dir, filename, step, model, optimizer, results):
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
    checkpoint_path = checkpoint_dir / (f'{model.__class__.__name__}_{filename}_checkpoint_{step}.pt')
    torch.save(
        {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results
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


def save_model(model_dir, filename, model):
    """
    Saves the model

    :param pathlib.Path model_dir: the path of the directory to save the models in
    :param nn.Module model: the model
    """
    print('Saving the model...', end=' ')
    model_path = model_dir / f'{model.__class__.__name__}_{filename}_model.pt'
    torch.save(model.state_dict(), model_path)
    print('Done!')


def save_results(results_dir, filename, results, model):
    """
    Saves the training results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{model.__class__.__name__}_{filename}_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8')
    print('Done!')