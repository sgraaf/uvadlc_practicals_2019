from pathlib import Path

import torch
from pandas import DataFrame as df
from pandas import read_csv

WEIRD_CHAR = '\xfe' 

def one_hot(batch, vocab_size, device='cpu'):
    """
    Converts the batch inputs to one-hot vectors of length vocab_size.

    :param torch.Tensor batch: the batch inputs
    :param int vocab_size: the size of the vocabulary (length of one-hot vectors)
    :param str device: the device to assign the batch input to
    :returns: the one-hot vectors of the batch inputs
    :rtype: torch.Tensor
    """
    shape = list(batch.shape) + [vocab_size]
    X = torch.zeros(shape, device=device)
    X.scatter_(2, batch.unsqueeze(dim=2), 1)

    return X


def get_accuracy(predicted_labels, true_labels):
    """
    Computes the accuracy of the predictions made.

    :param torch.Tensor predicted_labels: the predicted labels
    :param torch.Tensor true_labels: the true labels
    :returns: the accuracy
    :rtype: float
    """
    accuracy = (predicted_labels == true_labels.long()).float().mean()
    # accuracy /= (predicted_labels.shape[0] * predicted_labels.shape[1])

    return accuracy


def sample_character(output, temperature=1.0):
    """
    Sample a character from a probability distribution

    :param torch.Tensor output: the output from the model forward pass
    :param float temperature: the temperature with which to scale the probability distribution
    :returns: the character
    :rtype: int
    """
    if temperature is None or temperature == 0.0:
        sample = output.squeeze().argmax()
    else:
        output = (output.squeeze() / temperature).softmax(dim=0)
        sample = output.multinomial(1)

    return sample


def sample_sequence(model, vocab_size, T=30, char=None, temperature=1.0, device='cpu'):
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
    :rtype: list(int)
    """
    with torch.no_grad():
        # initialize sequence with random character
        if char is None:  # sample a random character and convert to one-hot
            char = torch.randint(vocab_size, (1, 1), device=device)
        
         # initialize the sequence to be sampled
        sequence = [char[0, 0].item()]
       
        # run the char through the model
        out, h_c = model.forward(char)
        
        # sample the next_char from the model
        char[0, 0] = sample_character(out, temperature)
        sequence.append(char[0, 0].item())

        for t in range(2, T):
            # run the char through the model
            out, h_c = model.forward(char, h_c)
            
            # sample the next_char from the model
            char[0, 0] = sample_character(out, temperature)
            sequence.append(char[0, 0].item())

        return sequence


def create_checkpoint(checkpoint_dir, filename, step, model, optimizer, results, sequences):
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
            'results': results,
            'sequences': sequences
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
    sequences = checkpoint['sequences']
    print('Done!')

    return step, results, sequences


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


def save_results(results_dir, filename, results, sequences, model):
    """
    Saves the training results

    :param pathlib.Path results_dir: the path of the directory to save the results in
    :param dict results: the results
    :param nn.Module model: the model
    """
    print('Saving the results...', end=' ')
    results_df = df.from_dict(results)
    results_path = results_dir / f'{model.__class__.__name__}_{filename}_results.csv'
    results_df.to_csv(results_path, sep=';', encoding='utf-8', index=False)

    sequences_df = df.from_dict(sequences)
    sequences_df['sequence'] = sequences_df['sequence'].str.replace('\n', WEIRD_CHAR)
    sequences_path = results_dir / f'{model.__class__.__name__}_{filename}_sequences.csv'
    sequences_df.to_csv(sequences_path, sep=';', encoding='utf-8', index=False)
    print('Done!')