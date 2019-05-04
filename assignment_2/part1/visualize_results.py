from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageChops, ImageOps

sns.set_palette('cubehelix')


def trim(image):
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def pad(image):
    return ImageOps.expand(image, border=5, fill='white')


RESULTS_DIR = Path.cwd() / 'results'
if not RESULTS_DIR.exists():
    raise ValueError(
        'The directory which should contain the results does not exist')

models = ['VanillaRNN', 'LSTM']

for i in range(len(models)):
    model = models[i]

    # open file dialog for the CSV file
    filepath = RESULTS_DIR / (model + '.csv')

    df = pd.read_csv(filepath, sep=';', encoding='utf-8')
    # print(df)

    # plot accuracy over timesteps for all palindrome lengths
    df_ = df
    df_ = df_[(df_['T'] % 2 == 1)]  # filter out certain T-values
    df_['Step'] = df_['Step'].apply(
        lambda x: x - x % 250)  # group rows by step
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='Step', y='Accuracy',
                            hue='T', data=df_, legend='full', ax=ax)
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.savefig(RESULTS_DIR / f'{model}_accuracy_steps_odd.png')
    plt.close()

    # plot loss over timesteps for all palindrome lengths
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='Step', y='Loss', hue='T',
                            data=df_, legend='full', ax=ax)
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.savefig(RESULTS_DIR / f'{model}_loss_steps_odd.png')
    plt.close()

    df_ = df
    df_ = df_[(df_['T'] % 2 == 0)]  # filter out certain T-values
    df_['Step'] = df_['Step'].apply(
        lambda x: x - x % 250)  # group rows by step
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='Step', y='Accuracy',
                            hue='T', data=df_, legend='full', ax=ax)
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.savefig(RESULTS_DIR / f'{model}_accuracy_steps_even.png')
    plt.close()

    # plot loss over timesteps for all palindrome lengths
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='Step', y='Loss', hue='T',
                            data=df_, legend='full', ax=ax)
    box = sns_plot.get_position()
    sns_plot.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    sns_plot.legend(loc='center right', bbox_to_anchor=(1.25, 0.5), ncol=1)
    plt.savefig(RESULTS_DIR / f'{model}_loss_steps_even.png')
    plt.close()

    # plot accuracy over timesteps for all palindrome lengths
    df_ = df
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='T', y='Accuracy', data=df_, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model}_accuracy_lengths.png')
    plt.close()

    # plot loss over timesteps for all palindrome lengths
    fig, ax = plt.subplots(1, 1)
    sns_plot = sns.lineplot(x='T', y='Loss', data=df_, ax=ax)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'{model}_loss_lengths.png')
    plt.close()

# trim images
images_file_paths = RESULTS_DIR.glob('*.png')
for images_file_path in images_file_paths:
	I = Image.open(images_file_path)
	I = trim(I)
	I = pad(I)
	I.save(images_file_path.parent / (images_file_path.stem + '_trimmed' + images_file_path.suffix))
