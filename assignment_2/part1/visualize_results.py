from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-white')

RESULTS_DIR = Path.cwd() / 'results'
if not RESULTS_DIR.exists():
    raise ValueError('The directory which should contain the results does not exist')

models = ['VanillaRNN', 'LSTM']

for model in models:
    # open file dialog for the CSV file
    filepath = RESULTS_DIR / (model + '.csv')

    df = pd.read_csv(filepath, sep=';')
    df = df[::20]

    # create the accuracy-loss curve plots for VanillaRNN
    fig, ax1 = plt.subplots()
    ax1.plot(df['Step'], df['Accuracy'], color='tab:orange', marker='.', label='Accuracy')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Accuracy')
    ax1.tick_params('y')
    ax1.set_ylim([0, 1])

    ax2 = ax1.twinx()
    ax2.plot(df['Step'], df['Loss'], color='tab:blue', marker='.', label='Loss')
    ax2.set_ylabel('Loss')
    ax2.tick_params('y')
    # ax2.set_ylim([0, 5])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / (model + '_accuracy_loss_curves.png'))
