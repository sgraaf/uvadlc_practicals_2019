from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-white')

RESULTS_DIR = Path.cwd() / 'results'
if not RESULTS_DIR.exists():
    raise ValueError('The directory which should contain the results does not exist')

models = ['VanillaRNN', 'LSTM']
colors = ['tab:orange', 'tab:blue']

fig, ax1 = plt.subplots()

for i in range(len(models)):
    model = models[i]
    color = colors[i]

    # open file dialog for the CSV file
    filepath = RESULTS_DIR / (model + '.csv')

    df = pd.read_csv(filepath, sep=';')
    # print(df)

    # create the accuracy-loss curve plots for VanillaRNN
    ax1.plot(df['T'], df['Accuracy'], color=color, marker='o', label=model)

ax1.set_xlabel('T')
ax1.set_ylabel('Accuracy')
ax1.set_xticks(np.arange(5, 21))
ax1.set_yticks(np.arange(0, 1.1, 0.1))
ax1.set_ylim([0, 1.05])

ax1.legend(loc=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'palindrome_length_accuracies.png')
