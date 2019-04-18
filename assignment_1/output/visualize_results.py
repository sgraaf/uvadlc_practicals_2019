from pathlib import Path
# from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('seaborn-white')

output_dir = Path.cwd()

# open file dialog for the CSV file
file_path = output_dir / 'additional_grid_mlp_pytorch.csv'
# Path(filedialog.askopenfilename(title='Please select the CSV-file',
#                                             filetypes=(('CSV', '*.csv'), ('All Files', '*.*')),
#                                             initialdir=output_dir))

# read the data
df = pd.read_csv(file_path, sep=';')
# print(df.dtypes)

# create the plot
# dnn_hidden_units = ['[400, 400, 400]', '[400, 400, 400, 400]', '[400, 400, 400, 400, 400]']
# step_ticks = [0, 500, 1000, 1500, 2000, 2500, 3000]
# accuracy_ticks = [0.0, 0.25, 0.50, 0.75, 1.0]

# fig, ax = plt.subplots()

# for k in range(len(dnn_hidden_units)):
#     sub_df = df[df['dnn_hidden_units'] == dnn_hidden_units[k]]
#     # print(sub_df)
#     ax.plot(sub_df['step'], sub_df['test_acc'], marker='.', label=f'{dnn_hidden_units[k]}')

# ax.set_xticks(step_ticks)
# ax.set_yticks(accuracy_ticks)
# ax.set_ylim([0, 1])

# ax.set(xlabel='Step')
# ax.set(ylabel='Accuracy')

# ax.legend(loc=1)
# ax.grid(True)
# plt.tight_layout()
# plt.savefig(file_path.parent / (file_path.stem + '_accuracy.png'))


# get the best model (highest test accuracy)
best_model = df.loc[df['test_acc'].idxmax()]
# print(best_model.to_frame().T)

# get all rows that correspond to the best model
df = df[
    (df['learning_rate'] == best_model['learning_rate']) &
    (df['max_steps'] == best_model['max_steps']) &
    (df['batch_size'] == best_model['batch_size']) &
    (df['dnn_hidden_units'] == best_model['dnn_hidden_units']) &
    (df['optimizer'] == best_model['optimizer'])
]

# print(df)

# create the accuracy-loss curve plot
fig, ax1 = plt.subplots()
ax1.plot(df['step'], df['test_acc'], color='tab:orange', marker='o', label='Test accuracy')
ax1.plot(df['step'], df['train_acc'], color='tab:red', marker='o', label='Train accuracy')
ax1.set_xlabel('Step')
ax1.set_ylabel('Accuracy')
ax1.tick_params('y')
ax1.set_ylim([0, 1])

ax2 = ax1.twinx()
ax2.plot(df['step'], df['test_loss'], color='tab:blue', marker='o', label='Test loss')
ax2.plot(df['step'], df['train_loss'], color='tab:green', marker='o', label='Train loss')
ax2.set_ylabel('Loss')
ax2.tick_params('y')
ax2.set_ylim([0, 5])

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc=0)
plt.tight_layout()
plt.savefig(file_path.parent / 'plots' / (file_path.stem + '_accuracy_loss_curves.png'))
