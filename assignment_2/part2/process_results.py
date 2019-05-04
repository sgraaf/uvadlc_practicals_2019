from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageChops, ImageOps

sns.set_palette('cubehelix')


RESULTS_DIR = Path.cwd() / 'results'
if not RESULTS_DIR.exists():
    raise ValueError(
        'The directory which should contain the results does not exist')

generated_sequences_file_paths = RESULTS_DIR.glob('*_sequences.csv')
for file_path in generated_sequences_file_paths:
    # determine which book it is
    book = '_'.join(file_path.stem.split('_')[1:-1])
    print(book)

    # read the data into a dataframe
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')

    # T=30, temp=0.0, 0.5, 1.0, 1.5
    temps = [0.0, 0.5, 1.0, 2.0]
    df_ = df[
        (df['t'] == 30) &
        (df['step'] % 2000 == 0)
    ]

    for temp in temps:
        df__ = df_[(df_['temperature'] == temp)]
        print(df__[:6])

    print()