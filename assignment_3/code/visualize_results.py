from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageChops, ImageOps

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'
GAN_FILE = RESULTS_DIR / 'GAN_results.csv'
PLOT_FILE = RESULTS_DIR / 'GAN_loss.png'

def trim(image):
    background = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, background)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return image.crop(bbox)


def pad(image):
    return ImageOps.expand(image, border=5, fill='white')


df = pd.read_csv(GAN_FILE, sep=';', encoding='utf-8')

fig, ax = plt.subplots()
ax.plot(df.index, df['D_loss'], color='tab:orange', label='Discriminator loss')
ax.plot(df.index, df['G_loss'], color='tab:red', label='Generator loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.tick_params('y')
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_FILE)

I = Image.open(PLOT_FILE)
I = trim(I)
I = pad(I)
I.save(PLOT_FILE.parent / (PLOT_FILE.stem + '_trimmed' + PLOT_FILE.suffix))