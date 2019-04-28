import os

model_types = ['RNN', 'LSTM']
ts = range(5, 21)

for model_type in model_types:
    for t in ts:
        os.system(f'python train.py --input_length {t} --model_type {model_type}')
