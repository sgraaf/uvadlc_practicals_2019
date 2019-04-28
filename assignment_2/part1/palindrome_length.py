from multiprocessing.dummy import Pool
from subprocess import run

if __name__ == "__main__":
    model_types = ['RNN', 'LSTM']
    cmds = [f'python train.py --input_length {i} --model_type {model_type}' for i in range(5, 21) for model_type in model_types]

    def run_cmd(cmd):
        run(cmd, shell=True)

    p = Pool(4)
    p.imap(run_cmd, cmds)
    p.join()