import subprocess
import os

seeds = [1,2,3,4,5]

for seed in seeds:
    command = f'python main2.py --lang 6L-java --model transformer --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main2.py --lang 6L-java --model transformer --bpe_pct 0 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()