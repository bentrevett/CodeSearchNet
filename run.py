import subprocess
import os

seeds = [1,2,3,4,5]

for seed in seeds:
    command = f'python main.py --lang java --model transformer --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()