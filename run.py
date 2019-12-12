import subprocess
import os

"""seeds = [2,3,4,5]

for seed in seeds:
    command = f'python main2.py --lang java --model transformer --bpe_pct 0.5 --batch_size 450 --hid_dim 512 --n_layers 3 --lr 0.0005 --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()"""

"""lrs = [0.00075, 0.0005, 0.00025, 0.0001]
seeds = [1,1,1,1]

for lr, seed in zip(lrs, seeds):
    command = f'python sequence_lm.py --lang java --model bow --data code --save_model --lr {lr} --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait() """

"""seeds = [1,2,3,4,5]

for seed in seeds:
    command = f'python sequence_lm.py --lang java --model transformer --save_model --data code --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()"""

"""seeds = [2,3,4,5]

for seed in seeds:
    command = f'python main2.py --lang java --model transformer --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()"""

"""command = f'python sequence_lm.py --lang java --model transformer --save_model --data desc --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed 1'
process = subprocess.Popen(command, shell=True)
process.wait()"""

"""command = f'python sequence_lm.py --lang 6L-java --model transformer --save_model --data code --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed 1'
process = subprocess.Popen(command, shell=True)
process.wait()"""

"""command = f'python sequence_lm.py --lang 6L-java --model transformer --save_model --data desc --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed 1'
process = subprocess.Popen(command, shell=True)
process.wait()"""

"""command = f'python main2.py --lang java --model transformer --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed 1 --load'
process = subprocess.Popen(command, shell=True)
process.wait()"""

"""command = f'python main2.py --lang 6L-java --model transformer --bpe_pct 0.5 --batch_size 450 --hid_dim 256 --n_layers 3 --lr 0.0005 --seed 1 --load'
process = subprocess.Popen(command, shell=True)
process.wait()"""

seeds = [1,2,3,4,5]

for seed in seeds:

    #pre-train

    command = f'python main_sequence_lm.py --lang java --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang java --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 6L-java --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 6L-java --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 5L-java --n_epochs 5 --data code --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_sequence_lm.py --lang 5L-java --n_epochs 5 --data desc --model transformer --seed {seed} --save_model'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    #random init

    command = f'python main_code_retrieval.py --lang java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 6L-java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 5L-java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 6L-java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 5L-java --model transformer --seed {seed}'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    #fine tune

    command = f'python main_code_retrieval.py --lang java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 6L-java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_code_retrieval.py --lang 5L-java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 6L-java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command = f'python main_method_prediction.py --lang 5L-java --model transformer --seed {seed} --load'
    process = subprocess.Popen(command, shell=True)
    process.wait()