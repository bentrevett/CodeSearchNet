import torch
from torchtext import data

import numpy as np

import argparse
import random

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

CODE = data.Field()
DESC = data.Field()

fields = {'code_tokens': ('code', CODE), 'docstring_tokens': ('desc', DESC)}

train_data, valid_data, test_data = data.TabularDataset.splits(
                                        path = f'data/{args.lang}/final/jsonl',
                                        train = f'train/{args.lang}_train.jsonl',
                                        validation = f'valid/{args.lang}_valid.jsonl',
                                        test = f'test/{args.lang}_test.jsonl',
                                        format = 'json',
                                        fields = fields)

print(f'{len(train_data):,} training examples')
print(f'{len(valid_data):,} valid examples')
print(f'{len(test_data):,} test examples')

print(vars(train_data[0]))

CODE.build_vocab(train_data)
DESC.build_vocab(train_data)

print(f'Code vocab size: {len(CODE.vocab):,}')
print(f'Description vocab size: {len(DESC.vocab):,}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                    (train_data, valid_data, test_data),
                                                    batch_size = args.batch_size,
                                                    device = device,
                                                    sort_key = lambda x : x.code)

