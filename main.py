import torch
from torchtext import data

import numpy as np

import argparse
import random
import argparse

import models

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vocab_max_size', type=int, default=1_000_000)
parser.add_argument('--vocab_min_freq', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--model', type=str, default='bow')
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hid_dim', type=int, default=256)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--pool_mode', type=str, default='weighted_mean')
args = parser.parse_args()

print(vars(args))

assert args.model in ['bow', 'lstm', 'gru']
assert args.pool_mode in ['mean', 'max', 'weighted_mean']

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

print(f'Example: {vars(train_data[0])}')

CODE.build_vocab(train_data,
                 max_size = args.vocab_max_size,
                 min_freq = args.vocab_min_freq)
DESC.build_vocab(train_data,
                 max_size = args.vocab_max_size,
                 min_freq = args.vocab_min_freq)

print(f'Code vocab size: {len(CODE.vocab):,}')
print(f'Description vocab size: {len(DESC.vocab):,}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                    (train_data, valid_data, test_data),
                                                    batch_size = args.batch_size,
                                                    device = device,
                                                    sort_key = lambda x : x.code)

if args.model == 'bow':

    code_encoder = models.BagOfWordsEncoder(len(CODE.vocab),
                                            args.emb_dim)

    desc_encoder = models.BagOfWordsEncoder(len(DESC.vocab),
                                            args.emb_dim)

    code_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

    desc_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

elif args.model in ['gru', 'lstm']:

    code_encoder = models.RNNEncoder(len(CODE.vocab),
                                     args.emb_dim,
                                     args.hid_dim,
                                     args.n_layers,
                                     args.bidirectional,
                                     args.dropout,
                                     args.model)

    desc_encoder = models.RNNEncoder(len(DESC.vocab),
                                     args.emb_dim,
                                     args.hid_dim,
                                     args.n_layers,
                                     args.bidirectional,
                                     args.dropout,
                                     args.model)

    code_pooler = models.EmbeddingPooler(args.emb_dim * 2 if args.bidirectional else args.emb_dim,
                                         args.pool_mode)
                                         
    desc_pooler = models.EmbeddingPooler(args.emb_dim * 2 if args.bidirectional else args.emb_dim,
                                         args.pool_mode)

else:
    raise ValueError(f'Model {args.model} not valid!')

code_encoder = code_encoder.to(device)
desc_encoder = desc_encoder.to(device)
code_pooler = code_pooler.to(device)
desc_pooler = desc_pooler.to(device)