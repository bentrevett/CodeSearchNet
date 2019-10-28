import torch
import torch.optim as optim

from torchtext import data

import numpy as np
from tqdm import tqdm

import argparse
import random
import argparse

import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--code_max_length', type=int, default=200)
parser.add_argument('--desc_max_length', type=int, default=30)
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--dropout', type=float, default=0.25)
parser.add_argument('--pool_mode', type=str, default='weighted_mean')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=25)
parser.add_argument('--grad_clip', type=float, default=1)
args = parser.parse_args()

print(vars(args))

assert args.model in ['bow', 'lstm', 'gru']
assert args.pool_mode in ['mean', 'max', 'weighted_mean']

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def cut_code_max_length(tokens):
    return tokens[:args.code_max_length]

def cut_desc_max_length(tokens):
    return tokens[:args.desc_max_length]

CODE = data.Field(preprocessing=cut_code_max_length)
DESC = data.Field(preprocessing=cut_desc_max_length)

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

    code_pooler = models.EmbeddingPooler(args.hid_dim * 2 if args.bidirectional else args.hid_dim,
                                         args.pool_mode)
                                         
    desc_pooler = models.EmbeddingPooler(args.hid_dim * 2 if args.bidirectional else args.hid_dim,
                                         args.pool_mode)

else:
    raise ValueError(f'Model {args.model} not valid!')

code_encoder = code_encoder.to(device)
desc_encoder = desc_encoder.to(device)
code_pooler = code_pooler.to(device)
desc_pooler = desc_pooler.to(device)

print(code_encoder)
print(desc_encoder)
print(code_pooler)
print(desc_pooler)

print(f'Total number of parameters to train: {utils.count_parameters([code_encoder, desc_encoder, code_pooler, desc_pooler]):,}')

optimizer = optim.Adam([{'params': code_encoder.parameters()},
                        {'params': desc_encoder.parameters()},
                        {'params': code_pooler.parameters()},
                        {'params': desc_pooler.parameters()}],
                        lr = args.lr)

criterion = utils.SoftMaxLoss(device)

def train(code_encoder, desc_encoder, code_pooler, desc_pooler, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_mrr = 0

    code_encoder.train()
    desc_encoder.train()
    code_pooler.train()
    desc_pooler.train()

    for batch in tqdm(iterator, desc='Training...'):

        optimizer.zero_grad()

        encoded_code = code_pooler(code_encoder(batch.code))
        encoded_desc = desc_pooler(desc_encoder(batch.desc))

        #encoded_code/desc = [batch size, emb dim/hid dim/hid dim * 2 (bow/rnn/bi-rnn)]

        loss, mrr = criterion(encoded_code, encoded_desc)

        loss.backward()

        torch.nn.utils.clip_grad_value_(code_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_value_(desc_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_value_(code_pooler.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_value_(desc_pooler.parameters(), args.grad_clip)

        optimizer.step()

        epoch_loss += loss.item()
        epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

def evaluate(code_encoder, desc_encoder, code_pooler, desc_pooler, iterator, criterion):

    epoch_loss = 0
    epoch_mrr = 0

    code_encoder.eval()
    desc_encoder.eval()
    code_pooler.eval()
    desc_pooler.eval()

    with torch.no_grad():

        for batch in tqdm(iterator, desc='Evaluating...'):

            encoded_code = code_pooler(code_encoder(batch.code))
            encoded_desc = desc_pooler(desc_encoder(batch.desc))

            loss, mrr = criterion(encoded_code, encoded_desc)

            epoch_loss += loss.item()
            epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

best_valid_loss = float('inf')

for epoch in range(args.n_epochs):

    train_loss, train_mrr = train(code_encoder,
                                  desc_encoder,
                                  code_pooler,
                                  desc_pooler, 
                                  train_iterator, 
                                  optimizer,
                                  criterion)

    valid_loss, valid_mrr = evaluate(code_encoder,
                                     desc_encoder,
                                     code_pooler,
                                     desc_pooler, 
                                     valid_iterator,
                                     criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(code_encoder.state_dict(), 'code_encoder.pt')
        torch.save(desc_encoder.state_dict(), 'desc_encoder.pt')
        torch.save(code_pooler.state_dict(), 'code_pooler.pt')
        torch.save(desc_pooler.state_dict(), 'desc_pooler.pt')

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}, Train MRR: {train_mrr:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid MRR: {valid_mrr:.3f}')