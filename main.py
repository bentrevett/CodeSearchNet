import torch
import torch.optim as optim

from torchtext import data

import numpy as np
from tqdm import tqdm

import argparse
import os
import random

import bpe
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--code_max_length', type=int, default=200)
parser.add_argument('--desc_max_length', type=int, default=30)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)
parser.add_argument('--bpe_pct', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--filter_size', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--pool_mode', type=str, default='weighted_mean')
parser.add_argument('--loss', type=str, default='softmax')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--save_model', action='store_true')
args = parser.parse_args()

assert args.model in ['bow', 'lstm', 'gru', 'cnn', 'transformer']
assert args.pool_mode in ['mean', 'max', 'weighted_mean']
assert args.loss in ['softmax', 'cosine']

if args.seed == None:
    args.seed = random.randint(0, 999)

args = utils.handle_args(args)

run_name = utils.get_run_name(args)

run_path = os.path.join('runs/', run_name)

assert not os.path.exists(run_path)

os.makedirs(run_path)

params_path = os.path.join(run_path, 'params.txt')
results_path = os.path.join(run_path, 'results.txt')

with open(params_path, 'w+') as f:
    for param, val in vars(args).items():
        f.write(f'{param}\t{val}\n')

with open(results_path, 'w+') as f:
    f.write('train_loss\ttrain_mrr\tvalid_loss\tvalid_mrr\n')

print(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def cut_code_max_length(tokens):
    return tokens[:args.code_max_length]

def cut_desc_max_length(tokens):
    return tokens[:args.desc_max_length]

if args.bpe_pct <= 0:

    CODE = data.Field(preprocessing = cut_code_max_length, include_lengths = True)
    DESC = data.Field(preprocessing = cut_desc_max_length, include_lengths = True)

    fields = {'code_tokens': ('code', CODE), 'docstring_tokens': ('desc', DESC)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = f'data/{args.lang}/final/jsonl',
                                            train = f'train/{args.lang}_train.jsonl',
                                            validation = f'valid/{args.lang}_valid.jsonl',
                                            test = f'test/{args.lang}_test.jsonl',
                                            format = 'json',
                                            fields = fields)

    CODE.build_vocab(train_data,
                     max_size = args.vocab_max_size,
                     min_freq = args.vocab_min_freq)

    DESC.build_vocab(train_data,
                     max_size = args.vocab_max_size,
                     min_freq = args.vocab_min_freq)

else:

    CODE = data.Field(preprocessing = cut_code_max_length, include_lengths = True)
    DESC = data.Field(preprocessing = cut_desc_max_length, include_lengths = True)

    fields = {'code_tokens': ('code', CODE), 'docstring_tokens': ('desc', DESC)}

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = f'data/{args.lang}/final/jsonl',
                                            train = f'train/{args.lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            validation = f'valid/{args.lang}_valid_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            test = f'test/{args.lang}_test_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            format = 'json',
                                            fields = fields)

    CODE.build_vocab(train_data)

    DESC.build_vocab(train_data)

print(f'{len(train_data):,} training examples')
print(f'{len(valid_data):,} valid examples')
print(f'{len(test_data):,} test examples')

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
                                            args.emb_dim,
                                            args.dropout)

    desc_encoder = models.BagOfWordsEncoder(len(DESC.vocab),
                                            args.emb_dim,
                                            args.dropout)

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

elif args.model == 'cnn':

    code_encoder = models.CNNEncoder(len(CODE.vocab),
                                     args.emb_dim,
                                     args.filter_size,
                                     args.n_layers,
                                     args.dropout,
                                     device)

    desc_encoder = models.CNNEncoder(len(DESC.vocab),
                                     args.emb_dim,
                                     args.filter_size,
                                     args.n_layers,
                                     args.dropout,
                                     device)

    code_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

    desc_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

elif args.model == 'transformer':

    code_pad_idx = CODE.vocab.stoi[CODE.pad_token]
    desc_pad_idx = DESC.vocab.stoi[DESC.pad_token]

    code_encoder = models.TransformerEncoder(len(CODE.vocab),
                                             args.emb_dim,
                                             args.hid_dim,
                                             args.n_layers,
                                             args.n_heads,
                                             args.dropout,
                                             code_pad_idx,
                                             device)

    desc_encoder = models.TransformerEncoder(len(DESC.vocab),
                                             args.emb_dim,
                                             args.hid_dim,
                                             args.n_layers,
                                             args.n_heads,
                                             args.dropout,
                                             desc_pad_idx,
                                             device)

    code_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

    desc_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

else:
    raise ValueError(f'Model {args.model} not valid!')

code_encoder.apply(utils.initialize_parameters)
desc_encoder.apply(utils.initialize_parameters)
code_pooler.apply(utils.initialize_parameters)
desc_pooler.apply(utils.initialize_parameters)

code_encoder = code_encoder.to(device)
desc_encoder = desc_encoder.to(device)
code_pooler = code_pooler.to(device)
desc_pooler = desc_pooler.to(device)

print(code_encoder)
print(desc_encoder)
print(code_pooler)
print(desc_pooler)

print(f'Code Encoder parameters: {utils.count_parameters(code_encoder):,}')
print(f'Desc Encoder parameters: {utils.count_parameters(desc_encoder):,}')
print(f'Code Pooler parameters: {utils.count_parameters(code_pooler):,}')
print(f'Desc Pooler parameters: {utils.count_parameters(desc_pooler):,}')

optimizer = optim.Adam([{'params': code_encoder.parameters()},
                        {'params': desc_encoder.parameters()},
                        {'params': code_pooler.parameters()},
                        {'params': desc_pooler.parameters()}],
                        lr = args.lr)

if args.loss == 'softmax':
    criterion = utils.SoftmaxLoss(device)
elif args.loss == 'cosine':
    criterion = utils.CosineLoss(device)
else:
    raise ValueError(f'Loss {args.loss} not valid!')

def train(code_encoder, desc_encoder, code_pooler, desc_pooler, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_mrr = 0

    code_encoder.train()
    desc_encoder.train()
    code_pooler.train()
    desc_pooler.train()

    for batch in tqdm(iterator, desc='Training...'):

        optimizer.zero_grad()

        code, code_lengths = batch.code
        desc, desc_lengths = batch.desc

        #code/desc = [seq len, batch size]

        code_mask = utils.make_mask(code, CODE.vocab.stoi[CODE.pad_token])
        desc_mask = utils.make_mask(desc, DESC.vocab.stoi[DESC.pad_token])

        #mask = [batch size, seq len]

        encoded_code = code_encoder(code)
        encoded_code = code_pooler(encoded_code, code_mask)

        encoded_desc = desc_encoder(desc)
        encoded_desc = desc_pooler(encoded_desc, desc_mask)

        #encoded_code/desc = [batch size, emb dim/hid dim/hid dim * 2 (bow/rnn/bi-rnn)]

        loss, mrr = criterion(encoded_code, encoded_desc)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(code_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(desc_encoder.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(code_pooler.parameters(), args.grad_clip)
        torch.nn.utils.clip_grad_norm_(desc_pooler.parameters(), args.grad_clip)

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

            code, code_lengths = batch.code
            desc, desc_lengths = batch.desc

            code_mask = utils.make_mask(code, CODE.vocab.stoi[CODE.pad_token])
            desc_mask = utils.make_mask(desc, DESC.vocab.stoi[DESC.pad_token])

            encoded_code = code_encoder(code)
            encoded_code = code_pooler(encoded_code, code_mask)

            encoded_desc = desc_encoder(desc)
            encoded_desc = desc_pooler(encoded_desc, desc_mask)

            loss, mrr = criterion(encoded_code, encoded_desc)

            epoch_loss += loss.item()
            epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

best_valid_mrr = float('inf')
patience_counter = 0

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

    if valid_mrr < best_valid_mrr:
        best_valid_mrr = valid_mrr
        patience_counter = 0
        if args.save_model:
            torch.save(code_encoder.state_dict(), os.path.join(run_path, 'code_encoder.pt'))
            torch.save(desc_encoder.state_dict(), os.path.join(run_path, 'desc_encoder.pt'))
            torch.save(code_pooler.state_dict(), os.path.join(run_path, 'code_pooler.pt'))
            torch.save(desc_pooler.state_dict(), os.path.join(run_path, 'desc_pooler.pt'))
        
    else:
        patience_counter += 1

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}, Train MRR: {train_mrr:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid MRR: {valid_mrr:.3f}')

    with open(results_path, 'a') as f:
        f.write(f'{train_loss}\t{train_mrr}\t{valid_loss}\t{valid_mrr}\n')

    if patience_counter >= args.patience:
        print('Ended early due to losing patience!')
        with open(results_path, 'a') as f:
            f.write(f'lost_patience')
        break