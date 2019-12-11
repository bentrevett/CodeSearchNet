import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.init as init

from torchtext import data

import numpy as np
from tqdm import tqdm

import argparse
import os
import random
import functools

import models
import utils

UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--vocab_max_size', type=int, default=10_000)
parser.add_argument('--vocab_min_freq', type=int, default=10)
parser.add_argument('--bpe_pct', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--bptt', type=int, default=50)
parser.add_argument('--emb_dim', type=int, default=128)
parser.add_argument('--hid_dim', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--bidirectional', action='store_true')
parser.add_argument('--filter_size', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--n_epochs', type=int, default=500)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--grad_clip', type=float, default=1.0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--save_model', action='store_true')
args = parser.parse_args()

assert args.data in ['code', 'desc']
assert args.model in ['bow', 'lstm', 'gru', 'cnn', 'transformer']

if args.seed == None:
    args.seed = random.randint(0, 999)

args = utils.handle_args(args)

run_name = utils.get_run_name(args)

run_path = os.path.join('lm_runs/', *run_name)

assert not os.path.exists(run_path)

os.makedirs(run_path)

params_path = os.path.join(run_path, 'params.txt')
results_path = os.path.join(run_path, 'results.txt')

with open(params_path, 'w+') as f:
    for param, val in vars(args).items():
        f.write(f'{param}\t{val}\n')

with open(results_path, 'w+') as f:
    f.write('train_loss\ttrain_ppl\tvalid_loss\tvalid_ppl\n')

print(vars(args))

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.lang.startswith('6L-'):
    train_lang = '6L'
    valid_lang = args.lang.split('-')[-1]
    test_lang = args.lang.split('-')[-1]
else:
    train_lang = args.lang
    valid_lang = args.lang
    test_lang = args.lang

if args.bpe_pct <= 0:

    if args.data == 'code':

        vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train.code_vocab',
                                    args.vocab_max_size,
                                    PAD_TOKEN,
                                    UNK_TOKEN)

    else:

        vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train.desc_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)
else:

    if args.data == 'code':

        vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.code_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)

    else:

        vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.desc_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)

vocab_size = len(vocab)

print(f'vocab has {vocab_size} tokens')

numericalizer = functools.partial(utils.numericalize, vocab, UNK_TOKEN, 1_000_000_000)

if args.bpe_pct <= 0:

    train_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/train/{train_lang}_train.jsonl', numericalizer, args.data)
    valid_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/valid/{train_lang}_valid.jsonl', numericalizer, args.data)
    test_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/test/{train_lang}_test.jsonl', numericalizer, args.data)

else:

    train_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl', numericalizer, args.data)
    valid_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/valid/{train_lang}_valid_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl', numericalizer, args.data)
    test_data = utils.load_lm_data(f'data/{train_lang}/final/jsonl/test/{train_lang}_test_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl', numericalizer, args.data)

print('train/valid/test tokens', [len(x) for x in [train_data, valid_data, test_data]])

train_data = utils.batchify(train_data, args.batch_size)
valid_data = utils.batchify(valid_data, args.batch_size)
test_data = utils.batchify(test_data, args.batch_size)

print('train/valid/test shape', [x.shape for x in [train_data, valid_data, test_data]])

#data is [length, batch size]

device = torch.device('cuda')

if args.model == 'bow':

    model = models.BagOfWordsEncoder(len(vocab),
                                     args.emb_dim,
                                     args.dropout)

elif args.model == 'transformer':
    
    pad_idx = vocab[PAD_TOKEN]

    model = models.TransformerEncoder(len(vocab),
                                      args.emb_dim,
                                      args.hid_dim,
                                      args.n_layers,
                                      args.n_heads,
                                      args.dropout,
                                      pad_idx,
                                      device)

else:
    raise ValueError

if args.model == 'bow':

    def initialize_parameters(m):
        if isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    model.apply(initialize_parameters)

elif args.model == 'transformer':

    def truncated_normal_(tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

    def initialize_parameters(m):
        if isinstance(m, nn.LayerNorm):
            pass
        elif hasattr(m, 'weight'):
            truncated_normal_(m.weight.data, std=0.02)

    model.apply(initialize_parameters)

else:
    raise ValueError

language_model = models.LanguageModel(model, args.emb_dim, len(vocab))

print(language_model)

print(f'Language model parameters: {utils.count_parameters(language_model):,}')

optimizer = optim.Adam(language_model.parameters(), lr = args.lr)

criterion = nn.CrossEntropyLoss()

language_model.to(device)

def generate_square_subsequent_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.masked_fill(mask == 0, 0).masked_fill(mask == 1, float(1.0))
    return mask

def get_batch(data_source, i, bptt):
    seq_len = min(bptt, data_source.shape[0] - 1 - i)
    data = data_source[i:i+seq_len]
    target = data_source[i+1:i+1+seq_len]
    return data, target

def train(language_model, data_source, optimizer, criterion):

    epoch_loss = 0

    language_model.train()

    for i in tqdm(range(0, data_source.size(0) - 1, args.bptt)):

        optimizer.zero_grad()

        data, targets = get_batch(data_source, i, args.bptt)

        data = data.to(device)

        #data = [bptt, batch size]
        #targets = [bptt, batch size]

        if args.model == 'transformer':
            mask = generate_square_subsequent_mask(data.shape[0])
            mask = mask.to(device)
            predictions = language_model(data, mask)
        else:
            predictions = language_model(data)

        #predictions = [batch size, bptt, vocab size]

        predictions = predictions.permute(1, 0, 2)

        #predictions = [bptt, batch size, vocab size]

        predictions = predictions.contiguous().view(-1, vocab_size)

        #predictions = [bptt * batch size, vocab size]

        targets = targets.view(-1)

        #targets = [bptt * batch size]

        targets = targets.to(device)

        loss = criterion(predictions, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(language_model.parameters(), args.grad_clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_source)

def evaluate(language_model, data_source, criterion):

    epoch_loss = 0

    language_model.eval()

    for i in tqdm(range(0, data_source.size(0) - 1, args.bptt)):

        optimizer.zero_grad()

        data, targets = get_batch(data_source, i, args.bptt)

        data = data.to(device)

        #data = [bptt, batch size]
        #targets = [bptt, batch size]

        with torch.no_grad():

            if args.model == 'transformer':
                mask = generate_square_subsequent_mask(data.shape[0])
                mask = mask.to(device)
                predictions = language_model(data, mask)
            else:
                predictions = language_model(data)

            #predictions = [batch size, bptt, vocab size]

            predictions = predictions.permute(1, 0, 2)

            #predictions = [bptt, batch size, vocab size]

            predictions = predictions.contiguous().view(-1, vocab_size)

            #predictions = [bptt * batch size, vocab size]

            targets = targets.view(-1)

            #targets = [bptt * batch size]

            targets = targets.to(device)

            loss = criterion(predictions, targets)

        epoch_loss += loss.item()

    return epoch_loss / len(data_source)

best_valid_loss = float('inf')
patience_counter = 0

import math

for epoch in range(args.n_epochs):

    train_loss = train(language_model, train_data, optimizer, criterion)
    valid_loss = evaluate(language_model, valid_data, criterion)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        patience_counter = 0
        if args.save_model:
            torch.save(language_model.model.state_dict(), os.path.join(run_path, 'language_model_model.pt'))
    else:
        patience_counter += 1

    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)

    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}, Train PPL: {train_ppl:.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f}, Valid PPL: {valid_ppl:.3f}')

    with open(results_path, 'a') as f:
        f.write(f'{train_loss}\t{train_ppl}\t{valid_loss}\t{valid_ppl}\n')
    
    if patience_counter >= args.patience:
        print('Ended early due to losing patience!')
        with open(results_path, 'a') as f:
            f.write(f'lost_patience')
        break