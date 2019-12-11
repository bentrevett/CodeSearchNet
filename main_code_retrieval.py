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
parser.add_argument('--load', action='store_true')
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
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--save_model', action='store_true')
args = parser.parse_args()

assert args.model in ['bow', 'lstm', 'gru', 'cnn', 'transformer']
assert args.pool_mode in ['max', 'weighted_mean']
assert args.loss in ['softmax']

if args.seed == None:
    args.seed = random.randint(0, 999)

args = utils.handle_args(args)

run_name = utils.get_run_name(args)

run_path = os.path.join('runs/', *run_name)

if args.load:

    code_load_path = f'lm_runs/lang={args.lang}/model={args.model}/data=code/vocab_max_size={args.vocab_max_size}/bpe_pct={args.bpe_pct}/batch_size={args.batch_size}/bptt=50/emb_dim={args.emb_dim}/hid_dim={args.hid_dim}/n_layers={args.n_layers}/n_heads={args.n_heads}/dropout={args.dropout}/lr={args.lr}/n_epochs={args.n_epochs}/patience={args.patience}/grad_clip={args.grad_clip}/seed=1/save_model=True'
    desc_load_path = f'lm_runs/lang={args.lang}/model={args.model}/data=desc/vocab_max_size={args.vocab_max_size}/bpe_pct={args.bpe_pct}/batch_size={args.batch_size}/bptt=50/emb_dim={args.emb_dim}/hid_dim={args.hid_dim}/n_layers={args.n_layers}/n_heads={args.n_heads}/dropout={args.dropout}/lr={args.lr}/n_epochs={args.n_epochs}/patience={args.patience}/grad_clip={args.grad_clip}/seed=1/save_model=True'

    assert os.path.exists(code_load_path), code_load_path
    assert os.path.exists(desc_load_path), desc_load_path

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

if args.lang.startswith('6L-'):
    train_lang = '6L'
    valid_lang = args.lang.split('-')[-1]
    test_lang = args.lang.split('-')[-1]
else:
    train_lang = args.lang
    valid_lang = args.lang
    test_lang = args.lang

if args.bpe_pct <= 0:
    code_vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train.code_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)

    desc_vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train.desc_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)
else:
    code_vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.code_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)

    desc_vocab = utils.load_vocab(f'data/{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.desc_vocab',
                                  args.vocab_max_size,
                                  PAD_TOKEN,
                                  UNK_TOKEN)

code_numericalizer = functools.partial(utils.numericalize, code_vocab, UNK_TOKEN, args.code_max_length)
desc_numericalizer = functools.partial(utils.numericalize, desc_vocab, UNK_TOKEN, args.desc_max_length)

CODE = data.Field(use_vocab = False,
                preprocessing = code_numericalizer,
                pad_token = code_vocab[PAD_TOKEN],
                unk_token = code_vocab[UNK_TOKEN],
                include_lengths = True)

DESC = data.Field(use_vocab = False,
                preprocessing = desc_numericalizer,
                pad_token = desc_vocab[PAD_TOKEN],
                unk_token = desc_vocab[UNK_TOKEN],
                include_lengths = True)

fields = {'code_tokens': ('code', CODE), 'docstring_tokens': ('desc', DESC)}

if args.bpe_pct <= 0:

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = f'data',
                                            train = f'{train_lang}/final/jsonl/train/{train_lang}_train.jsonl',
                                            validation = f'{valid_lang}/final/jsonl/valid/{valid_lang}_valid.jsonl',
                                            test = f'{test_lang}/final/jsonl/test/{test_lang}_test.jsonl',
                                            format = 'json',
                                            fields = fields)

else:

    train_data, valid_data, test_data = data.TabularDataset.splits(
                                            path = f'data',
                                            train = f'{train_lang}/final/jsonl/train/{train_lang}_train_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            validation = f'{valid_lang}/final/jsonl/valid/{valid_lang}_valid_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            test = f'{test_lang}/final/jsonl/test/{test_lang}_test_bpe_{args.vocab_max_size}_{args.bpe_pct}.jsonl',
                                            format = 'json',
                                            fields = fields)

print(f'{len(train_data):,} training examples')
print(f'{len(valid_data):,} valid examples')
print(f'{len(test_data):,} test examples')

print(f'Code vocab size: {len(code_vocab):,}')
print(f'Description vocab size: {len(desc_vocab):,}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
                                                    (train_data, valid_data, test_data),
                                                    batch_size = args.batch_size,
                                                    device = device,
                                                    sort_key = lambda x : x.code)

if args.model == 'bow':

    code_encoder = models.BagOfWordsEncoder(len(code_vocab),
                                            args.emb_dim,
                                            args.dropout)

    desc_encoder = models.BagOfWordsEncoder(len(desc_vocab),
                                            args.emb_dim,
                                            args.dropout)

    code_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

    desc_pooler = models.EmbeddingPooler(args.emb_dim,
                                         args.pool_mode)

elif args.model in ['gru', 'lstm']:

    code_encoder = models.RNNEncoder(len(code_vocab),
                                     args.emb_dim,
                                     args.hid_dim,
                                     args.n_layers,
                                     args.bidirectional,
                                     args.dropout,
                                     args.model)

    desc_encoder = models.RNNEncoder(len(desc_vocab),
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

    code_encoder = models.CNNEncoder(len(code_vocab),
                                     args.emb_dim,
                                     args.filter_size,
                                     args.n_layers,
                                     args.dropout,
                                     device)

    desc_encoder = models.CNNEncoder(len(desc_vocab),
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

    code_pad_idx = code_vocab[PAD_TOKEN]
    desc_pad_idx = desc_vocab[PAD_TOKEN]

    code_encoder = models.TransformerEncoder(len(code_vocab),
                                             args.emb_dim,
                                             args.hid_dim,
                                             args.n_layers,
                                             args.n_heads,
                                             args.dropout,
                                             code_pad_idx,
                                             device)

    desc_encoder = models.TransformerEncoder(len(desc_vocab),
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

def initialize_parameters(m):
    if isinstance(m, nn.Embedding):
        init.xavier_uniform_(m.weight.data)
    elif isinstance(m, (nn.GRU, nn.LSTM)):
        for n, p in m.named_parameters():
            if 'weight_ih' in n:
                init.xavier_uniform_(p.data)
            elif 'weight_hh' in n:
                init.orthogonal_(p.data)
            elif 'bias' in n:
                p.data.fill_(0)
            else:
                print(n)
    elif isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

if args.model == 'bow':

    def initialize_parameters(m):
        if isinstance(m, nn.Embedding):
            init.xavier_uniform_(m.weight.data)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)
        else:
            raise ValueError

    code_encoder.apply(initialize_parameters)
    desc_encoder.apply(initialize_parameters)
    code_pooler.apply(initialize_parameters)
    desc_pooler.apply(initialize_parameters)

if args.model == 'transformer':

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

    code_encoder.apply(initialize_parameters)
    desc_encoder.apply(initialize_parameters)
    code_pooler.apply(initialize_parameters)
    desc_pooler.apply(initialize_parameters)

else:
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

if args.load:

    if args.model == 'transformer':

        code_load_path = os.path.join(code_load_path, 'language_model_model.pt')
        desc_load_path = os.path.join(desc_load_path, 'language_model_model.pt')

        code_encoder.load_state_dict(torch.load(code_load_path))
        desc_encoder.load_state_dict(torch.load(desc_load_path))

    else:
        raise ValueError

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

        code_mask = utils.make_mask(code, code_vocab[PAD_TOKEN])
        desc_mask = utils.make_mask(desc, desc_vocab[PAD_TOKEN])

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

            code_mask = utils.make_mask(code, code_vocab[PAD_TOKEN])
            desc_mask = utils.make_mask(desc, desc_vocab[PAD_TOKEN])

            encoded_code = code_encoder(code)
            encoded_code = code_pooler(encoded_code, code_mask)

            encoded_desc = desc_encoder(desc)
            encoded_desc = desc_pooler(encoded_desc, desc_mask)

            loss, mrr = criterion(encoded_code, encoded_desc)

            epoch_loss += loss.item()
            epoch_mrr += mrr.item()

    return epoch_loss / len(iterator), epoch_mrr / len(iterator)

best_valid_mrr = 0
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

    if valid_mrr > best_valid_mrr:
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