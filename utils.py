import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import json

def get_run_name(args):
    return [f'{param}={val}' for param, val in vars(args).items()]

def load_vocab(path, max_size, pad_token, unk_token):
    vocab = {pad_token: 0 , unk_token: 1}
    with open(path, 'r') as f:
        for i, tok in enumerate(f):
            tok = tok.split('\t')[0]
            if i >= max_size:
                return vocab
            if tok not in vocab:
                vocab[tok.strip()] = len(vocab)
    return vocab

def numericalize(vocab, unk_token, max_length, tokens):
    idxs = [vocab.get(t, vocab[unk_token]) for t in tokens[:max_length]]
    return idxs

def count_parameters(models):
    if isinstance(models, list):
        return sum([count_parameters(model) for model in models])
    else:
        return sum(p.numel() for p in models.parameters() if p.requires_grad)

def load_lm_data(path, tokenizer, data):
    if data == 'desc':
        data = 'docstring'
    all_tokens = []
    with open(path, 'r') as f:
        for line in f:
            tokens = json.loads(line)
            tokens = tokens[f'{data}_tokens']
            tokens = tokenizer(tokens)
            all_tokens += tokens
    return torch.LongTensor(all_tokens)

def batchify(data, batch_size):
    # Work out how cleanly we can divide the dataset into bsz parts.
    n_batches = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, n_batches * batch_size)
    # Evenly divide the data across the bsz batches.
    data = data.view(batch_size, -1).t().contiguous()
    return data

class SoftmaxLoss(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, enc_code, enc_desc):

        #enc_code = [batch size, enc dim]
        #enc_desc = [batch size, enc dim]

        enc_desc = enc_desc.permute(1, 0)

        #enc_desc = [enc dim, batch size]

        similarity = torch.matmul(enc_code, enc_desc)

        #similarity = [batch size, batch size]

        classes = torch.arange(similarity.shape[0]).to(self.device)

        loss = F.cross_entropy(similarity, classes)

        with torch.no_grad():
            mrr = mrr_metric(similarity)

        return loss, mrr

def mrr_metric(similarity):
    correct_scores = torch.diagonal(similarity)
    compared_scores = similarity >= correct_scores.unsqueeze(-1)
    rr = 1 / compared_scores.float().sum(-1)
    mrr = rr.mean()
    return mrr

class SoftmaxLossPred(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, enc_code, labels):

        #enc_code = [batch size, out dim]
        #labels = [batch size]

        loss = F.cross_entropy(enc_code, labels)

        with torch.no_grad():
            mrr = mrr_metric_pred(enc_code, labels)

def mrr_metric_pred(enc_code, labels):
    correct_scores = 

def make_mask(sequence, pad_idx):
    mask = (sequence != pad_idx).permute(1, 0)
    return mask

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

def handle_args(args):

    if args.model == 'bow':
        del args.hid_dim
        del args.n_layers
        del args.bidirectional
        del args.filter_size
        del args.n_heads

    elif args.model in ['gru', 'lstm']:
        del args.filter_size
        del args.n_heads

    elif args.model == 'cnn':
        del args.hid_dim
        del args.bidirectional
        del args.n_heads

    elif args.model == 'transformer':
        del args.bidirectional
        del args.filter_size

    else:
        raise ValueError(f'Model {args.model} not valid!')

    if args.bpe_pct > 0:
        del args.vocab_min_freq

    return args

if __name__ == '__main__':

    x = torch.randn(3,3)

    print(x)

    mrr_metric(x)