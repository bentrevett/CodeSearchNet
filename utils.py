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

class SoftmaxLossRet(nn.Module):
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
            mrr = mrr_metric_ret(similarity)

        return loss, mrr

def mrr_metric_ret(similarity):
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

        labels = labels.squeeze(0)

        loss = F.cross_entropy(enc_code, labels)

        with torch.no_grad():
            mrr = mrr_metric_pred(enc_code, labels)

        return loss, mrr

def mrr_metric_pred(enc_code, labels):
    n_classes = enc_code.shape[-1]
    one_hot = F.one_hot(labels, n_classes)
    actual_score = (enc_code * one_hot).sum(-1).unsqueeze(-1).repeat(1, n_classes)
    compared_scores = enc_code >= actual_score
    rr = 1/compared_scores.float().sum(-1)
    mrr = rr.mean()
    return mrr

def make_mask(sequence, pad_idx):
    mask = (sequence != pad_idx).permute(1, 0)
    return mask

def truncated_normal_(tensor, mean=0, std=1):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)

def initialize_transformer(m):
    if isinstance(m, nn.LayerNorm):
        pass
    elif hasattr(m, 'weight'):
        truncated_normal_(m.weight.data, std=0.02)