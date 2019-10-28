import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def count_parameters(models):
    if isinstance(models, list):
        return sum([count_parameters(model) for model in models])
    else:
        return sum(p.numel() for p in models.parameters() if p.requires_grad)

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

        with torch.no_grad():
            acc = acc_metric(similarity, classes)

        return loss, mrr, acc

class CosineLoss(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        raise NotImplementedError('CosineLoss does not work (yet)')

        self.device = device

    def forward(self, enc_code, enc_desc):

        raise NotImplementedError('CosineLoss does not work (yet)')

        #enc_code = [batch size, enc dim]
        #enc_desc = [batch size, enc dim]

        enc_code_norm = torch.norm(enc_code, p=2, dim=-1, keepdim=True) + 1e-10
        enc_desc_norm = torch.norm(enc_desc, p=2, dim=-1, keepdim=True) + 1e-10

        enc_code = enc_code / enc_code_norm
        enc_desc = enc_desc / enc_desc_norm

        similarity = torch.matmul(enc_code, enc_desc.permute(1, 0))

        classes = torch.arange(similarity.shape[0]).to(self.device)

        negative_matrix = torch.diag(torch.zeros(similarity.shape[0]).fill_(-float('inf'))).to(self.device)

        per_sample_loss = 1 - torch.diagonal(similarity) + torch.max(F.relu(similarity + negative_matrix), dim=-1)[0]

        loss = per_sample_loss.mean()

        with torch.no_grad():
            mrr = mrr_metric(similarity)

        with torch.no_grad():
            acc = acc_metric(similarity, classes)

        return loss, mrr, acc

def mrr_metric(similarity):

    correct_scores = torch.diagonal(similarity)

    compared_scores = similarity >= correct_scores.unsqueeze(-1)

    rr = 1 / compared_scores.float().sum(-1)

    mrr = rr.mean()

    return mrr

def acc_metric(similarity, classes):

    max_similarity = similarity.argmax(dim=1)
    correct = max_similarity.eq(classes)
    acc = correct.sum() / torch.FloatTensor([classes.shape[0]])

    return acc

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

if __name__ == '__main__':

    x = torch.randn(3,3)

    print(x)

    mrr_metric(x)