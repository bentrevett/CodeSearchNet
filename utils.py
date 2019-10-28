import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(models):
    if isinstance(models, list):
        return sum([count_parameters(model) for model in models])
    else:
        return sum(p.numel() for p in models.parameters() if p.requires_grad)

class SoftMaxLoss(nn.Module):
    def __init__(self,
                 device):
        super().__init__()

        self.device = device

    def forward(self, enc_code, enc_desc):

        #enc_code = [batch size, enc dim]
        #enc_desc = [batch size, enc dim]

        enc_desc = enc_desc.permute(1, 0)

        #enc_desc = [enc dim, batch size]

        logits = torch.matmul(enc_code, enc_desc)

        #logits = [batch size, batch size]

        with torch.no_grad():
            mrr = mrr_metric(logits)

        classes = torch.arange(logits.shape[0]).to(self.device)

        loss = F.cross_entropy(logits, classes)

        return loss, mrr

def mrr_metric(similarity):

    correct_scores = torch.diagonal(similarity)

    compared_scores = similarity >= correct_scores.unsqueeze(-1)

    rr = 1 / compared_scores.float().sum(-1)

    mrr = rr.mean()

    return mrr