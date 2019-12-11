import torch

x = torch.randn(10, 5)

print(x)

labels = torch.LongTensor([1,2,3,3,0,0,0,0,0,0])

n_classes = x.shape[-1]

one_hot = torch.nn.functional.one_hot(labels, n_classes)

print(one_hot)

print(x * one_hot)

compare = (x * one_hot).sum(-1).unsqueeze(-1).repeat(1, n_classes)

print(compare)

compared_scores = x >= compare
print(compared_scores)
rr = 1 / compared_scores.float().sum(-1)
print(rr)
mrr = rr.mean()
print(mrr)