import torch.nn.functional as F


def cross_entropy(pred, label):
    return F.cross_entropy(pred, label)
