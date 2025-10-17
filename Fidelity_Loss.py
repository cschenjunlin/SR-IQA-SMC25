import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import itertools

esp = 1e-8

def fidelity_loss(y_pred, y):
    """ Predicts monotonicity related loss.
    Args:
        y_pred: Tensor of shape [bs]
        y: Tensor of shape [bs]
    """
    y_pred = y_pred.unsqueeze(1)
    y = y.unsqueeze(1)

    preds = y_pred - y_pred.t()
    gts = y - y.t()

    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]
    g = 0.5 * (torch.sign(gts) + 1)

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))

    g = g.view(-1, 1)
    p = p.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss


def fidelity_loss_majority_voting(y_pred, y):
    """ Predicts monotonicity related loss.
    Args:
        y_pred: Tensor of shape [bs]
        y: Tensor of shape [bs, k] (k = 14, num of voting models.)
    """
    # prediction matrix
    y_pred = y_pred.unsqueeze(1)
    preds = y_pred - y_pred.t()

    # ground-truth matrix
    bs, k = y.shape
    y_i = y.unsqueeze(1)  # [bs, 1, k]
    y_j = y.unsqueeze(0)  # [1, bs, k]

    win = (y_i > y_j).sum(dim=2)  # [bs, bs]: vote_i > vote_j
    lose = (y_i < y_j).sum(dim=2)  # [bs, bs]: vote_i < vote_j

    # gts = torch.zeros((bs, bs), device=y.device)
    # gts[win > lose] = 1
    # gts[win < lose] = -1
    gts = win / (win + lose)

    # flatten
    triu_indices = torch.triu_indices(y_pred.size(0), y_pred.size(0), offset=1)
    preds = preds[triu_indices[0], triu_indices[1]]
    gts = gts[triu_indices[0], triu_indices[1]]

    constant = torch.sqrt(torch.Tensor([2.])).to(preds.device)
    p = 0.5 * (1 + torch.erf(preds / constant))
    # g = 0.5 * (torch.sign(gts) + 1)
    g = gts

    p = p.view(-1, 1)
    g = g.view(-1, 1)

    loss = torch.mean((1 - (torch.sqrt(p * g + esp) + torch.sqrt((1 - p) * (1 - g) + esp))))

    return loss
