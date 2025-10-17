import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import itertools

esp = 1e-8

def fidelity_loss(y_pred, y):
    """prediction monotonicity related loss"""
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
