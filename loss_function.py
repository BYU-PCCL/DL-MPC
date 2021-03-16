import torch
from torch import nn


def loss_function(d, d_hat):
    # L2 = (d - d_hat).pow(2).mean()
    L1 = nn.L1Loss()(d, d_hat).mean()
    start = (d[:, 0] + d_hat[:, 0]) / 2
    start = start.reshape(-1, 1, 8)
    cosine_distance = 1 - nn.CosineSimilarity()(d - start, d_hat - start).mean()
    return cosine_distance + L1


def error_loss_function(d, d_hat):
    # L2 = (d - d_hat).pow(2).mean()
    L1 = nn.L1Loss()(d, d_hat).mean()
    start = (d[:, 0] + d_hat[:, 0]) / 2
    start = start.reshape(-1, 1, 8)
    return L1
