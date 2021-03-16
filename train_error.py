import glob
import os
import pdb
from time import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from CustomDatasets import LookaheadDataset
from FCNetwork import Network
from load_data import load_error_data
from loss_function import error_loss_function as loss_function

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(params):
    """
    Trains error model.
    Arguments:
        params (dict): hyperparameters with which to train
    """
    p, x = load_error_data()

    # calculate means
    p_mean = p.mean(axis=(0, 1))
    p_std = p.std(axis=(0, 1))

    x_mean = x.mean(axis=(0, 1))
    x_std = x.std(axis=(0, 1))

    # TODO - does this make sense?
    # delta = x[:,2::2] - x[:,:-2:2]
    # the number to look ahead
    delta = x[:, 1:] - x[:, :-1]
    delta_mean = delta.mean(axis=(0, 1))
    delta_std = delta.std(axis=(0, 1))

    # send to torch tensors
    p_mean, p_std = torch.Tensor(p_mean).to(device), torch.Tensor(p_std).to(device)
    x_mean, x_std = torch.Tensor(x_mean).to(device), torch.Tensor(x_std).to(device)
    delta_mean, delta_std = (
        torch.Tensor(delta_mean).to(device),
        torch.Tensor(delta_std).to(device),
    )

    # parameters
    buffer_size = int(params["buffer size"])
    activation = params["activation"]

    # train val split
    training_split = 0.8
    n = len(p)
    k = int(np.ceil(n * training_split))
    train_p, val_p = p[:k], p[k:]
    train_x, val_x = x[:k], x[k:]

    n_ahead = 1
    train_dataset = LookaheadDataset(states=train_x, actions=train_p, n_ahead=n_ahead)
    val_dataset = LookaheadDataset(states=val_x, actions=val_p, n_ahead=n_ahead)

    action_size = len(train_dataset[0][0][0])
    state_size = len(train_dataset[0][1])
    output_size = len(train_dataset[0][2][0])

    model_path = params.get("model path", None)
    dropout = params["dropout"]
    hidden_layers = int(params["hidden layers"])
    hidden_size = int(params["hidden size"])

    # LOAD ANALYTICAL MDOEL
    analytical_model = Network(
        action_size=action_size,
        state_size=state_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        activation=activation,
        action_mean=p_mean,
        action_std=p_std,
        state_mean=x_mean,
        state_std=x_std,
        output_mean=delta_mean,
        output_std=delta_std,
    )

    analytical_model.to(device)
    analytical_path = params["analytical model path"]
    analytical_model.load_state_dict(torch.load(analytical_path))

    model = Network(
        action_size=action_size,
        state_size=state_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        activation=activation,
        action_mean=p_mean,
        action_std=p_std,
        state_mean=x_mean,
        state_std=x_std,
        output_mean=delta_mean,
        output_std=delta_std,
    )

    model.to(device)
    if params.get("load", False):
        model.load_state_dict(torch.load(model_path))

    learning_rate = params["learning rate"]
    batch_size = int(params["batch size"])

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    train_losses = []
    val_losses = []

    best_loss = np.inf
    print_info = params.get("print", False)

    epochs = int(params["epochs"])
    max_batches = np.inf
    if print_info:
        loop = tqdm(total=min(len(train_dataloader), max_batches) * epochs)

    def step(state, deltas):
        s = state + deltas
        return s

    for epoch in range(epochs):
        model.train()
        # new_n_ahead = min((epoch + 1) * 5, 100)
        new_n_ahead = 10
        if new_n_ahead != n_ahead:
            n_ahead = new_n_ahead
            if print_info:
                print(n_ahead)
            train_dataset = LookaheadDataset(
                states=train_x, actions=train_p, n_ahead=n_ahead
            )
            val_dataset = LookaheadDataset(states=val_x, actions=val_p, n_ahead=n_ahead)
            train_dataloader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=True
            )
        for b, (a, s, d) in enumerate(train_dataloader):
            s = s.float().to(device)
            a = a.float().to(device)
            d = d.float().to(device)

            d_est = torch.zeros(d.shape).to(device)

            for i in range(n_ahead):
                d_hat = model(a[:, i], s) + analytical_model(a[:, i], s)
                if i == 0:
                    # d_est[:,i] = d_est[:,i] + d_hat
                    d_est[:, i] = d_hat
                else:
                    d_est[:, i] = d_est[:, i - 1] + d_hat
                s = s + d_hat

            # normalize d
            d = (d - delta_mean) / delta_std
            d_est = (d_est - delta_mean) / delta_std

            loss = loss_function(d, d_est)
            if print_info:
                if not val_losses:
                    loop.set_description("loss: {:.3f}".format(loss.item()))
                else:
                    loop.set_description(
                        "loss: {:.4f}, val loss: {:.4f}".format(
                            loss.item(), val_losses[-1]
                        )
                    )
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if print_info:
                loop.update(1)
            if b > max_batches:
                break
        with torch.no_grad():
            model.eval()
            epoch_losses = []
            for b, (a, s, d) in enumerate(val_dataloader):
                s = s.float().to(device)
                a = a.float().to(device)
                d = d.float().to(device)

                d_est = torch.zeros(d.shape).to(device)

                for i in range(n_ahead):
                    d_hat = model(a[:, i], s) + analytical_model(a[:, i], s)
                    if i == 0:
                        # d_est[:,i] = d_est[:,i] + d_hat
                        d_est[:, i] = d_hat
                    else:
                        d_est[:, i] = d_est[:, i - 1] + d_hat
                    s = s + d_hat

                # normalize d
                d = (d - delta_mean) / delta_std
                d_est = (d_est - delta_mean) / delta_std

                loss = loss_function(d, d_est)

                epoch_losses.append(loss.item())
                if b > max_batches:
                    break
            val_losses.append(np.mean(epoch_losses))

            if np.mean(epoch_losses) < best_loss:
                best_loss = np.mean(epoch_losses)
                if model_path:
                    torch.save(model.state_dict(), model_path)
                if print_info:
                    print("Best val loss: {:.4}".format(best_loss))
    n_ahead = 100
    val_dataset = LookaheadDataset(states=val_x, actions=val_p, n_ahead=n_ahead)
    val_dataloader = DataLoader(val_dataset, batch_size=100, shuffle=True)

    # calculate HZ
    start = time()
    with torch.no_grad():
        model.eval()
        for b, (a, s, d) in enumerate(val_dataloader):
            s = s.float().to(device)
            a = a.float().to(device)
            d = d.float().to(device)

            d_est = torch.zeros(d.shape).to(device)

            for i in range(n_ahead):
                d_hat = model(a[:, i], s) + analytical_model(a[:, i], s)
                if i == 0:
                    # d_est[:,i] = d_est[:,i] + d_hat
                    d_est[:, i] = d_hat
                else:
                    d_est[:, i] = d_est[:, i - 1] + d_hat
                s = s + d_hat
    elapsed = time() - start
    speed = elapsed / len(val_dataloader)
    return val_losses[-1].item(), speed


if __name__ == "__main__":
    from hyperparams import *

    print(train(params_error))
