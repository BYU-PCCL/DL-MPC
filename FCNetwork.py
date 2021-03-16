import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from hyperparams import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class Network(nn.Module):
    """ Network which takes as input a flattened buffer of actions and states, and predicts the derivitative. Scaling is done in network """

    def __init__(
        self,
        action_size,
        state_size,
        output_size,
        hidden_layers=3,
        hidden_size=64,
        dropout=0,
        state_mean=None,
        state_std=None,
        action_mean=None,
        action_std=None,
        output_mean=None,
        output_std=None,
        activation=nn.ReLU,
    ):
        """
        Input:
            action_size (int): size of action space
            state_size (int): size of state space
            output_size (int): size of output. Normally, same size as state space
            hidden_layers (int): number of hidden layers
            hidden_size (int): size of hidden nsize
            means, stds (torch.Tensor, optional): expected of same size as a single instance of action, state, or output.
                Before forward, subtracts mean from the state/action and divides by std.
                After forward, multiplies output by std and adds mean.
                If no means or stds are provided, do not touch data.
        """
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(action_size + state_size, hidden_size), activation()
        )
        self.hidden = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    activation(),
                )
            ]
        )
        self.out = nn.Linear(hidden_size, output_size)

        self.action_size = action_size
        self.state_size = state_size

        if state_mean == None or state_std == None:
            state_mean = torch.zeros(state_size)
            state_std = torch.ones(state_size)
        # if multiple steps in buffer, scale first
        if len(state_mean) != state_size:
            n_repeat = state_size // len(state_mean)
            state_mean = state_mean.repeat(n_repeat)
            state_std = state_std.repeat(n_repeat)
        self.state_mean = nn.Parameter(state_mean)
        self.state_mean.requires_grad = False
        self.state_std = nn.Parameter(state_std)
        self.state_std.requires_grad = False

        if action_mean == None or action_std == None:
            action_mean = torch.zeros(action_size)
            action_std = torch.ones(action_size)
        # if multiple steps in buffer, scale first
        if len(action_mean) != action_size:
            n_repeat = action_size // len(action_mean)
            action_mean = action_mean.repeat(n_repeat)
            action_std = action_std.repeat(n_repeat)
        self.action_mean = nn.Parameter(action_mean)
        self.action_mean.requires_grad = False
        self.action_std = nn.Parameter(action_std)
        self.action_std.requires_grad = False

        if output_mean == None:
            output_mean = torch.zeros(output_size)
        if output_std == None:
            output_std = torch.ones(output_size)
        # if multiple steps in buffer, scale first
        if len(output_mean) != output_size:
            n_repeat = output_size // len(output_mean)
            output_mean = output_mean.repeat(n_repeat)
        if len(output_std) != output_size:
            n_repeat = output_size // len(output_std)
            output_std = output_std.repeat(n_repeat)
        self.output_mean = nn.Parameter(output_mean)
        self.output_mean.requires_grad = False
        self.output_std = nn.Parameter(output_std)
        self.output_std.requires_grad = False

    def forward(self, a, s):
        """
        Input:
            a - torch.Tensor of shape (batch_size, action_size * buffer_size). It expects the actions in the flattened order (a_t-buffer_size+1, ..., a_t-1, a_t)
            s - torch.Tensor of shape (batch_size, state_size * buffer_size). It expects the states in the flattened order (s_t-buffer_size+1, ..., s_t-1, s_t)

            Please feed in UNNORMALIZED data. The normalization happens automatically

        Output:
            derivative_hat - torch.Tensor of shape (batch_size, derivative_size). Estimated derivative at time t+1

            The output is projected back into the original space and needs no normalizing (unless calculating loss - then divide the target and estimate by output_std and add output_mean)
        """

        # scale
        a = (a - self.action_mean) / self.action_std
        # scale
        s = (s - self.state_mean) / self.state_std

        # run through network
        x = torch.cat([a, s], dim=1)
        x = self.linear(x)
        x = self.hidden(x)
        x = self.out(x)

        # scale
        x = (x * self.output_std) + self.output_mean
        return x


def instantiate_model(name, device, MPC_directory=""):
    """
    Automatically reads in hyperparameters and instantiates model.
    Arguments:
        name (str): name of model. Allowed names: 'simulated', 'error'
        device (str): device on which to load model. Allowed devices: 'cuda', 'cpu'
        MPC_directory (str): path to location of model parameter files
    Returns:
        model (FCNetwork.Network): trained model
    """
    allowed_names = ["hardware", "simulated", "finetune", "error"]
    assert name in allowed_names, "Name must be 'hardware', 'simulated', or 'finetune'"
    if name == "simulated":
        params = params_simulated
    elif name == "error":
        params = params_error

    activation = params["activation"]
    model_path = MPC_directory
    model_path += params["model path"]
    dropout = params["dropout"]
    hidden_layers = int(params["hidden layers"])
    hidden_size = int(params["hidden size"])

    action_size = 4
    state_size = 8
    output_size = 8

    model = Network(
        action_size=action_size,
        state_size=state_size,
        output_size=output_size,
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        activation=activation,
    )

    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_simulated = instantiate_model("simulated", device)
    model_error = instantiate_model("error", device)
    print("Loaded models")
