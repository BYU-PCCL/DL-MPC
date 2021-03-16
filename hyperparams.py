from torch import nn

params_simulated = {
    "print": True,
    "model path": "models/model_simulated.pth",
    "dropout": 0.396,
    "hidden layers": 2,
    "hidden size": 505,
    "buffer size": 1,
    "epochs": 30,
    "learning rate": 4.79e-4,
    "batch size": 128,
    "activation": nn.LeakyReLU,
    "load": False,
}

params_error = params_simulated.copy()
params_error["analytical model path"] = params_simulated["model path"]
params_error["model path"] = "models/model_error.pth"
