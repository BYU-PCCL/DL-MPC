import numpy as np
import torch
from matplotlib import pyplot as plt

from CustomDatasets import LookaheadDataset
from FCNetwork import instantiate_model
from load_data import load_error_data, load_simulated_data

device = "cuda" if torch.cuda.is_available() else "cpu"


def example_inference(n_ahead=100):
    # load models
    print("Load models")
    model_simulated = instantiate_model("simulated", device)
    model_error = instantiate_model("error", device)

    # load data
    print("Load data")
    simulated_actions, simulated_states = load_simulated_data()
    hardware_actions, hardware_states = load_error_data()

    # create datasets
    simulated_dataset = LookaheadDataset(
        states=simulated_states, actions=simulated_actions, n_ahead=n_ahead
    )
    hardware_dataset = LookaheadDataset(
        states=hardware_states, actions=hardware_actions, n_ahead=n_ahead
    )

    print("Starting inference...")
    for i in range(10):
        save_plot_inference(
            simulated_dataset,
            "Simulated First-Principles",
            f"plots/simulated{i}.png",
            model_simulated,
            model_error,
        )
        save_plot_inference(
            hardware_dataset,
            "Actual Hardware Data",
            f"plots/hardware{i}.png",
            model_simulated,
            model_error,
        )
    print("Plots saved to ./plots/")


def save_plot_inference(dataset, suptitle, filename, model_simulated, model_error):
    """
    Simulates on a random sample of the dataset and saves plots.
    """
    # run
    actions, state0, deltas = dataset[np.random.randint(len(dataset))]
    actions, state0, deltas = (
        torch.Tensor(actions).to(device),
        torch.Tensor(state0).to(device),
        torch.Tensor(deltas).to(device),
    )
    state_est_sim = []
    state_est_both = []
    s_sim, s_both = state0, state0
    with torch.no_grad():
        for i in range(len(deltas)):
            # both models want batch as first dimension, reshape so batch=1
            d_sim = model_simulated(actions[i].reshape(1, -1), s_sim.reshape(1, -1))
            d_both = model_simulated(
                actions[i].reshape(1, -1), s_both.reshape(1, -1)
            ) + model_error(actions[i].reshape(1, -1), s_both.reshape(1, -1))
            # d_both = model_simulated(actions[i].reshape(1,-1), s_both.reshape(1,-1))

            d_sim = d_sim.reshape(-1)
            d_both = d_both.reshape(-1)

            s_sim = s_sim + d_sim
            s_both = s_both + d_both

            state_est_sim.append(s_sim.cpu().numpy())
            state_est_both.append(s_both.cpu().numpy())

    # to numpy
    state_est_sim = np.array(state_est_sim)
    state_est_both = np.array(state_est_both)
    state0 = state0.cpu().numpy()

    state_actual = deltas.cpu().numpy() + state0

    plt.figure(figsize=(12, 6))
    plt.suptitle(suptitle)
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        if i == 0:
            plt.plot(state_actual[:, i], label="Actual")
            plt.plot(state_est_sim[:, i], label="Simulated Model")
            plt.plot(state_est_both[:, i], label="Simulated + Error Model")
            plt.legend()
        else:
            plt.plot(state_actual[:, i])
            plt.plot(state_est_sim[:, i])
            plt.plot(state_est_both[:, i])
        plt.title(f"state_{i}")
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    example_inference()
