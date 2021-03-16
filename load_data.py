import numpy as np


def change_time_step(x, n):
    """
    Changes data to different time step frequency.
    Arguments:
        x (np.ndarray): data to change, size: (rollouts, total_time, data_dim)
        n (int): number of steps to skip
    Returns:
        x (np.ndarray): modified data, size: (rollouts*n, total_time//n, data_dim)
    """
    return np.vstack([x[:, i::n] for i in range(n)])


def load_error_data():
    """
    Load step input hardware data.
    returns:
        p (np.ndarray): actions (pressures), size: (rollout, time, action)
        x (np.ndarray): states, size: (rollout, time, state)
    """
    p, x, = (
        None,
        None,
    )
    for i in range(0, 7):
        pi = np.loadtxt(
            f"data/error/error_hardware_trajectory{i}_inputs.csv", delimiter=","
        )
        xi = np.loadtxt(
            f"data/error/error_hardware_trajectory{i}_states.csv", delimiter=","
        )
        if p is None:
            p, x, = (
                pi,
                xi,
            )
            # expand dims
            p = np.expand_dims(p, 0)
            x = np.expand_dims(x, 0)
        else:
            pi = np.expand_dims(pi, 0)
            xi = np.expand_dims(xi, 0)
            p = np.concatenate([p, pi], axis=0)
            x = np.concatenate([x, xi], axis=0)

    # change so (rollout, t, values)
    p = np.swapaxes(p, 1, 2)
    x = np.swapaxes(x, 1, 2)
    return p, x


def load_simulated_data():
    """
    Load simulated data using first principles model.
    returns:
        p (np.ndarray): actions (pressures), size: (rollout, time, action)
        x (np.ndarray): states, size: (rollout, time, state)
    """
    p, x, = (
        None,
        None,
    )
    for i in range(1, 13):
        pi = np.loadtxt(f"data/analytical/p{i}.csv", delimiter=",")
        xi = np.loadtxt(f"data/analytical/x{i}.csv", delimiter=",")
        if p is None:
            p, x, = (
                pi,
                xi,
            )
            # expand dims
            p = np.expand_dims(p, 0)
            x = np.expand_dims(x, 0)
        else:
            pi = np.expand_dims(pi, 0)
            xi = np.expand_dims(xi, 0)
            p = np.concatenate([p, pi], axis=0)
            x = np.concatenate([x, xi], axis=0)

    # change so (rollout, t, values)
    p = np.swapaxes(p, 1, 2)
    x = np.swapaxes(x, 1, 2)
    n = 20
    p = change_time_step(p, n)
    x = change_time_step(x, n)
    return p, x


if __name__ == "__main__":
    # print('Derivative')
    # p, x, xdot = load_derivative_data()
    print("Simulated")
    p, x = load_simulated_data()
    print(p.shape, x.shape)

    print("Error")
    p, x = load_error_data()
    print(p.shape, x.shape)
    # print('Hardware')
    # p, x = load_hardware_data()
    # print(p.shape, x.shape)
    # # np.set_printoptions(precision=2)
    # # print(x.mean(axis=(0,1)))
    # # print(x.std(axis=(0,1)))
    # print('MPC Hardware')
    # p, x = load_mpc_hardware_data()
    # print(p.shape, x.shape)
