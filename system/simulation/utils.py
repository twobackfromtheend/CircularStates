import datetime

import numpy as np


def get_time_str():
    return datetime.datetime.now().isoformat(timespec='minutes').replace(":", "_")


def tukey(timesteps: int, alpha: float):
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha has to be between 0 and 1 (noninclusive)")

    L = timesteps - 1

    def window_fn(x: float):
        if x <= 0 or x >= L:
            return 0
        if x < alpha * L / 2:
            return 0.5 * (1 - np.cos(2 * np.pi * x / alpha / L))
        elif x > timesteps - alpha * L / 2 - 1:
            return 0.5 * (1 - np.cos(np.pi * (-2 / alpha + 2 * x / alpha / L)))
        else:
            return 1

    return window_fn


if __name__ == '__main__':
    from scipy.signal.windows import tukey as scipytukey
    import matplotlib.pyplot as plt

    timesteps = 501
    alpha = 0.2
    scipy_window = scipytukey(timesteps, alpha=alpha)

    x = np.arange(timesteps)
    plt.plot(x / 100, scipy_window, alpha=0.5)

    window_fn = tukey(timesteps, alpha=alpha)
    window_fn_array = [window_fn(_x) for _x in x]
    window_fn_array = [window_fn(_x) for _x in np.linspace(0, timesteps - 1, 10)]
    plt.plot(np.linspace(0, timesteps - 1, 10) / 100, window_fn_array, '--', alpha=0.5)
    # plt.plot(x / 100, window_fn_array, '--', alpha=0.5)

    # print(np.isclose(window_fn_array, scipy_window).all())
    plt.show()
