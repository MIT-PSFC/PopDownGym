import matplotlib.pyplot as plt
import numpy as np

from jaxrl.helpers import get_default_rew_bounds


def _sig(x, c1, c2):
    expr1 = 1 / (1 + np.exp(-c1 * (x - c2)))
    expr2 = np.exp(c1 * (x - c2)) / (1 + np.exp(c1 * (x - c2)))
    return np.where(x - c2 >= 0, expr1, expr2)


def _cost_barrier(val: float, limit: float):
    norm_value = val / limit
    # clip_norm_value = np.clip(norm_value, 0.0, 1.1)
    clip_norm_value = np.clip(np.log(norm_value + 1) / np.log(2), 0.0, 1.0)
    return np.log(1 - _sig(clip_norm_value, 100, 0.95))


def _cost_barrier2(val, limit: float, slope: float = 100.):
    norm_value = np.abs(val / limit)
    sat_value = np.log(norm_value + 1) / np.log(2)
    clip_norm_value = sat_value.clip(0.0, 3.0)
    c2 = 0.95
    return -np.logaddexp(0.0, slope * (clip_norm_value - c2))


def main():
    rew_centers, _, _, _ = get_default_rew_bounds()
    limit = rew_centers["Wdot_mag"]

    b_xs = np.linspace(1e7, 7.5e7, num=2048)

    b_ys0 = _cost_barrier(b_xs, limit)
    # b_ys1 = _cost_barrier(b_xs, 0.9)
    # b_ys2 = _cost_barrier(b_xs, 1.3)

    b_zs0 = _cost_barrier2(b_xs, limit)
    b_zs1 = _cost_barrier2(b_xs, limit, slope=150)
    b_zs2 = _cost_barrier2(b_xs, limit, slope=200)

    figsize = 1.2 * np.array([4, 3])
    fig, ax = plt.subplots(layout="constrained", figsize=figsize, dpi=400)
    ax.plot(b_xs, b_ys0, label="Current")
    ax.plot(b_xs, b_zs0, label="New")
    ax.plot(b_xs, b_zs1)
    ax.plot(b_xs, b_zs2)
    # ax.plot(b_xs, b_ys1, alpha=0.55, color="C1")
    # ax.plot(b_xs, b_ys2, alpha=0.95, color="C1")
    ax.set(xlabel="Value", ylabel="Cost")
    ax.legend()

    ax.set_facecolor("#F0F0F0")

    fig.savefig("plots/W_barrier.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
