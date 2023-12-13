import matplotlib.pyplot as plt
import numpy as np


def _sig(x, c1, c2):
    return 1 / (1 + np.exp(-c1 * (x - c2)))


def _cost_barrier(val: float, limit: float):
    norm_value = val / limit
    clip_norm_value = np.clip(norm_value, 0.0, 1.1)
    return -np.log(1 - _sig(clip_norm_value, 100, 0.95))


def main():
    b_xs = np.linspace(0.0, 2.0, num=1024)

    b_ys0 = _cost_barrier(b_xs, 0.5)
    b_ys1 = _cost_barrier(b_xs, 0.9)
    b_ys2 = _cost_barrier(b_xs, 1.3)

    figsize = 0.8 * np.array([4, 3])
    fig, ax = plt.subplots(layout="constrained", figsize=figsize, dpi=400)
    ax.plot(b_xs, b_ys0, alpha=0.3, color="C1")
    ax.plot(b_xs, b_ys1, alpha=0.55, color="C1")
    ax.plot(b_xs, b_ys2, alpha=0.95, color="C1")
    ax.set(xlabel="Value", ylabel="Cost")

    ax.set_facecolor("#F0F0F0")

    fig.savefig("plots/barrier.pdf", bbox_inches="tight")
    fig.savefig("plots/barrier.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
