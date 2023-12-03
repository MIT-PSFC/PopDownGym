import time

import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from scipy import stats


def main():
    rng = np.random.default_rng(seed=51283)

    batch_size = 8192

    b_c0 = rng.binomial(1, 0.3, batch_size)
    b_x = b_c0 * rng.normal(-0.5, 1.5, size=batch_size) + (1 - b_c0) * rng.normal(1.0, 0.9, size=batch_size)

    def get_stat(x, axis: int = 0):
        print("x.shape: {}, Axis: {}".format(x.shape, axis))
        out = np.quantile(x, 0.80, axis=axis)
        print("out.shape: {}".format(out.shape))
        return out

    rng = np.random.default_rng(seed=5812431)
    data = (b_x,)
    t1 = time.time()
    res = stats.bootstrap(data, get_stat, random_state=rng, vectorized=True)
    t2 = time.time()
    logger.info("Took {:.2f} seconds to bootstrap.".format(t2 - t1))

    stat_all = get_stat(b_x)

    fig, ax = plt.subplots(layout="constrained")
    ax.hist(b_x, bins=64, color="C1")
    ax.axvline(stat_all, label="q80 all", zorder=3)
    ax.axvline(res.confidence_interval.low, ls="--", label="q80 low (95)", zorder=3)
    ax.axvline(res.confidence_interval.high, ls="--", label="q80 high (95)", zorder=3)
    fig.savefig("bootstrap_test.pdf")

    logger.info("CI: {}".format(res.confidence_interval))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
