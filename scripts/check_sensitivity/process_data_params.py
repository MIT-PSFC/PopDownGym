import pathlib
import pickle

import ipdb
import numpy as np
from loguru import logger
from scipy import stats

from jaxrl.utils.jax_types import BBool, BInt
from pop_down_gym.pd_gym_stateless import PopDownGymStateless


def main():
    dt = 0.05

    def _get_quantile_stat(q: float):
        def _fn(x, axis: int = -1):
            out = np.quantile(x, q, axis=axis)
            return out

        return _fn

    pkl_path = pathlib.Path(__file__).parent / "ppo_sens_envparam.pkl"
    with open(pkl_path, "rb") as f:
        data_dict: dict[str, dict[float, tuple[BBool, BInt]]] = pickle.load(f)

    dict_processed = {}
    for envparam_name, data_dict_rew in data_dict.items():
        logger.info("Processing {}...".format(envparam_name))
        C_vals = []
        C_data = []
        for constr_val, (b_has_hit_goal, b_hit_goal_steps) in data_dict_rew.items():
            assert np.all(b_has_hit_goal)
            b_hit_goal_s = b_hit_goal_steps * dt

            # Compute stats
            mean = np.mean(b_hit_goal_s)
            q05, q95 = np.quantile(b_hit_goal_s, 0.05), np.quantile(b_hit_goal_s, 0.95)

            # Get CI for stats.
            rng = np.random.default_rng(seed=1337)
            bs_data = (b_hit_goal_s,)
            res_mean = stats.bootstrap(bs_data, np.mean, vectorized=True, random_state=rng)
            res_q05 = stats.bootstrap(bs_data, _get_quantile_stat(0.05), vectorized=True, random_state=rng)
            res_q95 = stats.bootstrap(bs_data, _get_quantile_stat(0.95), vectorized=True, random_state=rng)

            C_vals.append(constr_val)
            C_data.append(
                [
                    [mean, res_mean.confidence_interval.low, res_mean.confidence_interval.high],
                    [q05, res_q05.confidence_interval.low, res_q05.confidence_interval.high],
                    [q95, res_q95.confidence_interval.low, res_q95.confidence_interval.high],
                ]
            )

        C_vals = np.array(C_vals)
        C_data = np.array(C_data)
        dict_processed[envparam_name] = (C_vals, C_data)

    # Save the processed data.
    pkl_path = pathlib.Path(__file__).parent / "ppo_sens_envparams_processed.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(dict_processed, f)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()
