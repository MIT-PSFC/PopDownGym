import jax
import jax.numpy as jnp


def sigmoid(x: float, c1: float, c2: float) -> float:
    """Sigmoid where "cutoff" is c2 and "slope" is c1.

    Args:
        x (float): sigmoid input.
        c1 (float): slope parameter.
        c2 (float): cutoff parameter.

    Returns:
        float: sigmoid output.
    """
    return 1.0 / (1.0 + jnp.exp(-c1 * (x - c2)))


class RewardModel:
    def __init__(self, reward_params) -> None:
        self.params = reward_params
        self.limits = reward_params["limits"]
        self.barrier = reward_params["barrier"]
        self.ip_ma = reward_params["ip_ma"]

    def reward(self, reward_inputs, action):
        reward = 0.0
        hit_goal = reward_inputs["Ip_MA"] <= self.ip_ma["target"]
        reward_terms = {
            "li": self.reward_barrier(reward_inputs["li"], self.limits["li"]),
            "ng_frac": self.reward_barrier(
                reward_inputs["ng_frac"], self.limits["ng_frac"]
            ),
            "beta_n": self.reward_barrier(
                reward_inputs["beta_n"], self.limits["beta_n"]
            ),
            "beta_p": self.reward_barrier(
                reward_inputs["beta_p"], self.limits["beta_p"]
            ),
            "Bv_dot_mag": self.reward_barrier(
                reward_inputs["Bv_dot_mag"], self.limits["Bv_dot_mag"]
            ),
            "Wdot_mag": self.reward_barrier(
                reward_inputs["Wdot_mag"], self.limits["Wdot_mag"]
            ),
            "shafranov_coeff": self.reward_barrier(
                reward_inputs["shafranov_coeff"], self.limits["shafranov_coeff"]
            ),
            "iota95": self.reward_barrier(
                reward_inputs["iota95"], self.limits["iota95"]
            ),
            "Ip": self.ip_reward(reward_inputs["Ip_MA"]),
            "hit_goal_reward": jax.lax.cond(
                hit_goal, lambda _: self.params["hit_goal_reward"], lambda _: 0.0, None
            ),
        }

        jreward = jnp.array(jax.tree_util.tree_leaves(reward_terms))
        reward = jnp.sum(jreward)
        return reward, reward_terms

    def ip_reward(self, Ip_MA: float) -> float:
        # Idea: always negative reward for Ip > 0.0.
        abs_error = jnp.abs(Ip_MA)
        x = (
            self.ip_ma["abs_error_mult"] * abs_error
        )  # An attempt to get steeper reward gradients.
        # Apply a funky logistic kernel thing.
        mult_factor = jnp.abs(self.ip_ma["min_reward"] / 0.25)
        reward = self.ip_ma["min_reward"] + mult_factor / (
            jnp.exp(x) + 2.0 + jnp.exp(-x)
        )
        return reward

    def reward_barrier(self, val: float, limit: float) -> float:
        """To enforce the constraint val < limit, we use a barrier function.
        The general idea is to have "risk" go from 0 to 1.0 as val approaches limit.
        We first normalize the value to the limit, then apply a sigmoid function.
        The sigmoid parameters are chosen s.t. at "BARRIER_THRESH", the sigmoid goes
        sharply from 0 to 1.0. To convert to reward, we take the log of 1.0 - sigmoid.
        For example, if 0.95 is the barrier threshold, then the reward will be approx
        0 until val is within 5% of the limit, then it will sharply go downwards.

        Args:
            val (float): value to be constrained.
            limit (float): limit to be enforced.
        """

        # Clip val at above above limit to avoid numerical issues.
        normalized_value = jnp.abs(val / limit)
        clipped_normalized_value = jnp.clip(normalized_value, 0.0, 1.1)
        return jnp.log(
            1.0
            - sigmoid(
                clipped_normalized_value, self.barrier["slope"], self.barrier["thresh"]
            )
        )
