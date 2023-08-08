import gymnasium as gym
import numpy as np

class RdPopGym:
    STATE_RANGES = {
        "li": (0.5, 6.0),       # Normalized internal inductance [-]
        "Ip_MA": (1.0, 9.0),    # Plasma current [MA]
        "vc_minus_vb": (-5.0, 5.0),       # vc-vb as defined by Romero [V]
        "Wth": (1e5, 3e7),      # Stored thermal energy [J]
        "nfuel19_vol": (1.0, 30.0),  # Volume averaged fuel density [10^19 m^-3]
        "Paux": (0.0, 25.0e6),    # Auxiliary heating power [MW]
        "gs": (0.0, 1.0),     # Geometry evolution parameter [0]
        "HMode": (0, 1)         # H-mode flag [0, 1]
    }

    """ 
    When generating a random initial state, we allow each state variable
    to vary by a certain percentage of its nominal value. If the variable
    is not included in this dict, then it is not varied.
    """
    RANDOM_INITIAL_STATE_PERCENT_VAR = {
        "li": 10.0,
        "Ip_MA": 2.5,
        "vc_minus_vb": 10.0,
        "Wth": 2.0,
        "nfuel19_vol": 2.0
    }
    ACTION_RANGES = {
        "dIp_dt": (-3.0, 0.0),   #
        "dPaux_dt": (-5.0e6, 5.0e6), #
        "fueling19": (0.0, 155.0), # Ad-hoc calculation [10^19/s]
        "dgs_dt": (0.0, 1.0)   # Rate of evolution through geometry space [1/s]
    }

    RANDOM_PARAM_RANGES = {
        "MAIN_ION_DILUTION":,
        "Zeff":,
        "HL_FUDGE":,

    }

    def __init__(self):
        self.nominal_initial_continuous_state = {
            "li": 0.757764,
            "Ip_MA": 8.7,
            "vc_minus_vb": 0.153183,
            "Wth": 2.482841e07,
            "nfuel19_vol": 27.0,
            "Paux": 14.0,
            "gs": 0.0,
            "HMode": True
        }

        self.action_space = gym.spaces.Box(
            low=-1.0*np.ones(len(self.ACTION_RANGES.keys())),
            high=np.ones(len(self.ACTION_RANGES.keys())),
            dtype=np.double,
        )

    def random_initial_state(self):
        """
        Generate a random initial state.
        """
        state = self.nominal_initial_continuous_state
        for var, percent_variation in self.RANDOM_INITIAL_STATE_PERCENT_VAR.items():
            fractional_variation = 0.01 * percent_variation * self.np_random.uniform(low=-1.0, high=1.0)
            state[var] = state[var] + fractional_variation * state[var]
        return state

    def reset(self, seed=None, options=None):
        # Use the following to seed self.np_random
        super().reset(seed=seed)

        # Reset time.
        self.time = 0.0

        """
        Compute a random initial state.
        """
        self.state = self.random_initial_state()

        info = {}

        return self.state_to_obs(self.state), info


    def step(self, state, action):
        pass