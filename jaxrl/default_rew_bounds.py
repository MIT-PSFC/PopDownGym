def get_default_rew_bounds():
    rew_bounds = {
        "li": [2, 3],
        "ng_frac": [0.5, 0.8],
        "beta_n": [0.015, 0.028],
        "beta_p": [0.25, 0.4],
        "Bv_dot_mag": [0.2, 0.4],
        "Wdot_mag": [20_000_000, 70_000_000],
        "shafranov_coeff": [3.4, 3.6],
        "iota95": [0.35, 0.45],
    }
    rew_centers = {k: 0.5 * (v[0] + v[1]) for k, v in rew_bounds.items()}
    shift_ranges = {k: 0.5 * (v[1] - v[0]) for k, v in rew_bounds.items()}
    rew_min = {k: v[0] for k, v in rew_bounds.items()}
    rew_max = {k: v[1] for k, v in rew_bounds.items()}

    return rew_centers, shift_ranges, rew_min, rew_max
