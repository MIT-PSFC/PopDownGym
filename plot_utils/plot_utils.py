import pathlib

import ipdb
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np


def get_segs(bT_rew, bT_valid_mask, dt: float):
    b_segs = []
    for bb, T_r in enumerate(bT_rew):
        # Truncate to valid only.
        if not np.all(bT_valid_mask[bb]):
            idx_first_invalid = bT_valid_mask[bb].argmin()
            T_r = T_r[:idx_first_invalid]

        T = len(T_r)
        T_ts = dt * np.arange(T)
        # (T, 2)
        segs = np.stack([T_ts, T_r], axis=1)
        b_segs.append(segs)
    return b_segs


def setup_nature_style():
    if "Helvetica" not in font_manager.fontManager.get_font_names():
        helvetica_path = pathlib.Path(__file__).parent.parent / "tmp/helvetica.ttf"
        assert helvetica_path.exists()
        font_manager.fontManager.addfont(helvetica_path)
        helvetica_path = pathlib.Path(__file__).parent.parent / "tmp/helvetica-bold.ttf"
        font_manager.fontManager.addfont(helvetica_path)

    params = {
        "savefig.transparent": True,
        "font.family": ["Helvetica", "sans-serif"],
        "axes.edgecolor": "0.01",
        "axes.labelpad": 5,
        "axes.xmargin": 0,
        "axes.ymargin": 0,
        # Remove top and right spines.
        "axes.spines.top": False,
        "axes.spines.right": False,
        #
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "lines.linewidth": 1.3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
    plt.rcParams.update(params)


def get_constr_labels_mathtext():
    return [
        r"$I_p$",
        r"$\frac{d}{dt} B_v\ \mathrm{(Ts^{-1})}$",
        r"$\frac{d}{dt} W\ \mathrm{(Js^{-1})}$",
        r"$\beta_n$",
        r"$\beta_p$",
        r"$l_i$",
        r"$n_{g, \mathrm{frac}}$",
        r"$\Gamma$",
        r"$\iota_{95}$",
    ]
