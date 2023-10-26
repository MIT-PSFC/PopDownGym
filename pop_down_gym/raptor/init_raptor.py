import os

import matlab.engine
from scipy.interpolate import interp1d

from .utils import numpy_to_matlab, to_numpy


def init_matlab(raptor_repo_root: str) -> matlab.engine.MatlabEngine:
    """Initialize an MatlabEngine instance and add project-relevant paths.

    Args:
        raptor_repo_root (str): directory of the raptor repository.

    Returns:
        matlab.engine.MatlabEngine: matlab engine handle.
    """
    eng = matlab.engine.start_matlab()
    raptor_path_file = os.path.join(raptor_repo_root, "RAPTOR_path.m")
    eng.run(raptor_path_file, nargout=0)
    genpath = eng.genpath(os.path.join(raptor_repo_root, "projects", "SPARC"))
    eng.addpath(genpath, nargout=0)
    return eng


def init_sparc_rd(raptor_repo_root: str, eng: matlab.engine.MatlabEngine, dt: float):
    transp_data = eng.load(
        os.path.join(
            raptor_repo_root, "projects", "SPARC", "SPARC_V1E_transp_3mod2.mat"
        )
    )

    x0, g, v, U0, model, params, simres0, out0, config = eng.init_raptor(
        transp_data, dt, nargout=9
    )

    g = to_numpy(g)
    v = to_numpy(v)
    U0 = to_numpy(U0)

    # Get g as a function of Ip.
    Ip = U0[0, :]
    g_interp = interp1d(Ip, g)

    # Get Vp as a function of Ip.
    Vp = eng.eval_Vp([], numpy_to_matlab(g), [], model, True)
    Vp_interp = interp1d(Ip, to_numpy(Vp))

    # Build a ne basis that will be used for updating the ne profile.
    # Note: index with [0] instead of 0 to preserve dimension information.
    ne_profile0 = eng.eval_ne([], [], numpy_to_matlab(v[:, [0]]), model, True)
    ne_line0 = eng.int_Ltot(ne_profile0, model)
    ne_basis = (
        to_numpy(ne_profile0) / ne_line0
    )  # Multiply this basis by line average ne to get ne profile.

    # Build a ni basis that will be used for updating the ni profile.
    # Note: index with [0] instead of 0 to preserve dimension information.
    ni_profile0 = eng.eval_ni([], [], numpy_to_matlab(v[:, [0]]), model, True)
    ni_line0 = eng.int_Ltot(ni_profile0, model)
    ni_basis = (
        to_numpy(ni_profile0) / ni_line0
    )  # Multiply this basis by line average ni to get ni profile.

    return (
        x0,
        g_interp,
        Vp_interp,
        v,
        U0,
        model,
        params,
        simres0,
        out0,
        config,
        ne_basis,
        ni_basis,
    )
