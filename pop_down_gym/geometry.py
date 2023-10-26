import equinox as eqx
import jax.numpy as jnp
from contrax.controls.controls import ControlTraj
from pop_down_gym.physics import plasma_volume


class Geometry(eqx.Module):
    """Provide an interpolation of equilibria data."""

    R0: float
    a: ControlTraj
    kappa_a: ControlTraj
    Vp: ControlTraj

    def __init__(
        self,
        R0: float,
        ts: jnp.ndarray,
        a: jnp.ndarray,
        kappa_a: jnp.ndarray,
        Vp: jnp.ndarray,
    ) -> None:
        s = ts / (ts[-1] - ts[0])
        self.R0 = R0
        self.a = ControlTraj.spline_interp(s, a, spline_order=1)
        self.kappa_a = ControlTraj.spline_interp(s, kappa_a, spline_order=1)
        Vp_traj = ControlTraj.spline_interp(s, [v for v in Vp], spline_order=1)
        self.Vp = lambda s: jnp.array(Vp_traj(s))

    def __call__(self, s: float):
        aminor = self.a(s)
        kappa_a = self.kappa_a(s)
        Vp = self.Vp(s)
        out = {
            "aminor": aminor,
            "kappa_a": kappa_a,
            "Vp": Vp,
            "volume": plasma_volume(self.R0, aminor, kappa_a),
        }
        return out
