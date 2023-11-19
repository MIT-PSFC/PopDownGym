import equinox as eqx
import jax.numpy as jnp
from contrax.controls.controls import cubic_interp
from pop_down_gym.physics import plasma_volume
from diffrax import CubicInterpolation


class Geometry(eqx.Module):
    """Provide an interpolation of equilibria data."""

    R0: float
    a: CubicInterpolation
    kappa: CubicInterpolation
    kappa_a: CubicInterpolation
    Vp: CubicInterpolation

    def __init__(
        self,
        R0: float,
        ts: jnp.ndarray,
        a: jnp.ndarray,
        kappa: jnp.ndarray,
        kappa_a: jnp.ndarray,
        Vp: jnp.ndarray,
    ) -> None:
        s = ts / (ts[-1] - ts[0])
        self.R0 = R0
        self.a = cubic_interp(s, a)
        self.kappa = cubic_interp(s, kappa)
        self.kappa_a = cubic_interp(s, kappa_a)
        self.Vp = cubic_interp(s, [v for v in Vp])

    def __call__(self, s: float):
        aminor = self.a.evaluate(s)
        kappa = self.kappa.evaluate(s)
        kappa_a = self.kappa_a.evaluate(s)
        Vp = jnp.array(self.Vp.evaluate(s))
        out = {
            "aminor": aminor,
            "kappa": kappa,
            "kappa_a": kappa_a,
            "Vp": Vp,
            "volume": plasma_volume(self.R0, aminor, kappa_a),
        }
        return out
