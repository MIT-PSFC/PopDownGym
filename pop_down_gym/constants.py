import jax_dataclasses as jdc
import numpy as np

MU0 = 4 * np.pi * 1e-7  # [H/m] or [N/A^2]
EV_TO_J = 1.602176565e-19  # [J/eV]
KEV_TO_J = 1e3 * EV_TO_J  # [J/keV]


@jdc.pytree_dataclass
class ShotConstants:
    R0: float  # Major radius [m]
    Bphi0: float  # On-axis toroidal field [T]

    @classmethod
    def for_sparc(cls):
        return cls(R0=1.85, Bphi0=12.2)

    def romero_norm(self) -> float:
        """Normalization factor to convert the Romero Model from (Li, Ip) to (li, Ip_MA)

        Returns:
            float: normalization factor for the Romero model.
        """
        return 0.5 * 1e6 * MU0 * self.R0
