import math
import jax.numpy as jnp
import equinox as eqx

class TauEInput(eqx.Module):
    IPB98_TABLEAU = {
        "C": 0.0562,
        "I": 0.93,
        "B": 0.15,
        "R": 1.97,
        "eps": 0.58,
        "n": 0.41,
        "kappa": 0.78,
        "M": 0.19,
        "P": -0.69,
    }
    IPB89_TABLEAU = {
        "C": 0.0380,
        "I": 0.85,
        "B": 0.20,
        "R": 1.50,
        "eps": 0.30,
        "n": 0.10,
        "kappa": 0.50,
        "M": 0.50,
        "P": -0.50,
    }
    R0: float  # Major radius [m]
    BPhi0: float  # On-axis toroidal field [T]
    H: float  # H factor [-]
    AFM: float  # Effective fuel atomic mass [AMU]
    ne19: float  # Electron density [x10^19 m^-3]
    kappa_a: float  # Plasma elongation defined as cross-sectional area/(pi*a^2) [-]
    a: float  # Plasma minor radius [m]
    Ip_MA: float  # Plasma current [MA]
    Psol: float  # Power conducted to the scrape-off layer [MW]

    def epsilon(self) -> float:
        return self.a / self.R0  # inverse aspect ratio [-]

    def plasma_volume(self) -> float:
        return plasma_volume(self.R0, self.a, self.kappa_a)

    def scaling_law(self, tableau):
        val = (
            tableau["C"]
            * self.H
            * self.Ip_MA ** tableau["I"]
            * self.BPhi0 ** tableau["B"]
            * self.R0 ** tableau["R"]
            * self.epsilon() ** tableau["eps"]
            * self.ne19 ** tableau["n"]
            * self.kappa_a ** tableau["kappa"]
            * self.AFM ** tableau["M"]
            * self.Psol ** tableau["P"]
        )
        return val

    def scaling_law_alt_psol(self, tableau, psol):
        val = (
            tableau["C"]
            * self.H
            * self.Ip_MA ** tableau["I"]
            * self.BPhi0 ** tableau["B"]
            * self.R0 ** tableau["R"]
            * self.epsilon() ** tableau["eps"]
            * self.ne19 ** tableau["n"]
            * self.kappa_a ** tableau["kappa"]
            * self.AFM ** tableau["M"]
            * psol ** tableau["P"]
        )
        return val

    def ipb98(self) -> float:
        return self.scaling_law(self.IPB98_TABLEAU)

    def ipb89(self) -> float:
        return self.scaling_law(self.IPB89_TABLEAU)

    def ipb98_altpsol(self, psol) -> float:
        return self.scaling_law_alt_psol(self.IPB98_TABLEAU, psol)

    def ipb89_altpsol(self, psol) -> float:
        return self.scaling_law_alt_psol(self.IPB89_TABLEAU, psol)

    @classmethod
    def cmod_default(cls):
        """
        Parameters for shot 1160930033 from:
        Hughes, J. W., et al. "Access to pedestal pressure relevant to burning plasmas on the high
        magnetic field tokamak Alcator C-Mod." Nuclear Fusion 58.11 (2018): 112003.
        """
        volume = 0.94  # m^3
        R0 = 0.68
        a = 0.22
        cross_sectional_area = volume / (2.0 * math.pi * R0)
        kappa_a = cross_sectional_area / (math.pi * a**2.0)
        # Note: paper reports 5.4 MW input power and 1.8 MW radiated power.
        return cls(
            R0=R0,
            BPhi0=5.7,
            H=0.8,  # H98 in this case.
            AFM=2.0,
            ne19=50,
            kappa_a=kappa_a,
            a=0.22,
            Ip_MA=1.4,
            Psol=5.4 - 1.8,  # 5.4 MW input power and 1.8 MW radiated power.
        )

    @classmethod
    def sparc_default(cls):
        return cls(
            R0=1.85,
            BPhi0=12.2,
            H=1.0,  # H98 in this case.
            AFM=2.5,
            ne19=31,
            kappa_a=1.97,
            a=0.57,
            Ip_MA=8.7,
            Psol=1.7  # 1.7 MW Ohmic.
            + 11.1  # 11.1 MW RF.
            + 0.2 * 140  # 140MW fusion, with 1/5 of the fusion power going to alpha particles.
            - 10.4,  # 10.4 MW radiated.
        )

def calc_Bv(Ip: float, kappa: float, beta_p: float, li3: float, R: float, a: float) -> float:
    """Calculate the vertical field required for radial force balance.

    Args:
        Ip (float): plasma current [A].
        kappa (float): plasma elongation [-] (TODO(allenw): which definition?).
        beta_p (float): poloidal beta [-].
        li3 (float): internal inductance [-].
        R (float): major radius [m].
        a (float): minor radius [m].

    Returns:
        float: _description_
    """
    MU0 = 4e-7 * jnp.pi
    Bv = (
        (MU0 * Ip)
        / (4 * jnp.pi * R)
        * (jnp.log(8 * R / (a * kappa**0.5)) + beta_p + 0.5 * li3 - 1.5)
    )
    return Bv


def plasma_volume(R0: float, a: float, kappa_a: float) -> float:
    """By definition, kappa_a = cross_sectional_area/(pi*a^2).
    Multiply the cross sectional area by 2*pi*R0 to get the volume.

    Args:
        R0 (float): _description_
        a (float): _description_
        kappa_a (float): _description_

    Returns:
        float: _description_
    """
    cross_sectional_area = math.pi * a**2.0 * kappa_a
    return 2 * math.pi * R0 * cross_sectional_area

def ohmic_power(R0: float, a: float, Ip_MA: float, kappa: float, Te_kev: float) -> float:
    """Equation obtained from exercise 6 of the 2023 EPFL Control & Operation of Tokamaks
    School."""
    epsilon = a / R0
    c1 = 5.6e4 / (1.0 - 1.31 * epsilon**0.5 + 0.46 * epsilon)
    # Discrepancy. The code says Ip^2, but the paper says Ip^1.
    # Because the Ohmic heating is general I^2R, I think the code is correct.
    c2 = (R0 * Ip_MA**2.0) / (a**2 * kappa * Te_kev**1.5)
    return (c1 * c2)


def brems_power_density(Zeff: float, ne19: float, Te_kev: float) -> float:
    """Equation obtained from exercise 6 of the 2023 EPFL Control & Operation of Tokamaks School.

    Args:
        Zeff (float): Effective atomic mass [AMU]
        ne19 (float): Electron density [x10^19 m^-3]
        Te_kev (float): Average electron temperature [keV]

    Returns:
        float: power radiated per unit volume []
    """
    ne20 = 0.1 * ne19
    # Note: I was hugely confused at first and thought Zeff^2, but it's Zeff^1.
    # this is because Zeff = (sum Zj^2nj)/(sum Zjnj) = sum Zj^2nj/ne (by quasineutrality)
    # so there's some canceling that happens.
    return 5.35e3 * Zeff * ne20**2 * Te_kev**0.5


def sigma_v(Ti_kev: float) -> float:
    """Approximation of reactivity <sigma v> for D-T fusion reactions.
    Uses equation S5 from Hively, L. M. "Convenient computational
    forms for Maxwellian reactivities." Nuclear Fusion 17 (1977): 873-876.
    Recommended by Exercise Session 6 of the 2023 EPFL Control & Operation of Tokamaks School.

    Args:
        Ti_kev (float): _description_

    Returns:
        float: m^3/s
    """
    alpha = 0.2935
    a_minus_one = -21.38
    a0 = -25.2
    a1 = -7.101e-2
    a2 = 1.938e-4
    a3 = 4.925e-6
    a4 = -3.984e-8
    exp_arg = (
        a_minus_one / Ti_kev**alpha
        + a0
        + a1 * Ti_kev
        + a2 * Ti_kev**2
        + a3 * Ti_kev**3
        + a4 * Ti_kev**4
    )
    return 1e-6 * jnp.exp(exp_arg)


def alpha_power_density(ni19: float, Ti_kev: float, f_dt: float) -> float:
    """Approximation of alpha heating power density.

    Args:
        ni (float): particle density [m^-3].
        Ti_kev (float): ion temperature [keV].
        f_dt (float): Deuterium-Tritium fraction.

    Returns:
        float: alpha heating energy density W/m^3
    """
    Ealpha = 5.68e-13  # Energy of alpha particles at birth ~3.5 MeV [J]
    return (f_dt / (1.0 + f_dt) ** 2.0) * Ealpha * 1e19**2.0 * ni19**2.0 * sigma_v(Ti_kev)


def W_to_pressure(W: float, volume: float) -> float:
    """Convert energy density to pressure.

    Args:
        W (float): _description_
        volume (float): _description_

    Returns:
        float: _description_
    """
    W_density = W / volume
    return (2.0 / 3.0) * W_density


def pressure_to_W(pressure: float, volume: float) -> float:
    """Convert pressure to energy density.

    Args:
        pressure (float): _description_
        volume (float): _description_

    Returns:
        float: _description_
    """
    W_density = (3.0 / 2.0) * pressure
    return W_density * volume


def pressure_to_beta(mean_pressure: float, B: float) -> float:
    """Compute (poloidal or toroidal) plasma beta given the volume average pressure
    and the relevant magnetic field.

    From Freidberg eq. (11.58), we have that:
        beta_p = 2*mu_0*<p>/Bta^2
        Bta = (mu0*Ip)/(2*pi*a)

    Args:
        mean_pressure (float): volume average plasma pressure [Pa].
        B (float): relevant magnetic field [T].

    Returns:
        float: poloidal or toroidal beta depending on which magnetic field is provided.
    """
    mu0 = 4e-7 * jnp.pi
    return (2.0 * mu0 * mean_pressure) / B**2.0

def pressure_to_beta_p(mean_pressure: float, Ip_MA: float, a: float) -> float:
    """Compute poloidal plasma beta.

    From Freidberg eq. (11.58), we have that:
        beta_p = 2*mu_0*<p>/Bta^2
        Bta = (mu0*Ip)/(2*pi*a)

    Args:
        mean_pressure (float): Mean plasma pressure [Pa]
        Ip (float): plasma current [A]
        a (float): Minor radius [m]

    Returns:
        float: beta_p.
    """
    mu0 = 4e-7 * jnp.pi
    Bta = (mu0 * 1e6 * Ip_MA) / (2.0 * math.pi * a)
    return pressure_to_beta(mean_pressure, Bta)


def betas_to_beta_n(betap: float, betat: float, Ip_MA: float, a: float, Bphi0: float) -> float:
    """

    Args:
        betap (float): _description_
        betat (float): _description_
        Ip_MA (float): _description_
        a (float): _description_
        Bphi0 (float): _description_

    Returns:
        float: _description_
    """
    beta = 1.0 / (1.0 / betap + 1.0 / betat)
    betan = beta * a * Bphi0 / Ip_MA
    return betan


def volume_integral(q: jnp.ndarray, Vp: jnp.ndarray, wgauss: jnp.ndarray) -> float:
    """Integrate a quantity over the plasma volume.

    Args:
        q (jnp.ndarray): quantity to integrate on a Legendre-Gauss grid.
        Vp (jnp.ndarray): plasma volume on a Legendre-Gauss grid.
        wgauss (jnp.ndarray): Legendre-Gauss quadarture weights.

    Returns:
        float: integrated quantity.
    """
    return jnp.sum(jnp.multiply(wgauss, jnp.multiply(q, Vp)))


def volume_average(q: jnp.ndarray, Vp: jnp.ndarray, wgauss: jnp.ndarray) -> float:
    return volume_integral(q, Vp, wgauss) / jnp.dot(wgauss, Vp)


def PLH_threshold(ne20: float, B0: float, a: float, R: float) -> float:
    """H-mode Pthreshold scaling from Y.Martin JP Conf.series 123 (2008)

    Args:
        ne20 (float): _description_
        B0 (float): _description_
        a (float): _description_
        R (float): _description_

    Returns:
        float: H-mode threshold in watts.
    """
    return 2.15e6 * math.exp(1) ** 0.107 * ne20**0.782 * B0**0.772 * a**0.975 * R**0.999