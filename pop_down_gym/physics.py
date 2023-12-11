import math

import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Dict


class TauEScaling(eqx.Module):
    class InputData(eqx.Module):
        R0: float
        BPhi0: float
        H: float
        AFM: float
        ne19: float
        kappa_a: float
        a: float
        Ip_MA: float
        Psol: float
        Hmode: bool

    HMODE_TABLEAU: Dict[str, float]
    LMODE_TABLEAU: Dict[str, float]

    def __init__(self):
        # Initialize the H-mode tableau to the IPB98(y,2) values.
        self.HMODE_TABLEAU = {
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
        # Initialize the L-mode tableau to the IPB89 values.
        self.LMODE_TABLEAU = {
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

    def __call__(self, input_data: InputData) -> float:
        tableau = jax.lax.cond(
            input_data.Hmode, lambda: self.HMODE_TABLEAU, lambda: self.LMODE_TABLEAU
        )
        epsilon = input_data.a / input_data.R0
        taue = (
            tableau["C"]
            * input_data.H
            * input_data.Ip_MA ** tableau["I"]
            * input_data.BPhi0 ** tableau["B"]
            * input_data.R0 ** tableau["R"]
            * epsilon ** tableau["eps"]
            * input_data.ne19 ** tableau["n"]
            * input_data.kappa_a ** tableau["kappa"]
            * input_data.AFM ** tableau["M"]
            * input_data.Psol ** tableau["P"]
        )
        return taue

    def trainable_params_filter_spec(self):
        return eqx.is_inexact_array_like


def shafranov_coeff(
    R0: float, aminor: float, kappa: float, beta_p: float, li3: float
) -> float:
    """The Shafranov coefficient, which is relevant for calculating the required vertical field
    as well as calculations of the open-loop vertical instability growth rate.

    Args:
        R0 (float): major radius [m].
        aminor (float): minor radius [m].
        kappa (float): elongation [-].
        beta_p (float): poloidal beta [-].
        li3 (float): internal inductance [-].

    Returns:
        float: Shafranov coefficient [-].
    """
    shafranov_coeff = (
        jnp.log(8 * R0 / (aminor * kappa**0.5)) + beta_p + 0.5 * li3 - 1.5
    )
    return shafranov_coeff


def calc_Bv(Ip_MA: float, R0: float, shafranov_coeff: float) -> float:
    """Calculate the vertical field required for radial force balance.

    Args:
        Ip_MA (float): plasma current [MA].
        R0 (float): major radius [m].
        shafranov_coeff (float): Shafranov coefficient [-].

    Returns:
        float: vertical field [T].
    """
    Ip = 1e6 * Ip_MA
    MU0 = 4e-7 * jnp.pi
    Bv = (MU0 * Ip) / (4 * jnp.pi * R0) * shafranov_coeff
    return Bv


def plasma_volume(R0: float, a: float, kappa_a: float) -> float:
    """By definition, kappa_a = cross_sectional_area/(pi*a^2).
    Multiply the cross sectional area by 2*pi*R0 to get the volume.

    Args:
        R0 (float): major radius [m].
        a (float): minor radius [m].
        kappa_a (float): areal elongation [-].

    Returns:
        float: plasma volume [m^3].
    """
    cross_sectional_area = math.pi * a**2.0 * kappa_a
    return 2 * math.pi * R0 * cross_sectional_area


def ohmic_power(
    R0: float, a: float, Ip_MA: float, kappa: float, Te_kev: float
) -> float:
    """Equation obtained from exercise 6 of the 2023 EPFL Control & Operation of Tokamaks
    School. In general, ohmic heating requires the electron temperature and current profiles
    to accurately calculate, but the SPARC PRD flattop is expected to have ohmic heating
    be only a few percent of the total heating power, so this approximate expression
    is likely sufficient.

    Args:
        R0 (float): major radius [m].
        a (float): minor radius [m].
        Ip_MA (float): plasma current [MA].
        kappa (float): plasma elongation [-].
        Te_kev (float): volume average electron temperature [keV].

    Returns:
        float: ohmic heating power [W].
    """
    epsilon = a / R0
    c1 = 5.6e4 / (1.0 - 1.31 * epsilon**0.5 + 0.46 * epsilon)
    # Discrepancy. The code says Ip^2, but the paper says Ip^1.
    # Because the Ohmic heating is general I^2R, I think the code is correct.
    c2 = (R0 * Ip_MA**2.0) / (a**2 * kappa * Te_kev**1.5)
    return c1 * c2


def brems_power_density(Zeff: float, ne19: float, Te_kev: float) -> float:
    """Equation obtained from exercise 6 of the 2023 EPFL Control & Operation of Tokamaks School.

    Args:
        Zeff (float): Effective atomic mass [AMU]
        ne19 (float): Electron density [x10^19 m^-3]
        Te_kev (float): Average electron temperature [keV]

    Returns:
        float: power radiated per unit volume.
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
        Ti_kev (float): local ion temperature [keV].

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
        float: alpha heating energy density [W/m^3]
    """
    Ealpha = 5.68e-13  # Energy of alpha particles at birth ~3.5 MeV [J]
    return (
        (f_dt / (1.0 + f_dt) ** 2.0)
        * Ealpha
        * 1e19**2.0
        * ni19**2.0
        * sigma_v(Ti_kev)
    )


def W_to_pressure(W: float, volume: float) -> float:
    """Convert energy density to pressure.

    Args:
        W (float): _description_
        volume (float): _description_

    Returns:
        float: _description_
    """
    W_avg = W / volume
    return (2.0 / 3.0) * W_avg


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


def betas_to_beta_n(
    betap: float, betat: float, Ip_MA: float, a: float, B0: float
) -> float:
    """Compute the normalized beta from the poloidal and toroidal beta.

    Args:
        betap (float): poloidal beta [-].
        betat (float): toroidal beta [-].
        Ip_MA (float): plasma current [MA].
        a (float): minor radius [m].
        B0 (float): on axis toroidal magnetic field [T].

    Returns:
        float: _description_
    """
    beta = 1.0 / (1.0 / betap + 1.0 / betat)
    betan = beta * a * B0 / Ip_MA
    return betan

def volume_integral(
    quantity: jnp.ndarray, Vp: jnp.ndarray, wgauss: jnp.ndarray
) -> float:
    """Integrate a quantity over the plasma volume.

    Args:
        q (jnp.ndarray): quantity to integrate on a Legendre-Gauss grid.
        Vp (jnp.ndarray): plasma volume on a Legendre-Gauss grid.
        wgauss (jnp.ndarray): Legendre-Gauss quadarture weights.

    Returns:
        float: integrated quantity.
    """
    product = jnp.multiply(wgauss, jnp.multiply(quantity, Vp))
    out = jnp.sum(product)
    return out


def volume_average(q: jnp.ndarray, Vp: jnp.ndarray, wgauss: jnp.ndarray) -> float:
    """Compute the volume average of a quantity on a Legendre-Gauss grid.

    Args:
        q (jnp.ndarray): quantity to integrate on a Legendre-Gauss grid.
        Vp (jnp.ndarray): dVolume/drho on a Legendre-Gauss grid.
        wgauss (jnp.ndarray): Legendre-Gauss quadarture weights.

    Returns:
        float: volume average of the quantity.
    """
    return volume_integral(q, Vp, wgauss) / jnp.dot(wgauss, Vp)


def PLH_threshold(ne20: float, B0: float, a: float, R: float) -> float:
    """H-mode Pthreshold scaling from Y.Martin JP Conf.series 123 (2008)

    Args:
        ne20 (float): line average electron density [10^20/m^-3].
        B0 (float): on axis toroidal magnetic field [T].
        a (float): minor radius [m].
        R (float): major radius [m].

    Returns:
        float: H-mode threshold [W].
    """
    return (
        2.15e6
        * math.exp(1) ** 0.107
        * ne20**0.782
        * B0**0.772
        * a**0.975
        * R**0.999
    )


def greenwald_fraction(ne19_line_average: float, Ip_MA: float, a: float) -> float:
    """Compute the greenwald fraction.

    Args:
        ne19_line_average (float): line average electron density [10^19/m^-3].
        Ip_MA (float): plasma current [MA].
        a (float): minor radius [m].

    Returns:
        float: greenwald fraction.
    """

    ne20_line_average = ne19_line_average / 10.0
    ng_max = Ip_MA / (jnp.pi * a**2)
    return ne20_line_average / ng_max


def q95(
    Ip_MA: float,
    B0: float,
    R0: float,
    aminor: float,
    kappa: float,
    delta: float,
    w07: float,
) -> float:
    """Approximation of the safety factor at the 95% flux surface from:

    Sauter, O. "Geometric formulas for system codes including the effect
    of negative triangularity." Fusion Engineering and Design 112 (2016).

    Args:
        Ip_MA (float): plasma current [MA].
        B0 (float): on axis toroidal magnetic field [T].
        R0 (float): major radius [m].
        aminor (float): minor radius [m].
        kappa (float): elongation [-].
        delta (float): triangularity [-].
        w07 (float): squareness factor introduced by Sauter [-].

    Returns:
        float: approximation of q95 [-].
    """
    epsilon = aminor / R0
    c0 = (4.1 * aminor**2.0 * B0) / (R0 * Ip_MA)
    c1 = 1 + 1.2 * (kappa - 1.0) + 0.56 * (kappa - 1.0) ** 2.0
    c2 = 1 + 0.09 * delta + 0.16 * delta**2.0
    c3 = (1 + 0.45 * delta * epsilon) / (1 - 0.74 * epsilon)
    c4 = 1 + 0.55 * (w07 - 1)
    return c0 * c1 * c2 * c3 * c4
