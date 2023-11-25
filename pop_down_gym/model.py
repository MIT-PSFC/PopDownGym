import jax
import jax.numpy as jnp
import equinox as eqx

import pop_down_gym
import pop_down_gym.physics as physics
from pop_down_gym import constants
from pop_down_gym.geometry import Geometry
from pop_down_gym.profiles import ProfileBases
from contrax.examples.plasma.li_ip.models import RomeroNNV
from contrax.simulate import SimFFControl

from pop_down_gym.load_data import load_data

class Model:
    geom: Geometry
    hmode_prof_basis: ProfileBases
    lmode_prof_basis: ProfileBases
    li_model: RomeroNNV

    def __init__(
        self,
        geom: Geometry,
        li_model: RomeroNNV,
        hmode_prof_bases: ProfileBases,
        lmode_prof_bases: ProfileBases,
        shot_constants: constants.ShotConstants,
    ):
        self.li_model = li_model
        self.geom = geom
        self.hmode_prof_basis = hmode_prof_bases
        self.lmode_prof_basis = lmode_prof_bases
        self.shot_constants = shot_constants

    def __call__(self, state, control, params, debug=False):
        """
        Maps state and control to derivatives of state.

        State: [li, Ip_MA, vc_minus_vb, Wth, nfuel19_vol, Paux, gs]
        Control: [dIp_dt, dPaux_dt, fueling19, dgs_dt]
        Params: [Hmode, Hfactor, Zeff, ion_dilution, Te_over_Ti, f_dt, tau_n_factor, prad_mult]

        """
        # Compute geometry parameters.
        geometry_params = self.geom(state["gs"])
        kappa, kappa_a, aminor, Vp, volume = (
            geometry_params["kappa"],
            geometry_params["kappa_a"],
            geometry_params["aminor"],
            geometry_params["Vp"],
            geometry_params["volume"],
        )

        # jacfwd gives derivatives of geometry parameters w.r.t. gs.
        # Multiply by dgs_dt to get derivatives of geometry parameters w.r.t. time.
        dgeom_dgs = jax.jacfwd(self.geom)(state["gs"])
        geometry_params_dot = jax.tree_map(
            lambda leaf: leaf * control["dgs_dt"], dgeom_dgs
        )
        volume_dot = geometry_params_dot["volume"]

        # Compute density profiles.
        wgauss = jax.lax.select(
            params["Hmode"], self.hmode_prof_basis.wgauss, self.lmode_prof_basis.wgauss
        )

        nfuel19_profile = jax.lax.select(
            params["Hmode"],
            self.hmode_prof_basis.ni_basis.volume_average_to_profile(
                state["nfuel19_vol"], Vp
            ),
            self.lmode_prof_basis.ni_basis.volume_average_to_profile(
                state["nfuel19_vol"], Vp
            ),
        )

        ne19_vol = state["nfuel19_vol"] / params["ion_dilution"]
        ne19_profile = jax.lax.select(
            params["Hmode"],
            self.hmode_prof_basis.ne_basis.volume_average_to_profile(ne19_vol, Vp),
            self.lmode_prof_basis.ne_basis.volume_average_to_profile(ne19_vol, Vp),
        )
        ne19_line = jax.lax.select(
            params["Hmode"],
            self.hmode_prof_basis.ne_basis.line_average(ne19_profile),
            self.lmode_prof_basis.ne_basis.line_average(ne19_profile),
        )

        # Compute the volume average pressure.
        pressure_vol_avg = physics.W_to_pressure(state["Wth"], volume)

        # ASSUMPTION:
        #   <p> = <n_e> * <T_e> + <n_i> * <T_i>
        # When Te = Te_over_Ti * Ti, this becomes:
        #   <p> = (<n_e> * Te_over_Ti + <n_i> ) * <T_i>
        # Which then implies that:
        #   <T_i> = <p>/(<n_e> * Te_over_Ti + <n_i> )
        Ti_joule_vol = pressure_vol_avg / (
            1e19 * ne19_vol * params["Te_over_Ti"] + 1e19 * state["nfuel19_vol"]
        )
        Te_joule_vol = params["Te_over_Ti"] * Ti_joule_vol
        Ti_kev_vol = Ti_joule_vol / constants.KEV_TO_J
        Te_kev_vol = Te_joule_vol / constants.KEV_TO_J

        Ti_kev_prof = jax.lax.select(
            params["Hmode"],
            self.hmode_prof_basis.Ti_basis.volume_average_to_profile(Ti_kev_vol, Vp),
            self.lmode_prof_basis.Ti_basis.volume_average_to_profile(Ti_kev_vol, Vp),
        )

        Te_kev_prof = jax.lax.select(
            params["Hmode"],
            self.hmode_prof_basis.Te_basis.volume_average_to_profile(Te_kev_vol, Vp),
            self.lmode_prof_basis.Te_basis.volume_average_to_profile(Te_kev_vol, Vp),
        )

        Salpha = physics.alpha_power_density(
            nfuel19_profile, Ti_kev_prof, params["f_dt"]
        )
        Palpha = physics.volume_integral(Salpha, Vp, wgauss)
        Srad = physics.brems_power_density(params["Zeff"], ne19_profile, Te_kev_prof)
        Prad = physics.volume_integral(Srad, Vp, wgauss)
        Pohm = physics.ohmic_power(
            self.shot_constants.R0, aminor, state["Ip_MA"], kappa, Te_kev_vol
        )
        Ptot = Palpha + Pohm + 1e6 * state["Paux"] - params["prad_mult"] * Prad

        tei = physics.TauEInput(
            R0=self.shot_constants.R0,
            BPhi0=self.shot_constants.Bphi0,
            H=params["Hfactor"],
            AFM=2.5,
            ne19=ne19_line,
            kappa_a=kappa_a,
            a=aminor,
            Ip_MA=state["Ip_MA"],
            Psol=1e-6 * (Ptot),
        )

        taue = jnp.where(params["Hmode"], tei.ipb98(), tei.ipb89())

        Wdot = -state["Wth"] / taue + Ptot

        """
        Compute li model derivatives.
        """
        # Convert ramprate control to Vind using Romero.
        vind = (
            control["dIp_dt"] * self.li_model.romero_norm * state["li"]
            - state["vc_minus_vb"]
        ) / 2.0
        li_state = {
            "li": state["li"],
            "ip_MA": state["Ip_MA"],
            "vc_minus_vb": state["vc_minus_vb"],
        }
        li_control = {
            "Vind": vind,
            "te_vol_avg_kev": Te_kev_vol,
        }
        li_derivs = self.li_model(li_state, li_control)

        # nfuel = Nfuel/Volume
        # By the quotient rule:
        # nfuel_dot = (Nfuel_dot * Volume - Nfuel * Volume_dot) / Volume^2
        Nfuel19 = volume * state["nfuel19_vol"]
        Nfuel19_dot = -Nfuel19 / (params["tau_n_factor"] * taue) + control["fueling19"]
        nfuel19_vol_dot = (Nfuel19_dot * volume - Nfuel19 * volume_dot) / volume**2

        derivatives = {
            "li": li_derivs["li"],
            "Ip_MA": li_derivs["ip_MA"],
            "vc_minus_vb": li_derivs["vc_minus_vb"],
            "Wth": Wdot,
            "nfuel19_vol": nfuel19_vol_dot,
            "Paux": control["dPaux_dt"],
            "gs": control["dgs_dt"],
        }

        if debug:
            info = {
                "Ploss": -state["Wth"] / taue,
                "taue": taue,
                "ne19_line": ne19_line,
                "kappa": kappa,
                "kappa_a": kappa_a,
                "aminor": aminor,
                "pressure_vol_avg": pressure_vol_avg,
                "Wdot": Wdot,
            }
            return derivatives, info
        else:
            return derivatives

    @classmethod
    def create_default(cls):
        ds, ds_geom = load_data()

        consts = constants.ShotConstants.for_sparc()
        g = Geometry(
            consts.R0,
            ds_geom.time.values,
            ds_geom.aminor.values.squeeze(),
            ds_geom.kappa.values.squeeze(),
            ds_geom.kappa_a.values.squeeze(),
            ds_geom.Vp.values.squeeze(),
        )

        #
        hmode = ds["te"].sel(rho=0.95, method="nearest") > 3000
        ds["Hmode"] = hmode.drop_vars("rho")  #
        hmode_data = ds.where(ds["Hmode"], drop=True)
        lmode_data = ds.where(~ds["Hmode"], drop=True)
        hmode_basis, _, _ = ProfileBases.from_dataset(hmode_data)
        lmode_basis, _, _ = ProfileBases.from_dataset(lmode_data)


        romero_nnv = RomeroNNV(
                        consts,
                        eqx.nn.MLP(
                            in_size=3 + 2,
                            out_size=1,
                            width_size=32,
                            depth=2,
                            key=jax.random.PRNGKey(0),
                            activation=jax.nn.softplus,
                        ),
                    )
        li_ip_sim = SimFFControl(romero_nnv, dt0=0.01)
        li_ip_sim = eqx.tree_deserialise_leaves(
            f"{pop_down_gym.ROOT_DIR}/../contrax/contrax/examples/plasma/models/romero_nnv.eqx",
            li_ip_sim,
        )
        return cls(g, li_ip_sim.model, hmode_basis, lmode_basis, consts), ds