from pop_down_gym.raptor.init_raptor import init_matlab, init_sparc_rd
from pop_down_gym.raptor.utils import (
    concat_simres,
    VWrapper,
    numpy_to_matlab,
    update_ustep,
    to_numpy,
)
import numpy as np

class ParticleModel:
    def __init__(
        self, ni19_vol, ni_basis, ne_basis, wgauss, tau_n_factor, main_ion_dilution
    ):
        self.ni19_vol = ni19_vol
        self.ni_basis = ni_basis
        self.ne_basis = ne_basis
        self.wgauss = wgauss
        self.tau_n_factor = tau_n_factor
        self.main_ion_dilution = main_ion_dilution

    def step(self, fueling19, volume, volume_dot, taue, dt):
        self.ni19_vol += dt * self.deriv(fueling19, volume, volume_dot, taue)
        return

    def deriv(self, fueling19, volume, volume_dot, taue):
        Ni19 = volume * self.ni19_vol
        Ni19_dot = -Ni19 / (self.tau_n_factor * taue) + fueling19
        ni19_vol_dot = (Ni19_dot * volume - Ni19 * volume_dot) / volume**2
        return ni19_vol_dot

    def ni19_line_average(self, Vp):
        return ParticleModel.volume_to_line_average(
            self.ni19_vol, self.wgauss, Vp, self.ni_basis
        )

    def ne19_line_average(self, Vp):
        ne19_vol = self.ni19_vol / self.main_ion_dilution
        return ParticleModel.volume_to_line_average(
            ne19_vol, self.wgauss, Vp, self.ne_basis
        )

    @staticmethod
    def volume_to_line_average(volume_average, wgauss, Vp, Q_basis):
        """
        The math is as follows.
        For a general profile we have that:
            Vol * <Q> = \int_0^1 Q(rho) dV = \int_0^1 Q(rho) dV/drho(rho) drho
        If Q is defined on a Legendre-Gauss grid, then we can write:
            Vol * <Q> ~ \sum_i w_i dV/drho(rho_i) Q(rho_i)
        where w_i are the Gauss weights and rho_i are the Gauss points.
        When Q is a profile with a fixed shape that is express by a constant times a basis function,
        we have that:
            Q(rho) = c * Q_basis(rho)
        And thus, we have that:
            Vol * <Q> ~ c * \sum_i w_i dV/drho(rho_i) Q_basis(rho_i)
        So given the Volume average, we can compute the  constant as such:
            c ~ Vol * <Q> / \sum_i w_i dV/drho(rho_i) Q_basis(rho_i)
        The line average is then:
            \sum_i w_i * c * Q_basis(rho_i)
            c \sum_i w_i * Q_basis(rho_i)
        """
        volume = np.dot(wgauss, Vp)
        c = volume * volume_average / np.dot(np.multiply(wgauss, Vp), Q_basis)
        return c * np.dot(wgauss, Q_basis)


class GeomVariable:
    def __init__(self):
        self.gs = 0

    def step(self, step_size):
        self.gs += step_size
        self.gs = np.clip(self.gs, 0.0, 1.0)
        return self.gs


class RaptorRDGym:
    def __init__(self, raptor_repo_root: str, raptor_dt: float, rspgs: int):
        """

        Args:
            raptor_repo_root (str): path to the raptor repository root.
            raptor_dt (float): raptor simulation step size (s).
            rspgs (int): number of raptor steps per gym step.
        """
        self.raptor_repo_root = raptor_repo_root
        self.eng_handle = init_matlab(raptor_repo_root)
        self.raptor_dt = raptor_dt
        self.rspgs = rspgs  # Number of raptor steps per gym step.
        self.step_times = raptor_dt * np.arange(rspgs)

    @property
    def raptor_iter(self):
        return self.gym_iter * self.rspgs

    def reset(self):
        (
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
            ne_line_avg,
            ni_line_avg,
            ne_vol_avg,
            ni_vol_avg,
        ) = init_sparc_rd(self.raptor_repo_root, self.eng_handle, self.raptor_dt)
        self.raptor_x = x0
        self.g_interp = g_interp
        self.Vp_interp = Vp_interp
        self.vwrapper = VWrapper(v, model)
        self.Ustep = U0[:, : self.rspgs]
        self.model = model
        self.wgauss = to_numpy(model["rgrid"]["wgauss"]).squeeze()

        self.params = params
        self.simres = None
        self.outs = [out0]
        self.tgrid = params["tgrid"].clone()
        self.config = config
        self.gym_iter = 0
        self.ne_basis = ne_basis
        self.ni_basis = ni_basis

        self.geom_var = GeomVariable()
        self.particle_model = ParticleModel(
            ni19_vol=1e-19 * ni_vol_avg,
            ni_basis=ni_basis,
            ne_basis=ne_basis,
            wgauss=self.wgauss,
            tau_n_factor=7.5,
            main_ion_dilution=ni_vol_avg / ne_vol_avg,
        )
        self.Paux_max = 25e6 # TODO(allenw): hard-coded.
        self.currently_h_mode = True

        # Do a quick test that the line average calculations work..
        assert (
            abs(
                1e19 * self.particle_model.ne19_line_average(self.Vp_interp(0.0))
                - ne_line_avg
            )
            / ne_line_avg
            < 1e-6
        )
        assert (
            abs(
                1e19 * self.particle_model.ni19_line_average(self.Vp_interp(0.0))
                - ni_line_avg
            )
            / ni_line_avg
            < 1e-6
        )

    def step_particle_model(self, fueling19):
        raptor_out = self.outs[-1]
        time = to_numpy(raptor_out["time"]).squeeze()
        Volume = to_numpy(raptor_out["Volume"][-1]).squeeze()
        Volume_dot = np.diff(Volume) / np.diff(time)
        self.particle_model.step(
            fueling19=fueling19,
            volume=Volume[-1],
            volume_dot=Volume_dot[-1],
            taue=raptor_out["tauE"][0][-1],
            dt=self.raptor_dt * self.rspgs,
        )
        return

    def state_for_pd_gym(self):
        raptor_out = self.outs[-1]
        Li = to_numpy(raptor_out["Li"]).squeeze()
        Ip = to_numpy(raptor_out["Ip"][-1]).squeeze()
        time = to_numpy(raptor_out["time"]).squeeze()
        Li_dot = np.diff(Li) / np.diff(time)
        Ip_dot = np.diff(Ip) / np.diff(time)

        vc_minus_vb = -Li_dot * Ip[1:] - Ip_dot * Li[1:]

        state = {
            "li": raptor_out["li3"][0][-1], # li = li3 under the assumption that the plasma is perfectly toroidal.
            "Ip_MA": 1e-6 * Ip[-1],
            # Take the mean over the last "rspgs" steps to avoid
            # numerical issues.
            "vc_minus_vb": np.mean(vc_minus_vb[-self.rspgs :]),
            "Wth": raptor_out["Wth"][0][-1],
            "nfuel19_vol": self.particle_model.ni19_vol[0],
            "Paux": 1e-6 * raptor_out["Pauxtot"][0][-1],
            "gs": self.geom_var.gs,
            "Hmode": int(self.currently_h_mode),
        }
        return state

    def get_vstep(self, ne_line, ni_line):
        """
        Handle H-mode.
        """
        # To check whether we are in H mode, it is more robust to check the previous time step.
        # This works in all cases except the first iteration where raptor_iter = 0. The max
        # in this case ensures that we don't go negative.
        hmode_iter_check = max(0, self.raptor_iter - 1)
        self.currently_h_mode = int(self.vwrapper.hmode[hmode_iter_check])
        out_prev = self.outs[-1]

        # We don't know what the H-L threshold is, for this sim lets just create a fudge factor.
        PLH = out_prev["PLH"][0][-1]
        fudge_factor = 0.6
        PHL = fudge_factor * PLH

        # I don't really like this, but it should work.
        # Once the HL transition occurs set the rest of the sim to LMode.
        # TODO(allenw): raptor defines loss as conduction plus rad.
        # In my gym, I define loss as just conduction.
        # We should make them consistent...
        # For now, lets see what happens if we do this...
        Ploss = np.mean(out_prev['Ploss'][0][-self.rspgs:]) - np.mean(out_prev['Prad'][0][-self.rspgs:])
        print(f"Ploss: {Ploss}, PHL: {PHL}")
        if Ploss > PHL and self.currently_h_mode:
            pass
        else:
            self.vwrapper.hmode[self.raptor_iter :] = 0
            self.vwrapper.te_bc[self.raptor_iter :] = self.config["hmode"]["params"][
                "te_rhoedge"
            ]
            self.vwrapper.ti_bc[self.raptor_iter :] = self.config["hmode"]["params"][
                "ti_rhoedge"
            ]

        vwrapper_step = self.vwrapper[self.raptor_iter : self.raptor_iter + self.rspgs]

        """
        Set particle densities.
        """
        ne_profile = ne_line * self.ne_basis
        vwrapper_step.ne = self.eng_handle.mldivide(
            self.model["ne"]["Lamgauss"], numpy_to_matlab(ne_profile)
        )

        ni_profile = ni_line * self.ni_basis
        vwrapper_step.ni = self.eng_handle.mldivide(
            self.model["ni"]["Lamgauss"], numpy_to_matlab(ni_profile)
        )
        return vwrapper_step.v

    def step(self, action):
        """
        Units:
            dIp_dt (A/s)
            dPaux_dt (W/s)
            dgs_dt [-]
            fueling19 [10^19/s]
        """

        dIp_dt, dPaux_dt, dgs_dt, fueling19 = (
            action["dIp_dt"],
            action["dPaux_dt"],
            action["dgs_dt"],
            action["fueling19"],
        )
        Vp = self.Vp_interp(self.geom_var.gs)
        ne_line_avg = 1e19 * self.particle_model.ne19_line_average(Vp)
        ni_line_avg = 1e19 * self.particle_model.ni19_line_average(Vp)

        self.Ustep = update_ustep(self.Ustep, dIp_dt, dPaux_dt, self.raptor_dt, Paux_max=self.Paux_max)

        # Get the geometry for this step.
        # Assume constant gs for each step.
        gstep = numpy_to_matlab(
            np.column_stack(
                [self.g_interp(self.geom_var.gs) for _ in range(self.rspgs)]
            )
        )

        # Update the v struct by using the latest ne and ni profiles.
        vstep = numpy_to_matlab(self.get_vstep(ne_line_avg, ni_line_avg))

        # A bit annoying, but we need to update the tgrid in the params struct for every time step.
        self.params["tgrid"] = self.tgrid[0][
            self.raptor_iter : self.raptor_iter + self.rspgs
        ]

        self.raptor_x, simres, out = self.eng_handle.step_raptor(
            self.raptor_x,
            gstep,
            vstep,
            numpy_to_matlab(self.Ustep),
            self.model,
            self.params,
            self.rspgs,
            nargout=3,
        )
        if self.simres:
            self.simres = concat_simres(self.simres, simres)
        else:
            self.simres = simres
        self.outs.append(out)
        self.gym_iter += 1

        # Update geom var.
        self.geom_var.step(self.raptor_dt * self.rspgs * dgs_dt)

        # Update particle model.
        self.step_particle_model(fueling19)
        return

    def raptor_out(self):
        self.params["tgrid"] = self.tgrid[0][: self.raptor_iter]
        raptor_out = self.eng_handle.RAPTOR_out(
            self.simres, self.model, self.params, nargout=1
        )
        return raptor_out
    
    def save_out(self, path):
        raptor_out = self.raptor_out()
        self.eng_handle.workspace['out'] = raptor_out
        self.eng_handle.save(path, 'out', nargout=0)


if __name__ == "__main__":
    gym = RaptorRDGym("/home/awang/raptor", 1e-2, 5)  # dts are 0.05s.
    gym.reset()
