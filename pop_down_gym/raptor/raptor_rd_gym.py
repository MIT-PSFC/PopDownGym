from rd_rl.raptor.init_raptor import init_matlab, init_sparc_rd
from rd_rl.raptor.utils import concat_simres, VWrapper, numpy_to_matlab, update_ustep
import matlab.engine
import numpy as np
import pandas as pd


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
        ) = init_sparc_rd(self.raptor_repo_root, self.eng_handle, self.raptor_dt)
        self.raptor_x = x0
        self.g_interp = g_interp
        self.Vp_interp = Vp_interp
        self.vwrapper = VWrapper(v, model)
        self.Ustep = U0[:, : self.rspgs]
        self.model = model
        self.params = params
        self.simres = None
        self.outs = [out0]
        self.tgrid = params["tgrid"].clone()
        self.config = config
        self.gym_iter = 0
        self.ne_basis = ne_basis
        self.ni_basis = ni_basis

    def get_vstep(self, ne_line, ni_line):
        """
        Handle H-mode.
        """
        # To check whether we are in H mode, it is more robust to check the previous time step.
        # This works in all cases except the first iteration where raptor_iter = 0. The max
        # in this case ensures that we don't go negative.
        hmode_iter_check = max(0, self.raptor_iter - 1)
        currently_h_mode = int(self.vwrapper.hmode[hmode_iter_check])
        out_prev = self.outs[-1]

        # We don't know what the H-L threshold is, for this sim lets just create a fudge factor.
        PLH = out_prev["PLH"][0][-1]
        fudge_factor = 1.0  # TODO(allenw): currently forcing LMode to see results.
        PHL = fudge_factor * PLH

        # I don't really like this, but it should work.
        # Once the HL transition occurs set the rest of the sim to LMode.
        print(f"Ploss: {out_prev['Ploss'][0][-1]}, PHL: {PHL}")
        if out_prev["Ploss"][0][-1] > PHL and currently_h_mode:
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
        dIp_dt, Paux, ne_line_avg, nfuel_line_avg = action
        self.Ustep = update_ustep(self.Ustep, dIp_dt, Paux, self.raptor_dt)

        # Given an array of Ips, use g_interp to compute geometry parameters for said Ips.
        gstep = numpy_to_matlab(
            np.column_stack([self.g_interp(Ip) for Ip in self.Ustep[0, :]])
        )

        vstep = numpy_to_matlab(self.get_vstep(ne_line_avg, nfuel_line_avg))

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
        return

    def raptor_out(self):
        self.params["tgrid"] = self.tgrid[0][: self.raptor_iter]
        raptor_out = self.eng_handle.RAPTOR_out(
            self.simres, self.model, self.params, nargout=1
        )
        return raptor_out


if __name__ == "__main__":
    df = pd.read_pickle("rl_traj_for_raptor_test.pkl")
    gym = RaptorRDGym("/home/awang/raptor", 1e-2, 5)  # dts are 0.05s.
    gym.reset()

    for index, row in df.iterrows():
        # Perform unit conversions.
        # TODO(allenw): ugh, units/
        dIp_dt = 1e6 * row["dIp_dt"]
        Paux = 1e6 * row["Paux"]
        nfuel = 1e19 * row["nfuel19"]
        ne = 1e19 * row["ne19_vol_avg"]
        action = [dIp_dt, Paux, ne, nfuel]
        gym.step(action)

    raptor_out = gym.raptor_out()
    gym.eng_handle.workspace["out"] = raptor_out
    gym.eng_handle.save("/tmp/test.mat", "out", nargout=0)
