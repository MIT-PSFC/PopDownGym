import math
import numpy as np
import time


def out_to_obs(out):
    observables = dict({
        "Ip": out['Ip'][-1][-1],
        "Vb": out['upl'][-1][-1], # Loop voltage at the edge.
        "psib": out['psi'][-1][-1],
        "Li": out['Li'][0][-1],
        "Wth": out['Wth'][0][-1],
        "Wpol": out['Wpol'][0][-1],
        "Zeff": to_numpy(out['ze'])[:, -1],

    })
    return observables

def plot_observations(obs_list, vars_to_plot):
    import matplotlib.pyplot as plt# Get g as a function of Ip.
    f, axs = plt.subplots(len(vars_to_plot), 1, sharex=True)
    for i, var in enumerate(vars_to_plot):
        axs[i].plot([obs[var] for obs in obs_list])
        axs[i].set_ylabel(var)
    plt.show()



states = [x0]
outs = [out0]
obs = [out_to_obs(out0)]


for i in range(0, ntotal_steps, raptor_steps_per_step):
    #
    # Check H/L Mode.
    #
    PLH = outs[-1]['PLH'][0][-1]
    fudge_factor = 0.6
    PHL = fudge_factor * PLH # We don't know what the H-L threshold is, for this sim lets just create a fudge factor.

    # Minus 1 to go from MATLAB to Python indexing.
    hmode_index = int(model['hmode']['vind']['activation'] - 1) 
    te_bc_index = int(model['te']['BC']['vind_value'] - 1)
    ti_bc_index = int(model['ti']['BC']['vind_value'] - 1)
    currently_h_mode = int(v[hmode_index][i])

    if outs[-1]['Ploss'][0][-1] > PHL and currently_h_mode:
        # Continue being in HMode.
        for j in range(i, ntotal_steps):
            v[hmode_index][j] = 1
            v[te_bc_index][j] = config['hmode']['params']['te_rhoped']
            v[ti_bc_index][j] = config['hmode']['params']['ti_rhoped']
    else:
        # Switch to LMode.
        # Note: if the sim switches to L-Mode and Ploss goes back above PHl, we don't go back into HMode
        # This is due to hysteresis.
        for j in range(i, ntotal_steps):
            v[hmode_index][j] = 0
            v[te_bc_index][j] = config['hmode']['params']['te_rhoedge']
            v[ti_bc_index][j] = config['hmode']['params']['ti_rhoedge']

    # Build the control, geometry, and external data matrices for the next step.
    Ustep = matlab.double(U0[:, i:i+raptor_steps_per_step].tolist())

    # Given an array of Ips, use g_interp to compute geometry parameters for said Ips.
    gstep = numpy_to_matlab(np.column_stack([g_interp(Ip) for Ip in Ustep[0]]))

    vstep = matlab.double(v[:, i:i+raptor_steps_per_step].tolist())
    params['tgrid'] = tgrid[0][i:i+raptor_steps_per_step]
    x0, simres_new, out = eng.step_raptor(x0, gstep, vstep, Ustep, model, params, raptor_steps_per_step, nargout=3)


    simres = concat_simres(simres, simres_new)
    observations = out_to_obs(out)

    states.append(x0)
    outs.append(out)
    obs.append(observations)

params['tgrid'] = config['grid']['tgrid']
final_out = eng.RAPTOR_out(simres, model, params, nargout=1)
eng.workspace['out'] = final_out
eng.save('/tmp/test.mat', 'out', nargout=0)
# vars_viz = ["Ip", "Li", "Wth"]
# plot_observations(obs, vars_viz)