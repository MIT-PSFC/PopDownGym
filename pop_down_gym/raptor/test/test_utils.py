import numpy as np
from rd_rl.raptor.utils import update_ustep

def test_update_Ustep():
    raptor_dt = 0.01
    Ustep = np.zeros((2, 5))
    Ustep[0, :] = 8.7e6
    dIp_dt = -1e6
    Paux = 11e6
    Ustep2 = update_ustep(Ustep, dIp_dt, Paux, raptor_dt)
    assert Ustep2[0, 0] == Ustep[0, 0] + dIp_dt * raptor_dt
    assert Ustep2[0, 1] == Ustep[0, 0] + dIp_dt * 2 * raptor_dt
    assert Ustep2[0, 2] == Ustep[0, 0] + dIp_dt * 3 * raptor_dt
    assert Ustep2[0, 3] == Ustep[0, 0] + dIp_dt * 4 * raptor_dt
    assert Ustep2[0, 4] == Ustep[0, 0] + dIp_dt * 5 * raptor_dt
    assert (Ustep2[1, :] == Paux).all()