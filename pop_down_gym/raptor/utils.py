import matlab.engine
import numpy as np


def to_numpy(arr):
    """Convert a matlab array to numpy.

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """
    if len(arr.size) > 1:
        # For whatever reason, matlab.engine doesn't support toarray() on multidimensional arrays.
        return np.array(arr.tomemoryview().tolist())
    else:
        return np.array(arr.toarray())


def numpy_to_matlab(arr):
    """Convert a numpy array to matlab.

    Args:
        arr (_type_): _description_

    Returns:
        _type_: _description_
    """
    return matlab.double(arr.tolist())


def concat_matlab_arrays(arr1, arr2):
    """Concatentate two matlab arrays.

    Args:
        arr1 (_type_): _description_
        arr2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    return numpy_to_matlab(
        np.hstack((to_numpy(arr1).squeeze(), to_numpy(arr2).squeeze()))
    )


def concat_simres(simres1, simres2):
    """Concatenate two simres structs output from RAPTOR.

    Args:
        simres1 (_type_): _description_
        simres2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    for k in simres1.keys():
        if type(simres1[k]) == list:
            simres1[k] += simres2[k]
        else:
            simres1[k] = concat_matlab_arrays(simres1[k], simres2[k])
    return simres1


def update_ustep(
    Ustep: np.array, dIp_dt: float, Paux: float, raptor_dt: float
) -> np.array:
    """

    Args:
        Ustep (np.array): previous Ustep array.
        dIp_dt (float): plasma current ramp-rate.
        Paux (float): auxiliary power.
        raptor_dt (float): RAPTOR time-step.

    Returns:
        _type_: _description_
    """
    n_raptor_steps = Ustep.shape[1]
    time_steps = raptor_dt * np.arange(1, n_raptor_steps + 1)
    Ustep_new = Ustep.copy()
    last_Ip = Ustep[0, -1]
    Ustep_new[0, :] = np.add(last_Ip, np.multiply(dIp_dt, time_steps))
    Ustep_new[1, :] = Paux
    return Ustep_new


class VWrapper:
    """
    RAPTOR uses a struct called v to store configurables such as H-mode, prescribed profiles, etc.
    Interacting with it is a bit non-trivial, so this class wraps it.
    """

    def __init__(self, v0, model) -> None:
        self.v = v0
        self.model = model

    @property
    def ne_index(self):
        ne_index = to_numpy(self.model["ne"]["vind"]).squeeze() - 1
        ne_index = ne_index.astype(int)
        return ne_index

    @property
    def ni_index(self):
        ni_index = to_numpy(self.model["ni"]["vind"]).squeeze() - 1
        ni_index = ni_index.astype(int)
        return ni_index

    @property
    def hmode_index(self):
        hmode_index = int(self.model["hmode"]["vind"]["activation"] - 1)
        return hmode_index

    @property
    def te_bc_index(self):
        te_bc_index = int(self.model["te"]["BC"]["vind_value"] - 1)
        return te_bc_index

    @property
    def ti_bc_index(self):
        ti_bc_index = int(self.model["ti"]["BC"]["vind_value"] - 1)
        return ti_bc_index

    @property
    def hmode(self):
        return self.v[self.hmode_index]

    @property
    def te_bc(self):
        return self.v[self.te_bc_index]

    @property
    def ti_bc(self):
        return self.v[self.ti_bc_index]

    @property
    def ne(self):
        return self.v[self.ne_index]

    @property
    def ni(self):
        return self.v[self.ni_index]

    # Setters
    @hmode.setter
    def hmode(self, val):
        self.v[self.hmode_index] = val

    @te_bc.setter
    def te_bc(self, val):
        self.v[self.te_bc_index] = val

    @ti_bc.setter
    def ti_bc(self, val):
        self.v[self.ti_bc_index] = val

    @ne.setter
    def ne(self, val):
        self.v[self.ne_index] = val

    @ni.setter
    def ni(self, val):
        self.v[self.ni_index] = val

    def __getitem__(self, val):
        if isinstance(val, slice):
            return VWrapper(self.v[:, val], self.model)
        else:
            raise NotImplementedError
