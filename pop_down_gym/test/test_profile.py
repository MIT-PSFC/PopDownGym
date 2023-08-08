
import xarray as xr
from pop_down_gym.profile import SimpleProfileBasis, ProfileBases
from pop_down_gym.data.load import load_data

def test_simple_profile_basis():
    ds = load_data()
    basis, ds, pcas = ProfileBases.from_dataset(ds)

if __name__ == "__main__":
    test_simple_profile_basis()