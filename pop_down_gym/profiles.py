import equinox as eqx
import jax.numpy as jnp
import xarray as xr
from sklearn.decomposition import PCA


def pca_profiles(
    dataset: xr.Dataset, profile_to_ncomps: dict[str, int]
) -> dict[str, PCA]:
    """Perform principal component analysis on profiles in a dataset.

    Args:
        dataset (xr.Dataset): dataset containing profiles to perform PCA on.
        profile_to_ncomps (dict[str, int]): mapping variable names of profiles to the number of components to use in the PCA.

    Returns:
        dict[str, PCA]: mapping variable names of profiles to the PCA object containing the results of the PCA.
    """
    dataset_stacked = dataset.stack(time_slices=["time", "episode"]).dropna(
        dim="time_slices"
    )
    pcas = dict()
    for profile, dim in profile_to_ncomps.items():
        pca = PCA(n_components=dim)
        pca.fit(dataset_stacked[profile].to_numpy().T)
        pcas[profile] = pca
    dataset_stacked = dataset_stacked.unstack(dim="time_slices")
    return pcas


def apply_pcas(dataset: xr.Dataset, pcas):
    stacked_dataset = dataset.stack(time_slices=["time", "episode"]).dropna(
        dim="time_slices"
    )
    pca_var_names = []
    for profile, pca in pcas.items():
        components = pcas[profile].transform(stacked_dataset[profile].to_numpy().T)

        # Create a new variable in the dataset for each component.
        for i in range(pca.n_components_):
            comp_da = xr.DataArray(
                components[:, i],
                dims=["time_slices"],
                coords={"time_slices": stacked_dataset["time_slices"]},
            )
            pca_var_name = f"{profile}_pca{i}"
            pca_var_names.append(pca_var_name)
            stacked_dataset[pca_var_name] = comp_da
            stacked_dataset[pca_var_name].attrs["v"] = pca.components_[i]

        # Apply PCA bias and variable names of components to the original profile variable
        stacked_dataset[profile].attrs["pca_bias"] = pca.mean_
        stacked_dataset[profile].attrs["pca_var_names"] = pca_var_names
    dataset = stacked_dataset.unstack(dim="time_slices")
    return dataset, pca_var_names


class SimpleProfileBasis(eqx.Module):
    """
    Represent profiles as:
        Q(rho) ~ c*v(rho) + b(rho)
    where v and b are principal components and c is a coefficient.
    Legendre-Gauss grid in rho with weights wgauss.
    """

    v: jnp.ndarray
    b: jnp.ndarray
    rhogauss: jnp.ndarray  # Legendre-Gauss quadrature grid points.
    wgauss: jnp.ndarray  # Legendre-Gauss quadrature weights.

    def __init__(
        self, v: jnp.ndarray, b: jnp.ndarray, rhogauss: jnp.ndarray, wgauss: jnp.ndarray
    ):
        self.v = v
        self.b = b
        self.rhogauss = rhogauss
        self.wgauss = wgauss

    def trainable_params_filter_spec(self):
        def filter_spec(x):
            return x is self.v or x is self.b
        return filter_spec
        
    def volume_average(self, profile: jnp.ndarray, Vp: jnp.ndarray) -> float:
        """Compute the volume average of a profile.

        Args:
            profile (jnp.ndarray): a profile, defined on the Legendre-Gauss grid, to compute the volume average of.

        Returns:
            float: volume average of the profile.
        """
        volume = jnp.dot(self.wgauss, Vp)
        return jnp.dot(jnp.multiply(self.wgauss, Vp), profile) / volume

    def line_average(self, profile: jnp.ndarray) -> float:
        """Compute the line average of a profile.

        Args:
            profile (jnp.ndarray): a profile, defined on the Legendre-Gauss grid, to compute the line average of.

        Returns:
            float: line average of the profile.
        """
        return jnp.dot(self.wgauss, profile)

    def volume_average_to_profile(self, Qvol: float, Vp: jnp.ndarray) -> jnp.ndarray:
        """Convert a volume average quantity to a profile.

        Args:
            Qvol (float): volume average quantity.
            Vp (jnp.ndarray): derivative of plasma volume w.r.t. rho on the Legendre-Gauss grid.

        Returns:
            jnp.ndarray: profile corresponding to the volume average quantity.
        """
        num = jnp.dot(self.wgauss, Vp) * Qvol - jnp.dot(
            jnp.multiply(self.wgauss, self.b), Vp
        )
        denom = jnp.dot(jnp.multiply(self.wgauss, self.v), Vp)
        c = num / denom  # Coefficient for the temperature profile.
        profile = c * self.v + self.b
        return profile

    def line_average_to_profile(self, Qline: float) -> jnp.ndarray:
        """Convert a line average quantity to a profile.

        Args:
            Qline (float): line average quantity.

        Returns:
            jnp.ndarray: profile corresponding to the line average quantity.
        """
        num = Qline - jnp.dot(self.wgauss, self.b)
        denom = jnp.dot(self.wgauss, self.v)
        c = num / denom
        profile = c * self.v + self.b
        return profile

    def volume_average_to_line_average(
        self, Q_vol_avg: float, Vp: jnp.ndarray
    ) -> float:
        profile = self.volume_average_to_profile(Q_vol_avg, Vp, self.v, self.b)
        return self.line_average(profile)

    def line_average_to_volume_average(
        self, Q_line_avg: float, Vp: jnp.ndarray
    ) -> float:
        profile = self.line_average_to_profile(Q_line_avg, self.v, self.b)
        return self.volume_average(profile, Vp)


class ProfileBases(eqx.Module):
    Te_basis: SimpleProfileBasis
    Ti_basis: SimpleProfileBasis
    ne_basis: SimpleProfileBasis
    ni_basis: SimpleProfileBasis

    def __init__(self, Te_basis, Ti_basis, ne_basis, ni_basis):
        self.Te_basis = Te_basis
        self.Ti_basis = Ti_basis
        self.ne_basis = ne_basis
        self.ni_basis = ni_basis

        # Check that all bases use the same grid.
        assert (self.Te_basis.rhogauss == self.Ti_basis.rhogauss).all()
        assert (self.Te_basis.rhogauss == self.ne_basis.rhogauss).all()
        assert (self.Te_basis.rhogauss == self.ni_basis.rhogauss).all()
        assert (self.Te_basis.wgauss == self.Ti_basis.wgauss).all()
        assert (self.Te_basis.wgauss == self.ne_basis.wgauss).all()
        assert (self.Te_basis.wgauss == self.ni_basis.wgauss).all()

    def trainable_params_filter_spec(self):
        def filter_spec(x):
            funcs = [
                self.Te_basis.trainable_params_filter_spec(),
                self.Ti_basis.trainable_params_filter_spec(),
                self.ne_basis.trainable_params_filter_spec(),
                self.ni_basis.trainable_params_filter_spec(),
            ]
            return any([f(x) for f in funcs])
        return filter_spec
            
    @property
    def rhogauss(self):
        return self.Te_basis.rhogauss

    @property
    def wgauss(self):
        return self.Te_basis.wgauss

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset) -> "ProfileBases":
        n_comp = 1  # This basis only uses one component for each profile.
        pcas = pca_profiles(
            dataset,
            {
                "te_kev": n_comp,
                "ti_kev": n_comp,
                "ne19_prof": n_comp,
                "ni19_prof": n_comp,
            },
        )
        pca_ds, pca_var_names = apply_pcas(dataset, pcas)
        rhogauss = dataset["rho"].to_numpy().squeeze()
        wgauss = dataset["wgauss"].to_numpy().squeeze()
        Te_basis = SimpleProfileBasis(
            v=jnp.array(pca_ds["te_kev_pca0"].attrs["v"]),
            b=jnp.array(pca_ds["te_kev"].attrs["pca_bias"]),
            rhogauss=jnp.array(rhogauss),
            wgauss=jnp.array(wgauss),
        )
        Ti_basis = SimpleProfileBasis(
            v=jnp.array(pca_ds["ti_kev_pca0"].attrs["v"]),
            b=jnp.array(pca_ds["ti_kev"].attrs["pca_bias"]),
            rhogauss=jnp.array(rhogauss),
            wgauss=jnp.array(wgauss),
        )
        ne_basis = SimpleProfileBasis(
            v=jnp.array(pca_ds["ne19_prof_pca0"].attrs["v"]),
            b=jnp.array(pca_ds["ne19_prof"].attrs["pca_bias"]),
            rhogauss=jnp.array(rhogauss),
            wgauss=jnp.array(wgauss),
        )
        ni_basis = SimpleProfileBasis(
            v=jnp.array(pca_ds["ni19_prof_pca0"].attrs["v"]),
            b=jnp.array(pca_ds["ni19_prof"].attrs["pca_bias"]),
            rhogauss=jnp.array(rhogauss),
            wgauss=jnp.array(wgauss),
        )

        return (
            cls(Te_basis, Ti_basis, ne_basis, ni_basis),
            pca_ds,
            pcas,
        )
