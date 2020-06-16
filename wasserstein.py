from rascal.representations import SphericalInvariants
import numpy as np
from scipy.interpolate import interp1d

def compute_squared_wasserstein_distance(feature_paramaters, frames, environment_idx):
    if feature_paramaters["soap_type"] == "RadialSpectrum":
        compute_squared_radial_spectrum_wasserstein_distance(feature_paramaters, frames)
    elif feature_paramaters["soap_type"] == "PowerSpectrum":
        raise ValueError("The soap_type="+feature_paramaters["soap_type"]+" is not implemented yet.")
    else:
        raise ValueError("The soap_type="+feature_paramaters["soap_type"]+" is not known.")

def compute_squared_radial_spectrum_wasserstein_distance(feature_paramaters, frames, environment_idx, nb_grid_points=200):
    compute_radial_spectrum_wasserstein_features(feature_paramaters, frames, environment_idx)
    raise ValueError("Not Implemented")
    return np.zeros(5)

# 200 grid points is just chosen arbitrary to obtain an approximation which is accurate enough
def compute_radial_spectrum_wasserstein_features(feature_paramaters, frames, environment_idx, nb_grid_points=200):
    """Compute"""
    if  feature_paramaters["soap_parameters"]["soap_type"] != "RadialSpectrum":
        raise ValueError('Wasserstein features can be only computed for soap_type="RadialSpectrum".')
    if  feature_paramaters["soap_parameters"]["radial_basis"] != "DVR":
        raise ValueError('Wasserstein features can be only computed for radial_basis="DVR".')

    nb_basis_functions = feature_paramaters["soap_parameters"]["max_radial"]
    feature_paramaters["soap_parameters"]["max_radial"] = nb_grid_points
    normalize_wasserstein_features = feature_paramaters["soap_parameters"]["normalize"]
    feature_paramaters["soap_parameters"]["normalize"] = False
    cutoff = feature_paramaters["soap_parameters"]["interaction_cutoff"]

    # compute soap representation for interpolation
    representation = SphericalInvariants(**feature_paramaters["soap_parameters"])
    densities = representation.transform(frames).get_features(representation)[environment_idx]
    nb_envs = densities.shape[0]
    nb_species = densities.shape[1]//nb_grid_points
    densities = densities.reshape(nb_envs * nb_species, nb_grid_points)

    # DVR uses gaussian quadrature points as basis function, we reproduce the original grid points for the interpolation
    density_grid, density_weights = np.polynomial.legendre.leggauss(nb_grid_points)
    density_grid = density_grid*cutoff/2 + cutoff/2
    density_grid = np.hstack((0, density_grid))
    densities /= np.sqrt(density_weights)
    cdf = np.cumsum(densities, axis=1)

    if feature_paramaters["delta_normalization"]:
        cdf = cdf.reshape(nb_envs, nb_species, nb_grid_points)
        for i in range(nb_species):
            max_norm = np.max(cdf[:,i,-1])
            cdf[:,i,-1] += max_norm-cdf[:,i,-1]
        cdf = cdf.reshape(nb_envs * nb_species, nb_grid_points)

    # normalize nonzero environments
    nonzero_mask = cdf[:,-1] != 0
    cdf[nonzero_mask] /= cdf[:,-1][nonzero_mask][:,np.newaxis]
    # insert the zero probabilty point at the beginning to help interpolating at the beginning
    cdf = np.concatenate((np.zeros((cdf.shape[0],1)),cdf),axis=1)

    # gaussian quadratarue points as grid
    if feature_paramaters["grid_type"] == "gaussian_quadrature":
        interp_grid = np.polynomial.legendre.leggauss(nb_basis_functions)[0]/2 + 0.5
    elif feature_paramaters["grid_type"] == "equispaced":
        interp_grid = np.linspace(0, 1, nb_basis_functions)
    else:
        raise ValueError("The wasserstein grid_type="+feature_parameters["grid_type"] +" is not known.")

    wasserstein_features = np.zeros((nb_envs*nb_species, nb_basis_functions))
    for i in np.where(nonzero_mask)[0]: # subset of nb_envs*nb_species
        interpolator = interp1d(cdf[i,:], density_grid, assume_sorted=True)
        wasserstein_features[i,:] = interpolator(interp_grid)

    wasserstein_features = wasserstein_features.reshape(nb_envs, nb_species * nb_basis_functions)

    if  normalize_wasserstein_features:
        wasserstein_features /= np.linalg.norm(wasserstein_features,axis=1)[:,np.newaxis]
    return wasserstein_features
