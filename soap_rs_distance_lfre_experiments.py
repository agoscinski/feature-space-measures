#!/usr/bin/env python3
# coding: utf-8
from experiment import lfre_experiment

### Experiment metadata

dataset_name = "dragged-methane.extxyz"
# feature space measures hyperparameteres
two_split = False
if two_split:
    seed = 0x5f3759df
else:
    seed = None
nb_samples = 1000
latent_feature_name = "hydrogen_distance"

# feature space hyperparameteres
## Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False


## Tested hyperparameters
features_hypers = [{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "RadialSpectrum",
        "radial_basis": "DVR",
        "interaction_cutoff": cutoff,
        "max_radial": 25,
        "max_angular": 0,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": cutoff_smooth_width,
        "normalize": normalize
    }
}]

for delta_normalization in [False, True]:
    features_hypers.append({
        "feature_type": "wasserstein",
        "feature_parameters": {
            "grid_type": "gaussian_quadrature",
            "delta_normalization": delta_normalization,
            "soap_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "DVR",
                "interaction_cutoff": cutoff,
                "max_radial": 25,
                "max_angular": 0,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            }
        }
    })

lfre_experiment(dataset_name, nb_samples, features_hypers, two_split, seed, latent_feature_name)
