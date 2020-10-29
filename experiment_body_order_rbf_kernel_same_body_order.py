#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_all_pairs_experiment

### Experiment metadata

## Constant hyperparameters
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
noise_removal = False
regularizer = "CV 2 fold"
nb_samples = 4000


cutoff = 4
sigma = 0.5
radial_basis = "GTO"
cutoff_smooth_width = 0.5
max_radial = 6
max_angular = 4
normalize = False

## Tested hyperparameters
gammas = [0.1,1,10]
hash_values = []
for dataset_name in ["selection-10k.extxyz"]:
    for soap_type in ["RadialSpectrum", "PowerSpectrum"]:
        features_hypers = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_type,
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
        }]
        features_hypers.extend([{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_type,
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
            "feature_selection_parameters": {
                "type": "PCA",
                "explained_variance_ratio": 0.99,
            },
            "hilbert_space_parameters": {
                "computation_type" : "implicit_distance",
                "distance_parameters": {"distance_type": "euclidean"},
                "kernel_parameters": {"kernel_type": "rbf", "gamma": gamma}
            }
        } for gamma in gammas])

        hash_value = gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer, compute_distortion=False)
        hash_values.append(hash_value)
        print(f"dataset_name={dataset_name} soap_type={soap_type} hash_value={hash_value}")
print('"' + ' '.join(hash_values).replace(' ','" "' ) + '" ')
print(hash_values)
