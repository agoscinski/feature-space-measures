#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_all_pairs_experiment

nb_samples = 10000
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
noise_removal = False
regularizer = 1e-6

## Constant hyperparameters
cutoff = 4
cutoff_smooth_width = 0.5
normalize = False
sigma = 0.5
radial_scaling_exponents = [0,1,2]
### Experiment metadata
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO", "DVR"]:

        ## Constant hyperparameters

        ## Tested hyperparameters
        features_hypers = [{
            "feature_type": "soap",
            "feature_parameters":  {
                "soap_type": "PowerSpectrum",
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": 10,
                "max_angular": 6,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": False,
                "cutoff_function_type": "RadialScaling",
                "cutoff_function_parameters": {"rate": 1,
                                               "scale": 1,
                                               "exponent": exponent}
                }
        } for exponent in radial_scaling_exponents]

        gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer)
