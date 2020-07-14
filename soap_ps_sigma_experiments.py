#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_all_pairs_experiment, gfr_pairwise_experiment

### Experiment metadata
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
nb_samples = 10000
cutoff = 4
normalize = False

## Tested hyperparameters
#sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
sigmas = [0.1, 0.3, 0.5]
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO"]:
        features_hypers = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "PowerSpectrum",
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": 10,
                "max_angular": 6,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": normalize
            }
        } for sigma in sigmas]

        gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer)

#for dataset_name in ["selection-10k.extxyz"]:
#    for radial_basis in ["GTO"]:
#        features_hypers1 = [{
#            "feature_type": "soap",
#            "feature_parameters": {
#                "soap_type": "PowerSpectrum",
#                "radial_basis": radial_basis,
#                "interaction_cutoff": cutoff,
#                "max_radial": 10,
#                "max_angular": 6,
#                "gaussian_sigma_constant": 0.1,
#                "gaussian_sigma_type": "Constant",
#                "cutoff_smooth_width": 0.5,
#                "normalize": normalize
#            }
#        } for _ in [0.3, 0.5]]
#
#        features_hypers2 = [{
#            "feature_type": "soap",
#            "feature_parameters": {
#                "soap_type": "PowerSpectrum",
#                "radial_basis": radial_basis,
#                "interaction_cutoff": cutoff,
#                "max_radial": 10,
#                "max_angular": 6,
#                "gaussian_sigma_constant": 0.1,
#                "gaussian_sigma_type": "Constant",
#                "cutoff_smooth_width": 0.5,
#                "normalize": normalize
#            }
#        } for sigma in [0.3, 0.5]]
#
#        gfr_pairwise_experiment(
#            dataset_name,
#            nb_samples,
#            features_hypers1,
#            features_hypers2,
#            two_split=two_split,
#            train_ratio=train_ratio,
#            seed=seed,
#            noise_removal=False,
#            regularizer=regularizer
#        )
