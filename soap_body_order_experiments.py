#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_all_pairs_experiment

### Experiment metadata
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.6
else: 
    train_ratio = None
    seed = None

noise_removal = False
regularizer = "CV"
nb_samples = 10000

## Constant hyperparameters
cutoff = 4
max_radial = 6
max_angular = 4
sigma = 0.5
normalize = False

## Tested hyperparameters
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO", "DVR"]:
        soap_types = ["RadialSpectrum", "PowerSpectrum", "BiSpectrum"]
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
                "cutoff_smooth_width": 0.5,
                "normalize": normalize
            },
            #"hilbert_space_parameters": {
            #    "distance_parameters": {"distance_type": "euclidean"},
            #    #"kernel_parameters": {"kernel_type": "center"}
            #    "kernel_parameters": {"kernel_type": "rbf", "gamma": 1}
            #}
 
        } for soap_type in soap_types]

        gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer)
