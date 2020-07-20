#!/usr/bin/env python3
# coding: utf-8
from experiment import lfre_pairwise_experiment
from experiment import gfr_pairwise_experiment

### Experiment metada

dataset_name = "manif-minus-plus.extxyz"
nb_samples = 162
nbs_local_envs = [10,15,20]

#dataset_name = "C-VII-pp-wrapped.xyz"
#nb_samples = 4000
#nbs_local_envs = [10]

inner_epsilon = None#1e-5
outer_epsilon = None#1e-1

## Constant hyperparameters
radial_basis = "GTO"
cutoff = 4
max_radial = 6
max_angular = 4
sigma = 0.5
normalize = False

two_split = True
if two_split:
    seed = 0x5f4759df
    train_ratio = 0.6
else:
    seed = None
    train_ratio = None
regularizer = "CV"

## Tested hyperparameters
features_hypers1 = [{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "PowerSpectrum",
        "radial_basis": radial_basis,
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": 0.5,
        "normalize": normalize
    },
}]
features_hypers2 = [{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "BiSpectrum",
        "radial_basis": radial_basis,
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": 0.5,
        "normalize": normalize
    },
}]

#noise_removal = False
#gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, seed, noise_removal)

for nb_local_envs in nbs_local_envs:
    lfre_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, train_ratio, regularizer, inner_epsilon, outer_epsilon)
