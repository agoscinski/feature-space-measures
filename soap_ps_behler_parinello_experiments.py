#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment

# Experiment metadata
nb_samples = 4000
# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    if dataset_name == "selection-10k.extxyz":
        precomputed_name = 'methane'
        BP_sizes = [35, 191, 534, 1147]
        # max_radials_angulars sizes are 36, 192, 540, 1152
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5)]
    elif dataset_name == "C-VII-pp-wrapped.xyz":
        precomputed_name = 'carbon'
        BP_sizes = [11, 61, 181, 377, 699]
        # max_radials_angulars sizes are 12, 64, 180, 384, 700
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5), (10, 6)]

    features_hypers1 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "PowerSpectrum",
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        }
    } for max_radial, max_angular in max_radials_angulars]

    features_hypers2 = [{
        "feature_type": "precomputed_BP",
        "feature_parameters": {
            "file_root": "data/Alex",
            "dataset": precomputed_name,
            "SF_count": count,
        },
    } for count in BP_sizes]

    two_split = True
    if two_split:
        seed = 0x5f3759df
    else:
        seed = None
    noise_removal = False

    gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, seed, noise_removal)
