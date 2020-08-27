#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment
import numpy as np

dataset_name = "displaced-methane-step_size=0.05-range=[0.5,4.5]-seed=None.extxyz"
two_split = True
if two_split:
    seed = 0x5f3759df
else: 
    train_ratio = None
    seed = None
regularizer = 1e-3 #"CV 2 fold"
nb_samples = ""

# feature space hyperparameteres
## Constant hyperparameters
cutoff = 4
max_radial = 200
cutoff_smooth_width = 0.5
normalize = False
hash_values = []

## Tested hyperparameters
features_hypers1 = []
features_hypers2 = []
for sigma in [0.1,0.3,0.5]:
    features_hypers1.append({
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "RadialSpectrum",
            "radial_basis": "DVR",
            "interaction_cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": 0,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        },
    })

    features_hypers1.append({
        "feature_type": "wasserstein",
        "feature_parameters": {
            "grid_type": "gaussian_quadrature",
            "nb_basis_functions": max_radial,
            "delta_normalization": True,
            "delta_sigma": None,
            "delta_offset_percentage": 0,
            "soap_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "DVR",
                "interaction_cutoff": cutoff,
                "max_radial": 200,
                "max_angular": 0,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            }
        },
    })

features_hypers1.append({
    "feature_type": "sorted_distances",
    "feature_parameters": {
        "interaction_cutoff": cutoff,
        "padding_type": "max"
    }
})

features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "file_root": "data/",
            "dataset": "pulled_hydrogen_distance",
        }} for _ in range(len(features_hypers1))]

hash_values = []
for train_ratio in ["equispaced 2", "equispaced 20"]:
    hash_value, _ = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, one_direction=True, compute_distortion = False, train_test_gfrm=True)
    hash_values.append(hash_value)

print(hash_values)
