#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment

### Experiment metadata

nbs_features = [10,50,100,200]
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    nb_samples = 4000

    ## Constant hyperparameters
    cutoff = 4
    sigma = 0.5
    cutoff_smooth_width = 0.5
    normalize = False

    ## Tested hyperparameters
    features_hypers1 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "PowerSpectrum",
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": 10,
            "max_angular": 6,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        },
        "feature_selection_parameters" : {"nb_features": nb_features} # TODO I guess there will be more parameters
    } for nb_features in nbs_features]

    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "file_name": "features/behler_parinello-(10,6)-"+dataset_name+".npy", # TODO just pseudo code, 
        },
        "feature_selection_parameters" : {"nb_features": nb_features} # TODO I guess there will be more parameters
    } for nb_features in nbs_features]

    two_split = True
    if two_split:
        seed = 0x5f3759df
    else:
        seed = None
    noise_removal = False

    gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, seed, noise_removal)
