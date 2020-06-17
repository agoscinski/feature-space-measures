#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment

### Experiment metadata

for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for delta_normalization in [True, False]:
        nb_samples = 2000

        ## Constant hyperparameters
        cutoff = 4
        sigma = 0.5
        cutoff_smooth_width = 0.5
        normalize = False

        ## Tested hyperparameters
        max_radials_angulars = [(10*i,0) for i in range(1,21)]
        features_hypers2 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "DVR",
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
            "hilbert_space_parameters": {
                "distance_parameters": {"distance_type": "wasserstein", "grid_type": "gaussian_quadrature", "delta_normalization": delta_normalization},
                "kernel_parameters": {"kernel_type": "center"}
                }
        } for max_radial, max_angular in max_radials_angulars]

        features_hypers1 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "DVR",
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
            "hilbert_space_parameters": {
                "distance_parameters": {"distance_type": "euclidean"},
                "kernel_parameters": {"kernel_type": "center"}
                }
       } for max_radial, max_angular in max_radials_angulars]

        two_split = True
        if two_split:
            seed = 0x5f3759df
        else:
            seed = None
        noise_removal = False

        gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, seed, noise_removal)