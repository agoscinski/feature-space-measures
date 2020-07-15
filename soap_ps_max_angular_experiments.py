#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment

### Experiment metadata

nb_samples = 4000
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else:
    seed = None
noise_removal = False
regularizer = 1e-6

# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO", "DVR"]:


        ## Tested hyperparameters
        max_radials_angulars = [(4,2),(4,4),(4,6),(4,8),(4,10),(4,12),(4,14),(4,16),(4,18)]
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
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            }
        } for max_radial, max_angular in max_radials_angulars]

        max_radials_angulars = [(n,l+1) for n, l in max_radials_angulars]
        features_hypers2 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "PowerSpectrum",
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            }
        } for max_radial, max_angular in max_radials_angulars]

        gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, train_ratio, seed, noise_removal, regularizer)
