#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment
import numpy as np
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

nb_samples = 4000
two_split = True
if two_split:
    seed = 0x5f3759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
noise_removal = False
regularizer = "CV 2 fold"

# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

empty_hash_values = ""
bracket_hash_values = ""

### Experiment metadata

for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO", "DVR"]:
        ## Tested hyperparameters
        max_radials_angulars = [(2,4),(4,4),(6,4),(8,4),(10,4),(12,4),(14,4),(16,4),(18,4)]
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
            },
            #"hilbert_space_parameters": {
            #    "distance_parameters": {"distance_type": "euclidean"},
            #    #"kernel_parameters": {"kernel_type": "center"}
            #    "kernel_parameters": {"kernel_type": "rbf", "gamma": 1}
            #}
        } for max_radial, max_angular in max_radials_angulars]

        max_radials_angulars = [(n+1,l) for n, l in max_radials_angulars]
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
            },
            #"hilbert_space_parameters": {
            #    "distance_parameters": {"distance_type": "euclidean"},
            #    #"kernel_parameters": {"kernel_type": "center"}
            #    "kernel_parameters": {"kernel_type": "rbf", "gamma": 10}
            #}
        } for max_radial, max_angular in max_radials_angulars]

        hash_values = []
        gfre_mat = np.zeros((2, len(max_radials_angulars), len(seeds)))
        i=0
        for seed in seeds:
            hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer)
            hash_values.append(hash_value)
            gfre_mat[:,:,i] = gfre_vec
            i += 1
        means = np.mean(gfre_mat,axis=2)
        stds = np.std(gfre_mat,axis=2)
        print()
        print(dataset_name)
        print(f"GFRE({radial_basis} n, n+1) means:", means[0])
        print(f"GFRE({radial_basis} n, n+1) stds:", stds[0])
        print()
        empty_hash_values += '"' + ' '.join(hash_values).replace(' ','" "' ) + '" '
        bracket_hash_values += f'max_radial_{radial_basis}_{dataset_name}_hash_values = '+'["' + ' '.join(hash_values).replace(' ','", "' ) + '"]\n'


print(empty_hash_values)
print(bracket_hash_values )
