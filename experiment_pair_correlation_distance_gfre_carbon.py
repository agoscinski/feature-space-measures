#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment, gfr_all_pairs_experiment
import numpy as np

### Experiment metadata

two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else:
    seed = None
noise_removal = False
nb_samples = 4000

regularizer = "CV 2 fold"

## Constant hyperparameters
cutoff = 4
cutoff_smooth_width = 0.5
normalize = False
delta_sigma = None
delta_offset_percentage = 0
delta_normalization = 1

## Tested hyperparameters
max_radial = 200 
sigmas_wasserstein = [0.1, 0.5, 0.1, 0.5]
sigmas_euclidean = [0.1, 0.1, 0.5, 0.5]

#seeds = [seed + i for i in range(20)] # seed creator
seeds = [1597463007]
hash_values = []
#"C-VII-pp-wrapped.xyz", "selection-10k.extxyz"
for dataset_name in ["selection-10k.extxyz"]:
    ## Tested hyperparameters
    features_hypers1 = [{
        "feature_type": "wasserstein",
        "feature_parameters": {
            "nb_basis_functions": max_radial,
            "grid_type": "gaussian_quadrature",
            "delta_normalization": delta_normalization,
            "delta_sigma": delta_sigma,
            "delta_offset_percentage": delta_offset_percentage,
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
            },
        }
    } for sigma in sigmas_wasserstein]

    features_hypers2 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "RadialSpectrum",
            "radial_basis": "DVR",
            "interaction_cutoff": cutoff,
            "max_radial": 200,
            "max_angular": 0,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        },
    } for sigma in sigmas_euclidean]

    gfre_mat = np.zeros((2,len(sigmas_wasserstein),len(seeds)))
    i=0
    for seed in seeds:
        hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer)
        hash_values.append(hash_value)
        gfre_mat[:,:,i] = gfre_vec
        i += 1
    means = np.mean(gfre_mat,axis=2)
    stds = np.std(gfre_mat,axis=2)
    wasserstein_name = "delta" if delta_normalization else "scaling"
    print()
    print(dataset_name, wasserstein_name+" vs Euclidean")
    print(gfre_mat)
    print("W sigmas:",sigmas_wasserstein, "E sigmas:", sigmas_euclidean)
    print("GFRE("+wasserstein_name+",Euclidean) means:", means[0])
    print("GFRE(Euclidean,"+wasserstein_name+") means:", means[1])
    print("GFRE("+wasserstein_name+",Euclidean) stds:", stds[0])
    print("GFRE(Euclidean,"+wasserstein_name+") stds:", stds[1])
    print(hash_values)
    print('"' + ' '.join(hash_values).replace(' ','" "' ) + '"')
    print()

offset = len(seeds)
print('delta_carbon_hash_values = '+'["' + ' '.join(hash_values).replace(' ','", "' ) + '"]')


sigmas = [0.1, 0.5]

features_hypers = [{
    "feature_type": "wasserstein",
    "feature_parameters": {
        "nb_basis_functions": max_radial,
        "grid_type": "gaussian_quadrature",
        "delta_normalization": delta_normalization,
        "delta_sigma": delta_sigma,
        "delta_offset_percentage": delta_offset_percentage,
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
        },
    }
} for sigma in sigmas]

features_hypers.extend([{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "RadialSpectrum",
        "radial_basis": "DVR",
        "interaction_cutoff": cutoff,
        "max_radial": 200,
        "max_angular": 0,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": cutoff_smooth_width,
        "normalize": normalize
    },
} for sigma in sigmas])


print(gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer))
