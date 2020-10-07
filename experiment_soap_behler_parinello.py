#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
nb_samples = 4000
# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False
nb_features = 40
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(20)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

empty_hash_values = ""
bracket_hash_values = ""

experiment_ids = []
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    if dataset_name == "selection-10k.extxyz":
        dataset_shorthand = "methane"
        BP_sizes = [35, 191, 534, 1147]
        ## max_radials_angulars sizes are 36, 192, 540, 1152
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5)]
        #BP_sizes = [1147]
        # max_radials_angulars sizes are 36, 192, 540, 1152
        #max_radials_angulars = [(8, 5)]
    elif dataset_name == "C-VII-pp-wrapped.xyz":
        dataset_shorthand = "carbon"
        BP_sizes = [11, 61, 181, 377, 699]
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5), (10, 6)]

        #BP_sizes = [377]
        #max_radials_angulars = [(8, 5)]


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
        },
        #"hilbert_space_parameters": {
        #    "distance_parameters": {"distance_type": "euclidean"},
        #    "kernel_parameters": {"kernel_type": "center"}
        #},
        #"feature_selection_parameters" : {"nb_features": nb_features}
    } for max_radial, max_angular in max_radials_angulars]


    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "BPSF",
            "filename": f"{dataset_shorthand}_{count}SF",
            "filetype": "txt"
        },
        #"hilbert_space_parameters": {
        #    "distance_parameters": {"distance_type": "euclidean"},
        #    "kernel_parameters": {"kernel_type": "center"}
        #},
        #"feature_selection_parameters" : {"nb_features": nb_features},
    } for count in BP_sizes]

    hash_values = []
    gfre_mat = np.zeros((2,len(max_radials_angulars),len(seeds)))
    i=0
    for seed in seeds:
        hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False)
        hash_values.append(hash_value)
        gfre_mat[:,:,i] = gfre_vec
        i += 1
    means = np.mean(gfre_mat,axis=2)
    stds = np.std(gfre_mat,axis=2)
    print()
    print(dataset_name)
    print(gfre_mat)
    print("GFRE(SOAP, BPSF) means:", means[0])
    print("GFRE(BPSF, SOAP) means:", means[1])
    print("GFRE(SOAP, BPSF) stds:", stds[0])
    print("GFRE(BPSF, SOAP) stds:", stds[1])
    print()

    empty_hash_values += '"' + ' '.join(hash_values).replace(' ','" "' ) + '" '
    bracket_hash_values += f'bpsf_{dataset_name}_hash_value = '+'"' + ' '.join(hash_values).replace(' ','", "' ) + '"\n'

print(empty_hash_values)
print(bracket_hash_values )
