#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
nb_samples = 4000
# Constant hyperparameters
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False
nb_features = 40
two_split = True
if two_split:
    seed = 0x5f4759df
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizers = [1e-6]


for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]: #"selection-10k.extxyz",
    if dataset_name == "selection-10k.extxyz":
        cutoff = 4
        precomputed_name = 'methane'
        BP_sizes = [35, 191, 534, 1147]
        # max_radials_angulars sizes are 36, 192, 540, 1152
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5)]
    elif dataset_name == "C-VII-pp-wrapped.xyz":
        cutoff = 7.5
        precomputed_name = 'carbon'
        BP_sizes = [11, 61, 181, 377, 699]
        max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5), (10, 6)]
        #BP_sizes = [377]
        #max_radials_angulars = [(8, 5)]

        #BP_sizes = [699]
        #max_radials_angulars = [(10, 6)]


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
        "hilbert_space_parameters": {
            "distance_parameters": {"distance_type": "euclidean"},
            "kernel_parameters": {"kernel_type": "rbf", "gamma": 15}
        },
        #"feature_selection_parameters" : {"nb_features": nb_features}
    } for max_radial, max_angular in max_radials_angulars]

    features_hypers2 = [{
        "feature_type": "precomputed_BP",
        "feature_parameters": {
            #"file_root": "data/radial",
            "file_root": "data/Alex",
            "dataset": precomputed_name,
            "SF_count": count,
        },
        #"hilbert_space_parameters": {
        #    "distance_parameters": {"distance_type": "euclidean"},
        #    "kernel_parameters": {"kernel_type": "center"}
        #},
        #"feature_selection_parameters" : {"nb_features": nb_features},
    } for count in BP_sizes]

    for regularizer in regularizers:
        gfr_pairwise_experiment(
            dataset_name,
            nb_samples,
            features_hypers1,
            features_hypers2,
            two_split=two_split,
            train_ratio=train_ratio,
            seed=seed,
            noise_removal=False,
            regularizer=regularizer
        )
