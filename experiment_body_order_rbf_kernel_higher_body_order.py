#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment
import os

### Experiment metada

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

two_split = True
if two_split:
    seed = 0x5f3759df+4
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
noise_removal = False
regularizer = 1e-5#"CV 2 fold"
nb_samples = 4000

inner_epsilon = None
outer_epsilon = None

## Constant hyperparameters
radial_basis = "GTO"
cutoff = 4
max_radial = 6
max_angular = 4
sigma = 0.5
normalize = False

experiment_ids = []
gammas = [0.01, 0.1, 1, 10, 100]
for dataset_name in ["selection-10k.extxyz"]:
    for soap_types in [("RadialSpectrum", "PowerSpectrum"), ("PowerSpectrum", "BiSpectrum")]:
    #for soap_types in [("PowerSpectrum", "BiSpectrum")]:
        ## Tested hyperparameters
        features_hypers1 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_types[0],
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": normalize
        }}]

        features_hypers1.extend([{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_types[0],
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": normalize
            },
            "hilbert_space_parameters": {
                "computation_type" : "implicit_distance",
                "distance_parameters": {"distance_type": "euclidean"},
                "kernel_parameters": {"kernel_type": "rbf", "gamma": gamma}
            }} for gamma in gammas])

        features_hypers2 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_types[1],
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": normalize
            },
        } for _ in range(len(features_hypers1))]

        hash_value, _ = gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=False, one_direction=True, train_test_gfrm=True)
        print(f"dataset_name={dataset_name} soap_types={soap_types} gammas={gammas} hash_value={hash_value}")
        experiment_ids.append(hash_value)
print(experiment_ids)
