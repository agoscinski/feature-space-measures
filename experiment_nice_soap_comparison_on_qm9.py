#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment
import os, sys

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

import resource
def memory_limit(nb_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (nb_bytes, hard))
# limits memory usage to 800 GB
memory_limit(8e+11)


# Constant hyperparameters
cutoff = 4 # schnet, gschnet

#sigma = 0.2041 # schnet
#cutoff_smooth_width = 10 # schnet
sigma = 0.5#0.4167 # gschnet
cutoff_smooth_width = 0.1 # gschnet, simulate hard cutoff

max_radial = 6
max_angular = 4
normalize = False

# Experiment metadata
dataset_name = "qm9.extxyz"
nb_samples = sys.argv[1] if len(sys.argv) > 1 else 100
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"



features_hypers1 = [{
    "feature_type": "nice",
    "feature_parameters": {
        "nb_blocks": 2,
        "nus": [1,2,3],
        "n_components": 100,
        "mode": "invariants",
        "spherical_coeffs": {
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": max_angular,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            #"normalize": normalize
        }
    },
    "feature_selection_parameters": {
        "type": "PCA",
        #"explained_variance_ratio": 0.99,
        "explained_variance": 1e-9,
    },
}]
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
    "feature_selection_parameters": {
        "type": "PCA",
        #"explained_variance_ratio": 0.99,
        "explained_variance": 1e-9,
    },
}]

hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments", target="Structure")
print(hash_value)
print(gfre_vec)
print()
