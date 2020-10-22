#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
nb_samples = 500
# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

# Experiment metadata
dataset_name = "qm9.db"
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

for soap_type in ["RadialSpectrum"]:
    features_hypers1 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": soap_type,
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": 4,
            "max_angular": 3,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        },
        "feature_selection_parameters": {
            "type": "PCA",
            "n_features": 128,
        },
        "hilbert_space_parameters": {
            "computation_type ": "explicit",
            "distance_parameters": {"distance_type": "euclidean"},
            "kernel_parameters": {"kernel_type": "polynomial", "degree": 2}
        }
        #"hilbert_space_parameters": {
        #    "distance_parameters": {"distance_type": "euclidean"},
        #    "kernel_parameters": {"kernel_type": "poly", "degree": 2}
        #}
    }]
    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "schnet",
            "filename": "schnet_qm9_energy_U0_nb_structures=10000_layer=0.npy",
            "filetype": "npy",
        }
    }]

    hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")

    print(hash_value)
    print(gfre_vec)
    print()