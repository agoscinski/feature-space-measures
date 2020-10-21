#!/usr/bin/env python3
# coding: utf-8
from src.experiment import lfre_pairwise_experiment
from src.feature_space_measures import compute_lfre_from_deprecated_pointwise_lfre
import os


os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
nb_samples = 501
nb_local_envs=50
# Constant hyperparameters
cutoff = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"
dataset_name = "qm9.db"
for nb_local_envs in [1,2,3,4,5]:
    #features_hypers1 = [{
    #    "feature_type": "soap",
    #    "feature_parameters": {
    #        "soap_type": "BiSpectrum",
    #        "radial_basis": "GTO",
    #        "interaction_cutoff": cutoff,
    #        "max_radial": 4,
    #        "max_angular": 3,
    #        "gaussian_sigma_constant": sigma,
    #        "gaussian_sigma_type": "Constant",
    #        "cutoff_smooth_width": cutoff_smooth_width,
    #        "normalize": normalize
    #    },
    #    "feature_selection_parameters": {
    #        "type": "PCA",
    #        "n_features": 250,
    #    }
    #}]
    features_hypers1 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "Schnet",
            "filename": "gschnet_qm9_interaction_layer=6.npy",
            #"filename": "schnet_qm9_energy_U0_interaction_layer=6.npy",
            "filetype": "npy"
        }
    }]


    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "Schnet",
            #"filename": "schnet_qm9_energy_U0.npy",
            "filename": "gschnet_qm9_interaction_layer=9.npy",
            "filetype": "npy"
        }
    }]

    hash_value, pointwise_lfre = lfre_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, train_ratio, regularizer, one_direction=False, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")

    print(hash_value)
    print(compute_lfre_from_deprecated_pointwise_lfre(pointwise_lfre))
