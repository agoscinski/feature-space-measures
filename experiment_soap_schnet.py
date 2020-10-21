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
nb_samples = 501
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

for soap_type in ["BiSpectrum"]:
    #features_hypers1 = [{
    #    "feature_type": "soap",
    #    "feature_parameters": {
    #        "soap_type": soap_type,
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
            "filename": "gschnet_qm9_interaction_layer=9.npy",
            #"filename": "schnet_qm9_energy_U0_interaction_layer=6.npy",
            "filetype": "npy"
        }
    }]
    #features_hypers2 = [{
    #    "feature_type": "schnet",
    #    "feature_parameters": {
    #        "model_name": "schnet_qm9_energy_U0",
    #        "interaction_layer": -1
    #    }
    #}]

    hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")

    print(hash_value)
    print(gfre_vec)
    print()
