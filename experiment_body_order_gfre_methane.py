#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_all_pairs_experiment
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

### Experiment metadata

nb_samples = 4000
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else: 
    train_ratio = None
    seed = None
noise_removal = False
regularizer = "CV 2 fold"

## Constant hyperparameters
cutoff = 4
max_radial = 6
max_angular = 4
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

empty_hash_values = ""
bracket_hash_values = ""

## Tested hyperparameters
for dataset_name in ["selection-10k.extxyz"]:
    for radial_basis in ["GTO"]:
        soap_types = ["RadialSpectrum", "PowerSpectrum", "BiSpectrum"]
        features_hypers = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": soap_type,
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": max_angular,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
        } for soap_type in soap_types]

        if dataset_name == "selection-10k.extxyz":
            filename = "methane-allc.npy"
        elif dataset_name == "C-VII-pp-wrapped.xyz":
            filename = "carbon-first.npy"
        nice_feature_hypers = {
            "feature_type": "precomputed",
            "feature_parameters": {
                "feature_name": "NICE",
                "filename": filename,
                "filetype": "npy"
            }
        }
        features_hypers.append(nice_feature_hypers)

        hash_values = [gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer)]
        empty_hash_values += '"' + ' '.join(hash_values).replace(' ','" "' ) + '" '
        bracket_hash_values += f'body_order_{radial_basis}_{dataset_name}_hash_value = '+'"' + ' '.join(hash_values).replace(' ','", "' ) + '"\n'

print(empty_hash_values)
print(bracket_hash_values )
