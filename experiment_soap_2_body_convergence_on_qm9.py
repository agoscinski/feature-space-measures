#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment, lfre_pairwise_experiment
from src.feature_space_measures import compute_lfre_from_deprecated_pointwise_lfre
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Constant hyperparameters
cutoff = 10 # schnet, gschnet

sigma = 0.2041 # schnet
cutoff_smooth_width = 10 # schnet
#sigma = 0.4167 # gschnet
#cutoff_smooth_width = 0.001 # gschnet, simulate hard cutoff

normalize = False

# data metadata
dataset_name = "qm9.db"
nb_samples = 1000
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

max_radials = [4,8,12,16,20]
nbs_local_envs = [100] # if LFRE experiments uncomment code below

gfre_hash_values = [] 
lfre_hash_values = [] 
gfre_mat = np.zeros( (len(nbs_local_envs)*len(max_radials), 2) )
lfre_mat = np.zeros( (len(nbs_local_envs)*len(max_radials), 2) )

i = 0
for nb_local_envs in nbs_local_envs:
    for max_radial in max_radials:
        features_hypers1 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "GTO",
                "interaction_cutoff": cutoff,
                "max_radial": max_radial,
                "max_angular": 0,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
        }]

        features_hypers2 = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "GTO",
                "interaction_cutoff": cutoff,
                "max_radial": max_radial+4,
                "max_angular": 0,
                "gaussian_sigma_constant": sigma,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": cutoff_smooth_width,
                "normalize": normalize
            },
        }]

        hash_value, gfre_vec = gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=False, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")
        gfre_hash_values.append(hash_value)
        gfre_mat[i] = gfre_vec.reshape(-1)
        print(hash_value)
        print(gfre_vec)
        print()

        #hash_value, pointwise_lfre = lfre_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, train_ratio, regularizer, one_direction=False, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")
        #lfre_hash_values.append(hash_value)
        #lfre = compute_lfre_from_deprecated_pointwise_lfre(pointwise_lfre )
        #print(lfre)
        #lfre_mat[i] = lfre 
        #print(hash_value)
        #print(lfre_mat[i])
        print()

        i += 1

    print(gfre_hash_values)
    print("GFRE( (n,0) , (n+4,0) )", gfre_mat[:,0])
    print("GFRE( (n+4,0) , (n,0) )", gfre_mat[:,1])
    print()

    #print(lfre_hash_values)
    #print("LFRE( (n,0) , (n+4,0) )", lfre_mat[:,0])
    #print("LFRE( (n+4,0) , (n,0) )", lfre_mat[:,1])
