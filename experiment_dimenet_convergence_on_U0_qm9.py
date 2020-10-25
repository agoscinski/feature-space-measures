#!/usr/bin/env python3
# coding: utf-8
import numpy as np
from src.experiment import gfr_pairwise_experiment
import os, sys

import resource
def memory_limit(nb_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (nb_bytes, hard))
# limits memory usage to 800 GB
memory_limit(8e+11)

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
dataset_name = "qm9.db"
nb_samples = sys.argv[1] if len(sys.argv) > 1 else 1000
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

layers = [0,1,2,3,4,5]

hash_values = [] 
gfre_mat = np.zeros( (len(layers), 2) )
i = 0
for layer in layers:
    features_hypers1 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "dimenet",
            "filename": "dimenet_qm9_U0_nb_structures=10000_layer="+str(layer)+".npy",
            "filetype": "npy",
        }
    }]
    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "dimenet",
            "filename": "dimenet_qm9_U0_nb_structures=10000_layer=6.npy",
            "filetype": "npy",
        }
    }]

    hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=True, one_direction=True, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")
    hash_values.append(hash_value)
    gfre_mat[i] = gfre_vec.reshape(-1)
    print(hash_value)
    print(gfre_mat[i])
    print()
    i += 1

print(hash_values)
print("GFRE( dimenet layer ",layers, ", dimenet layer 6) = ",gfre_mat[:,0])
print("GFRE( dimenet layer 6, dimenet layer ",layers, " ) = ",gfre_mat[:,0])
