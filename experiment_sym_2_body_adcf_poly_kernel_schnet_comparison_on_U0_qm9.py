#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment
import numpy as np
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

# Constant hyperparameters
cutoff = 10 # schnet, gschnet

sigma = 0.2041 # schnet
cutoff_smooth_width = 10 # schnet
#sigma = 0.4167 # gschnet
#cutoff_smooth_width = 0.001 # gschnet, simulate hard cutoff

normalize = False


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

polynomial_degrees = [1,2,3,4,5,6,7]

hash_values = [] 
gfre_mat = np.zeros( (len(polynomial_degrees), 2) )
i = 0
for degree in polynomial_degrees:
    features_hypers1 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "RadialSpectrum",
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": 12,
            "max_angular": 0,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
            "feature_selection_parameters": {
                "type": "PCA",
                "explained_variance_ratio": 0.99,
                #"n_features": 128,
            },
        "hilbert_space_parameters": {
            "computation_type": "explicit",
            "distance_parameters": {"distance_type": "euclidean"},
            "kernel_parameters": {"kernel_type": "polynomial", "degree": degree, "gamma":1}
        }
    }]
    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "schnet",
            "filename": "schnet_qm9_energy_U0_nb_structures=10000_layer=6.npy",
            "filetype": "npy",
        }
    }]

    hash_value, gfre_vec = gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=True, one_direction=False, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")
    hash_values.append(hash_value)
    gfre_mat[i] = gfre_vec.reshape(-1)
    print(hash_value)
    print(gfre_vec)
    print()
    i += 1

print(hash_values)
print("GFRE( 2-body poly degree [1,2,3,4,5,6,7], schnet interaction layer [0,1,2,3,4,5,6] )", gfre_mat[:,0])
print("GFRE( schnet interaction layer [0,1,2,3,4,5,6],  2-body poly degree [1,2,3,4,5,6,7] )", gfre_mat[:,1])
