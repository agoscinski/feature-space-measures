#!/usr/bin/env python3
# coding: utf-8
from src.experiment import gfr_pairwise_experiment, gfr_all_pairs_experiment
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

# Experiment metadata
dataset_name = "qm9.db"
nb_frames = sys.argv[1] if len(sys.argv) > 1 else 500
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

properties_key = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
hash_values = []
features_hypers = [{
    "feature_type": "precomputed",
    "feature_parameters": {
        "feature_name": "schnet",
        "filename": "schnet_"+key+"_U0_nb_structures=500_layer=6.npy",
        "filetype": "npy",
    }
} for key in properties_key]

hash_value, gfre_vec = gfr_all_pairs_experiment( dataset_name, nb_frames, features_hypers, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments")
hash_values.append(hash_value)
print(hash_value)
print(gfre_vec)
print()
