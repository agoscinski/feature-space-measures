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
# limits memory usage to 700 GB
memory_limit(int(7e+11))

## imitate soap hypers
cutoff = 10 # schnet, gschnet
sigma = 0.2041 # schnet
#sigma = 0.4167 # gschnet
cutoff_function_type = "ShiftedCosine"
cutoff_smooth_width = 10 # schnet
cutoff_function_parameters = None 
#cutoff_smooth_width = 0.001 # gschnet, simulate hard cutoff
normalize = False # there is no schnet parameter for this but we keep it to False

### cosmo paper https://doi.org/10.1039/C8CP05921G hypers
#cutoff = 5
#sigma = 0.3
#cutoff_function_type = "RadialScaling"
#cutoff_smooth_width =  0.5
#cutoff_function_parameters = dict(rate=1, scale=2, exponent=7)
#normalize = False # it is actually True, but we still keep to False, to be consistent with normalization only on training set

# Experiment metadata
dataset_name = "qm9.extxyz"
nb_samples = sys.argv[1] if len(sys.argv) > 1 else 130831
two_split = True
if two_split:
    seed = 0x5f4759df
    seeds = [seed + i for i in range(1)] # seed creator
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

#properties_key = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
#properties_key = ["energy_U0"]
properties_key = ["soap_876b10bc"]
hash_values = []
for key in properties_key:
    features_hypers1 = [{
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "PowerSpectrum",
            "radial_basis": "GTO",
            "interaction_cutoff": cutoff,
            "max_radial": 12,
            "max_angular": 9,
            "gaussian_sigma_constant": sigma,
            "cutoff_function_type": cutoff_function_type,
            "cutoff_smooth_width": cutoff_smooth_width,
            "gaussian_sigma_type": "Constant",
            "normalize": normalize
        },
        "feature_selection_parameters": {
            "type": "PCA",
            #"explained_variance_ratio": 1-1e-6,
            "explained_variance": 1e-9,
        },
        #"hilbert_space_parameters": {
        #    "computation_type": "sparse_implicit_distance",
        #    "distance_parameters": {"distance_type": "euclidean"},
        #    "kernel_parameters": {"kernel_type": "polynomial", "degree": 2, "gamma":1, "c":0}
        #}
    }]
    if cutoff_function_parameters is not None:
        features_hypers1[0]["feature_parameters"]["cutoff_function_parameters"] = cutoff_function_parameters
    features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "feature_name": "schnet",
            "target": "Structure",
            #"filename": "schnet_qm9_"+key+"_nb_structures=130831_layer=6_target=Structure.npy",
            "filename": "schnet_qm9_"+key+"_nb_structures=10000_layer=6_target=Structure.npy",
            "filetype": "npy",
        }
    }]

    hash_value, gfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, set_methane_dataset_to_same_species=False, center_atom_id_mask_description="all environments", target="Structure")
    hash_values.append(hash_value)
    print(hash_value)
    print(gfre_vec)
    print()
print(hash_values)