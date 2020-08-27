#!/usr/bin/env python3
# coding: utf-8
from experiment import lfre_pairwise_experiment, gfr_pairwise_experiment
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

### Experiment metada

#dataset_name = "manif-minus-plus.extxyz"
#nb_samples = 162
#nbs_local_envs = [10,15]

#dataset_name = "C-VII-pp-wrapped.xyz"
#nb_samples = 1000
#nbs_local_envs = [125]

dataset_name = "selection-10k.extxyz"
nb_samples = 4000

nbs_local_envs = [10,50,100,200,300,500]

inner_epsilon = None#1e-5
outer_epsilon = None#1e-1

## Constant hyperparameters
radial_basis = "GTO"
cutoff = 4
max_radial = 6
max_angular = 4
sigma = 0.5
normalize = False

noise_removal=False
two_split = True
if two_split:
    seed = 0x5f4759df
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
regularizer = "CV 2 fold"

experiment_ids = []

def execute_experiment(features_hypers1, features_hypers2):
    experiment_id, gfre_vec =  gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=False, one_direction=True)
    experiment_ids.append(experiment_id)
    print(gfre_vec)


## RS -> PS, BS
## RS poly 2 -> PS,
## RS poly 3 -> BS
#max_radials = [2,4,6,8,10,12,14,16,18]
#max_radials = [10,20,30,40,50,60]
#features_hypers1 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#features_hypers2 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "PowerSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#execute_experiment(features_hypers1, features_hypers2)
#
#features_hypers1 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#features_hypers2 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "BiSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#execute_experiment(features_hypers1, features_hypers2)
#
#features_hypers1 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    },
#    "hilbert_space_parameters": {
#        "distance_parameters": {"distance_type": "euclidean"},
#        "kernel_parameters": {"kernel_type": "polynomial", "gamma": 1, "degree":2}
#    }} for max_radial in max_radials]
#
#features_hypers2 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "PowerSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#execute_experiment(features_hypers1, features_hypers2)
#
#features_hypers1 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    },
#    "hilbert_space_parameters": {
#        "distance_parameters": {"distance_type": "euclidean"},
#        "kernel_parameters": {"kernel_type": "polynomial", "gamma": 1, "degree":3}
#    }} for max_radial in max_radials]
#
#features_hypers2 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "BiSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": 0,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }} for max_radial in max_radials]
#
#execute_experiment(features_hypers1, features_hypers2)


#features_hypers1 = [{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": max_angular,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    }
#    },{
#   "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": max_angular,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    },
#    "hilbert_space_parameters": {
#        "distance_parameters": {"distance_type": "euclidean"},
#        "kernel_parameters": {"kernel_type": "rbf", "gamma": 0.1}
#    },
#    },{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": max_angular,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    },
#    "hilbert_space_parameters": {
#        "distance_parameters": {"distance_type": "euclidean"},
#        "kernel_parameters": {"kernel_type": "rbf", "gamma": 1}
#    }
#    },{
#    "feature_type": "soap",
#    "feature_parameters": {
#        "soap_type": "RadialSpectrum",
#        "radial_basis": radial_basis,
#        "interaction_cutoff": cutoff,
#        "max_radial": max_radial,
#        "max_angular": max_angular,
#        "gaussian_sigma_constant": sigma,
#        "gaussian_sigma_type": "Constant",
#        "cutoff_smooth_width": 0.5,
#        "normalize": normalize
#    },
#    "hilbert_space_parameters": {
#        "distance_parameters": {"distance_type": "euclidean"},
#        "kernel_parameters": {"kernel_type": "rbf", "gamma": 10}
#    }
#}]



# Tested hyperparameters
features_hypers1 = [{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "RadialSpectrum",
        "radial_basis": radial_basis,
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": 0.5,
        "normalize": normalize
    }
    #},{
    #"feature_type": "soap",
    #"feature_parameters": {
    #    "soap_type": "RadialSpectrum",
    #    "radial_basis": radial_basis,
    #    "interaction_cutoff": cutoff,
    #    "max_radial": max_radial,
    #    "max_angular": max_angular,
    #    "gaussian_sigma_constant": sigma,
    #    "gaussian_sigma_type": "Constant",
    #    "cutoff_smooth_width": 0.5,
    #    "normalize": normalize
    #},
    #"hilbert_space_parameters": {
    #    "distance_parameters": {"distance_type": "euclidean"},
    #    "kernel_parameters": {"kernel_type": "polynomial", "degree": 2, "gamma": 1}
    #},
    #},{
    #"feature_type": "soap",
    #"feature_parameters": {
    #    "soap_type": "RadialSpectrum",
    #    "radial_basis": radial_basis,
    #    "interaction_cutoff": cutoff,
    #    "max_radial": max_radial,
    #    "max_angular": max_angular,
    #    "gaussian_sigma_constant": sigma,
    #    "gaussian_sigma_type": "Constant",
    #    "cutoff_smooth_width": 0.5,
    #    "normalize": normalize
    #},
    #"hilbert_space_parameters": {
    #    "distance_parameters": {"distance_type": "euclidean"},
    #    "kernel_parameters": {"kernel_type": "polynomial", "degree": 3, "gamma": 1}
    #}

}]

features_hypers2 = [{
    "feature_type": "soap",
    "feature_parameters": {
        "soap_type": "PowerSpectrum",
        "radial_basis": radial_basis,
        "interaction_cutoff": cutoff,
        "max_radial": max_radial,
        "max_angular": max_angular,
        "gaussian_sigma_constant": sigma,
        "gaussian_sigma_type": "Constant",
        "cutoff_smooth_width": 0.5,
        "normalize": normalize
    }
    #},{
    #"feature_type": "soap",
    #"feature_parameters": {
    #    "soap_type": "PowerSpectrum",
    #    "radial_basis": radial_basis,
    #    "interaction_cutoff": cutoff,
    #    "max_radial": max_radial,
    #    "max_angular": max_angular,
    #    "gaussian_sigma_constant": sigma,
    #    "gaussian_sigma_type": "Constant",
    #    "cutoff_smooth_width": 0.5,
    #    "normalize": normalize
    #}
    #},{
    #"feature_type": "soap",
    #"feature_parameters": {
    #    "soap_type": "PowerSpectrum",
    #    "radial_basis": radial_basis,
    #    "interaction_cutoff": cutoff,
    #    "max_radial": max_radial,
    #    "max_angular": max_angular,
    #    "gaussian_sigma_constant": sigma,
    #    "gaussian_sigma_type": "Constant",
    #    "cutoff_smooth_width": 0.5,
    #    "normalize": normalize
    #}
    #},{
    #"feature_type": "soap",
    #"feature_parameters": {
    #    "soap_type": "PowerSpectrum",
    #    "radial_basis": radial_basis,
    #    "interaction_cutoff": cutoff,
    #    "max_radial": max_radial,
    #    "max_angular": max_angular,
    #    "gaussian_sigma_constant": sigma,
    #    "gaussian_sigma_type": "Constant",
    #    "cutoff_smooth_width": 0.5,
    #    "normalize": normalize
    #}
}]


for nb_local_envs in nbs_local_envs:
    hash_value = lfre_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, train_ratio, regularizer, inner_epsilon, outer_epsilon, one_direction=False)
    experiment_ids.append(hash_value)
    print(f"nb_local_envs={nb_local_envs} hash_value={hash_value}")
print(experiment_ids)
hash_value, _ = gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, compute_distortion=False)
print(f"GFRE={hash_value}")
