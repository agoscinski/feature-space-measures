#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment
import numpy as np

### Experiment metadata

#dataset_name = "dragged-methane.extxyz"
#dataset_name = "pulled-1H-methane.extxyz"
#dataset_name = "pulled-1H-methane-step_size=0.1.extxyz"
dataset_name = "pulled-1H-methane-step_size=0.05.extxyz"
#dataset_name = "pulled-1H-methane-step_size=0.04.extxyz"
#dataset_name = "pulled-1H-methane-step_size=0.03.extxyz"
#dataset_name = "pulled-1H-methane-step_size=0.02.extxyz"
#dataset_name = "pulled-1H-methane-step_size=0.01.extxyz"
# feature space measures hyperparameteres
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.6
else: 
    train_ratio = None
    seed = None
regularizer = "CV"#3e-2
nb_samples = ""
nb_features = 3
hidden_feature_name = ""

# feature space hyperparameteres
## Constant hyperparameters
cutoff = 4
max_radial = 200
cutoff_smooth_width = 0.5
normalize = False
hash_values = []

## Tested hyperparameters
seeds = [seed +i for i in range(1)] # seed creator
ghfre_mat = np.zeros((len(seeds),9))
i = 0
features_hypers1 = []
for sigma in [0.1,0.3,0.5]:
    features_hypers1.append({
        "feature_type": "soap",
        "feature_parameters": {
            "soap_type": "RadialSpectrum",
            "radial_basis": "DVR",
            "interaction_cutoff": cutoff,
            "max_radial": max_radial,
            "max_angular": 0,
            "gaussian_sigma_constant": sigma,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": cutoff_smooth_width,
            "normalize": normalize
        },
        "hilbert_space_parameters": {
            "distance_parameters": {"distance_type": "euclidean"},
            "kernel_parameters": {"kernel_type": "center"}
        },
        #"feature_selection_parameters": {
        #        "type": "CUR",
        #        "n_features": nb_features,
        #}
    })

    for delta_normalization in [True, False]:
        features_hypers1.append({
            "feature_type": "wasserstein",
            "feature_parameters": {
                "grid_type": "gaussian_quadrature",
                "delta_normalization": delta_normalization,
                "soap_parameters": {
                    "soap_type": "RadialSpectrum",
                    "radial_basis": "DVR",
                    "interaction_cutoff": cutoff,
                    "max_radial": max_radial,
                    "max_angular": 0,
                    "gaussian_sigma_constant": sigma,
                    "gaussian_sigma_type": "Constant",
                    "cutoff_smooth_width": cutoff_smooth_width,
                    "normalize": normalize
                }
            },
            "hilbert_space_parameters": {
                "distance_parameters": {"distance_type": "euclidean"},
                "kernel_parameters": {"kernel_type": "center"}
            },
            #"feature_selection_parameters": {
            #        "type": "CUR",
            #        "n_features": nb_features,
            #}
        })

features_hypers2 = [{
        "feature_type": "precomputed",
        "feature_parameters": {
            "file_root": "data/",
            "dataset": "pulled_hydrogen_distance",
        }} for _ in range(len(features_hypers1))]
for seed in seeds:
    hash_value, ghfre_vec = gfr_pairwise_experiment( dataset_name, nb_samples, features_hypers1, features_hypers2, two_split=two_split, train_ratio=train_ratio, seed=seed, noise_removal=False, regularizer=regularizer, one_direction=True, compute_distortion = False)
    ghfre_mat[i] = ghfre_vec[0]
    hash_values.append(hash_value)
    i += 1
means = np.mean(ghfre_mat,axis=0)
stds = np.std(ghfre_mat,axis=0)
print("sigma E means", means[0::3],"std", stds[0::3])
print("sigma WD means", means[1::3],"std", stds[1::3])
print("sigma WS means", means[2::3],"std", stds[2::3])
print(hash_values)
print('"' + ' '.join(hash_values).replace(' ','" "' ) + '"')
