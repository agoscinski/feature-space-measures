
#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_all_pairs_experiment, gfr_pairwise_experiment
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

### Experiment metadata
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.5
else:
    seed = None
    train_ratio = None
noise_removal = False
regularizer = "CV 2 fold"

## Constant hyperparameters
nb_samples = 4000

empty_hash_values = ""
bracket_hash_values = ""

## Tested hyperparameters
cutoffs = [2, 4, 6]
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for radial_basis in ["GTO", "DVR"]:
        features_hypers = [{
            "feature_type": "soap",
            "feature_parameters": {
                "soap_type": "PowerSpectrum",
                "radial_basis": radial_basis,
                "interaction_cutoff": cutoff,
                "max_radial": 10,
                "max_angular": 6,
                "gaussian_sigma_constant": 0.5,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": False
            }
        } for cutoff in cutoffs]

        hash_values = [gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer)]
        empty_hash_values += '"' + ' '.join(hash_values).replace(' ','" "' ) + '" '
        bracket_hash_values += f'cutoff_{radial_basis}_{dataset_name}_hash_value = '+'"' + ' '.join(hash_values).replace(' ','", "' ) + '"\n'

print(empty_hash_values)
print(bracket_hash_values)

