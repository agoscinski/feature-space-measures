#!/usr/bin/env python3
# coding: utf-8
from experiment import do_experiment

### Experiment metadata

dataset_name = "selection-10k.extxyz"
nb_samples = 10000

## Constant hyperparameters
# cutoff was determined by
# In [7]: frames = ase.io.read("data/selection-10k.extxyz", ':'); np.max([np.max(frame.get_all_distances()) for frame in frames]) #
# Out[7]: 5.936250715807484
cutoff = 6
normalize = False

## Tested hyperparameters
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5]
features_hypers = [{
    "soap_type": "PowerSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": 10,
    "max_angular": 6,
    "gaussian_sigma_constant": sigma,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
} for sigma in sigmas]

two_split = True
if two_split:
    seed = 0x5f3759df
else:
    seed = None

do_experiment(dataset_name, nb_samples, features_hypers, two_split, seed)
