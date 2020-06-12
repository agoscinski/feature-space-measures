#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_all_pairs_experiment

### Experiment metadata

dataset_name = "selection-10k.extxyz"
nb_samples = 10000

## Constant hyperparameters
# cutoff was determined by
# In [7]: frames = ase.io.read("data/selection-10k.extxyz", ':'); np.max([np.max(frame.get_all_distances()) for frame in frames]) #
# Out[7]: 5.936250715807484
cutoff = 6
max_radial = 12
max_angular = 7
normalize = False

## Tested hyperparameters
radial_bases = ["DVR", "GTO"]
soap_types = ["RadialSpectrum", "PowerSpectrum", "BiSpectrum"]
features_hypers = [{
    "soap_type": "RadialSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
} for radial_basis in radial_bases for soap_type in soap_types]

two_split = True
if two_split:
    seed = 0x5f3759df
else:
    seed = None

noise_removal = False

gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal)
