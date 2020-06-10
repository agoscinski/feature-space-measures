#!/usr/bin/env python3
# coding: utf-8
from experiment import do_experiment

### Experiment metadata

dataset_name = "selection-10k.extxyz"

# cutoff was determined by
# In [7]: frames = ase.io.read("data/selection-10k.extxyz", ':'); np.max([np.max(frame.get_all_distances()) for frame in frames]) #
# Out[7]: 5.936250715807484
cutoff = 6
max_radial = 7
max_angular = 5
normalize = False

hypers_rs_dvr = {
    "soap_type": "RadialSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_rs_gto = {
    "soap_type": "RadialSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_ps_dvr = {
    "soap_type": "PowerSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_ps_gto = {
    "soap_type": "PowerSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_bs_dvr = {
    "soap_type": "BiSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_bs_gto = {
    "soap_type": "BiSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}




### Experiment

nb_samples = 10000
features_hypers = [hypers_rs_dvr, hypers_rs_gto, hypers_ps_dvr, hypers_ps_gto]#, hypers_bs_dvr, hypers_bs_gto]
two_split = False
if two_split:
    seed = 0x5f3759df
else:
    seed = None

do_experiment(dataset_name, nb_samples, features_hypers, two_split, seed)
