#!/usr/bin/env python3
# coding: utf-8
from experiment import gfr_pairwise_experiment
import numpy as np
import os

os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

# Experiment metadata
nb_samples = 10000
two_split = True
if two_split:
    seed = 0x5f3759df
    train_ratio = 0.6
else:
    seed = None
    train_ratio = None
regularizer = "CV"
# Constant hyperparameters
sigma = 0.5
cutoff_smooth_width = 0.5
normalize = False

# 10, 30, 50, 100, 150, 200
for dataset_name in ["selection-10k.extxyz", "C-VII-pp-wrapped.xyz"]:
    for feature_select_type in ["CUR"]:
        for feature_count in [100,200,300,400]:
            if dataset_name == "selection-10k.extxyz":
                cutoff = 4
                precomputed_name = 'methane'
                #BP_sizes = [35, 191, 534, 1147]
                # max_radials_angulars sizes are 36, 192, 540, 1152
                #max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5)]

                BP_sizes = [1147]
                max_radials_angulars = [(8, 5)]
            elif dataset_name == "C-VII-pp-wrapped.xyz":
                cutoff = 7.5
                precomputed_name = 'carbon'
                BP_sizes = [11, 61, 181, 377, 699]
                # max_radials_angulars sizes are 12, 64, 180, 384, 700
                max_radials_angulars = [(2, 2), (4, 3), (6, 4), (8, 5), (10, 6)]
                BP_sizes = [699]
                max_radials_angulars = [(10, 6)]


            features_hypers1 = [{
                "feature_type": "soap",
                "feature_parameters": {
                    "soap_type": "PowerSpectrum",
                    "radial_basis": "GTO",
                    "interaction_cutoff": cutoff,
                    "max_radial": max_radial,
                    "max_angular": max_angular,
                    "gaussian_sigma_constant": sigma,
                    "gaussian_sigma_type": "Constant",
                    "cutoff_smooth_width": cutoff_smooth_width,
                    "normalize": normalize
                },
                "feature_selection_parameters": {
                    "type": feature_select_type,
                    "n_features": feature_count,
                }
            } for max_radial, max_angular in max_radials_angulars]

            features_hypers2 = [{
                "feature_type": "precomputed_BP",
                "feature_parameters": {
                    "file_root": "data/Alex",
                    "dataset": precomputed_name,
                    "SF_count": count,
                },
                "feature_selection_parameters": {
                    "type": feature_select_type,
                    "n_features": feature_count,
                }
            } for count in BP_sizes]

            data = gfr_pairwise_experiment(
                dataset_name,
                nb_samples,
                features_hypers1,
                features_hypers2,
                two_split=two_split,
                train_ratio=train_ratio,
                seed=seed,
                noise_removal=False,
                regularizer=regularizer
            )

            fre = np.zeros((3, len(BP_sizes)))
            fre[:2, :] = data['fre']
            fre[2, :] = BP_sizes
            np.savetxt(
                f"results/gfre-{data['hash']}.dat",
                fre.T,
                header="# GFRE(SOAP -> BP)   GFRE(BP -> SOAP)   BP_size",
            )

            frd = np.zeros((3, len(BP_sizes)))
            frd[:2, :] = data['frd']
            frd[2, :] = BP_sizes
            np.savetxt(
                f"results/gfrd-{data['hash']}.dat",
                frd.T,
                header="# GFRD(SOAP -> BP)   GFRD(BP -> SOAP)   BP_size",
            )
