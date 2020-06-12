#!/usr/bin/env python3
# coding: utf-8
import hashlib
import json

from feature_space_measures import two_split_reconstruction_measure_all_pairs, reconstruction_measure_all_pairs, two_split_reconstruction_measure_pairwise, reconstruction_measure_pairwise
from rascal.representations import SphericalInvariants
import numpy as np
import scipy

import ase.io

import subprocess

DATASET_FOLDER = "data/"
RESULTS_FOLDER = "results/"

# This experiment produces GFR(features_hypers1_i, features_hypers2_i) pairs
def gfr_pairwise_experiment(dataset_name, nb_samples, features_hypers1, features_hypers2, two_split, seed, noise_removal):
    metadata, experiment_id = store_metadata(dataset_name, nb_samples, list(zip(features_hypers1, features_hypers2)), two_split, seed, noise_removal)
    data = read_dataset(dataset_name, nb_samples)
    feature_spaces1 = compute_representations(features_hypers1, data)
    feature_spaces2 = compute_representations(features_hypers2, data)
    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(two_split, seed, noise_removal, feature_spaces1, feature_spaces2)
    store_results("mat-", experiment_id, FRE_matrix, FRD_matrix)

# This experiment produces gfre and gfrd matrices for all pairs from features_hypers
def gfr_all_pairs_experiment(dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal):
    metadata, experiment_id = store_metadata(dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal)
    data = read_dataset(dataset_name, nb_samples)
    feature_spaces = compute_representations(features_hypers, data)
    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(two_split, seed, noise_removal, feature_spaces)
    store_results("mat-", experiment_id, FRE_matrix, FRD_matrix)


def store_metadata(dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal):
    metadata = {
        # All datasets are from CH4
        # datasets "selection-10k.extxyz" -  random-methane
        # datasets "manif-minus.extxyz" + "manif-plus.extxyz" -  degenerated manifold
        # datasets "decomposition.extxyz" -  degenerated manifold
        "dataset": dataset_name,
        # the hypers of targeted features spaces for the experiment
        "features_hypers": features_hypers,
        "two_split": two_split,
        # seed for the random two split if on
        "seed": seed,
        # the number of samples used from the corresponding dataset
        "nb_samples": nb_samples,
        # for FRD one removes the distortion of the samples
        "noise_removal": noise_removal,
        # if something general in the procedure is changed which cannot be captured with the above hyperparameters
        "git_last_commit_id": subprocess.check_output(["git", "describe" ,"--always"]).strip().decode("utf-8"),
        "additional_info": "CH4 environments"
    }

    sha = hashlib.sha1(json.dumps(metadata).encode('utf8')).hexdigest()[:8]
    output_hash = f'{sha}'

    with open(RESULTS_FOLDER+"metadata-"+output_hash + '.json', 'w') as fd:
        json.dump(metadata, fd, indent=2)
    return metadata, output_hash

def read_dataset(dataset_name, nb_samples):
    print("Load data...", flush=True)
    frames = ase.io.read(DATASET_FOLDER+dataset_name, ":"+str(nb_samples))
    for frame in frames:
        frame.cell = np.eye(3) * 15
        frame.center()
        frame.wrap(eps=1e-11)
    print("Load data finished.", flush=True)
    return frames

def compute_representations(features_hypers, frames):
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        representation = SphericalInvariants(**feature_hypers)
        feature_spaces.append(representation.transform(frames).get_features(representation)[::5])
    print("Compute representations finished", flush=True)
    return feature_spaces

def compute_feature_space_reconstruction_measures(two_split, seed, noise_removal, feature_spaces1, feature_spaces2=None):
    print("Compute feature space reconstruction measures...", flush=True)
    if two_split:
        if feature_spaces2 is None:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_all_pairs(feature_spaces1, seed=seed, noise_removal=noise_removal)
        else:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_pairwise(feature_spaces1, feature_spaces2, seed=seed, noise_removal=noise_removal)
    else:
        if feature_spaces2 is None:
            FRE_matrix, FRD_matrix = reconstruction_measure_all_pairs(feature_spaces1, noise_removal=noise_removal)
        else:
            FRE_matrix, FRD_matrix = reconstruction_measure_pairwise(feature_spaces1, feature_spaces2, noise_removal=noise_removal)
    print("Compute feature space reconstruction measures finished.", flush=True)
    return FRE_matrix, FRD_matrix

def store_results(prefix, experiment_id, FRE_matrix, FRD_matrix):
    ### Store experiment results
    print("Store results...")
    np.save(RESULTS_FOLDER+"fre_"+prefix+experiment_id, FRE_matrix)
    np.save(RESULTS_FOLDER+"frd_"+prefix+experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)
