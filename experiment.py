#!/usr/bin/env python3
# coding: utf-8
import sys
sys.path.insert(0, "/home/alexgo/lib/librascal/build")

import hashlib
import json

from feature_space_measures import two_split_reconstruction_measure_matrix, reconstruction_measure_matrix
from rascal.representations import SphericalInvariants
import numpy as np
import scipy

import ase.io

import subprocess

DATASET_FOLDER = "data/"
RESULTS_FOLDER = "results/"


def do_experiment(dataset_name, nb_samples, features_hypers, two_split, seed):
    metadata, output_hash = store_metadata(dataset_name, nb_samples, features_hypers, two_split, seed)
    data = read_dataset(dataset_name, nb_samples)
    feature_spaces = compute_representations(features_hypers, data)
    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(feature_spaces, two_split, seed)
    store_results(output_hash, FRE_matrix, FRD_matrix)

def store_metadata(dataset_name, nb_samples, features_hypers, two_split, seed):
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
        "noise_removal": False, # option is not yet available, please do not change
        # if something general in the procedure is changed which cannot be captured with the above hyperparameters
        "git_last_commit_id": subprocess.check_output(["git", "describe" ,"--always"]).strip().decode("utf-8"),
        "additional_info": "C-H evironment"
    }

    sha = hashlib.sha1(json.dumps(metadata).encode('utf8')).hexdigest()[:8]
    output_hash = f'{sha}'

    with open(RESULTS_FOLDER+"metadata-"+output_hash + '.json', 'w') as fd:
        json.dump(metadata, fd, indent=2)
    return metadata, output_hash

def read_dataset(dataset_name, nb_samples):
    print("Load data...")
    frames = ase.io.read(DATASET_FOLDER+dataset_name, ":"+str(nb_samples))
    for frame in frames:
        frame.cell = np.eye(3) * 15
        frame.center()
        frame.wrap(eps=1e-11)
    print("Load data finished.")
    return frames

def compute_representations(features_hypers, frames):
    print("Compute representations...")
    feature_spaces = []
    for feature_hypers in features_hypers:
        representation = SphericalInvariants(**feature_hypers)
        feature_spaces.append(representation.transform(frames).get_features(representation)[::5])
    print("Compute representations finished")
    return feature_spaces

def compute_feature_space_reconstruction_measures(feature_spaces, two_split, seed):
    print("Compute feature space reconstruction measures...")
    if two_split:
        FRE_matrix, FRD_matrix = two_split_reconstruction_measure_matrix(feature_spaces, seed=seed)
    else:
        FRE_matrix, FRD_matrix = reconstruction_measure_matrix(feature_spaces)
    print("Compute feature space reconstruction measures finished.")
    return FRE_matrix, FRD_matrix

def store_results(output_hash, FRE_matrix, FRD_matrix):
    ### Store experiment results
    print("Store results...")
    np.save(RESULTS_FOLDER+"fre_mat-"+output_hash, FRE_matrix)
    np.save(RESULTS_FOLDER+"frd_mat-"+output_hash, FRD_matrix)
    print(f"Store results finished. Hash value {output_hash}")
