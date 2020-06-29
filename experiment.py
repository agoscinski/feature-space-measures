#!/usr/bin/env python3
# coding: utf-8
import hashlib
import json

from feature_space_measures import (
    two_split_reconstruction_measure_all_pairs,
    reconstruction_measure_all_pairs,
    two_split_reconstruction_measure_pairwise,
    reconstruction_measure_pairwise,
    compute_local_feature_reconstruction_error_for_pairwise_feature_spaces,
    compute_local_feature_reconstruction_error_for_all_feature_spaces_pairs,
    feature_spaces_hidden_feature_reconstruction_errors
)
from representation import compute_representations
import numpy as np

import ase.io

import subprocess

DATASET_FOLDER = "data/"
RESULTS_FOLDER = "results/"


# This experiment produces GFR(features_hypers1_i, features_hypers2_i) pairs
def gfr_pairwise_experiment(
    dataset_name,
    nb_samples,
    features_hypers1,
    features_hypers2,
    two_split,
    seed,
    noise_removal,
    regularizer
):
    metadata, experiment_id = store_metadata(
        dataset_name,
        nb_samples,
        list(zip(features_hypers1, features_hypers2)),
        two_split,
        seed,
        noise_removal,
        regularizer
    )
    frames = read_dataset(dataset_name, nb_samples)
    feature_spaces1 = compute_representations(features_hypers1, frames)
    feature_spaces2 = compute_representations(features_hypers2, frames)
    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(
        two_split, seed, noise_removal, regularizer, feature_spaces1, feature_spaces2
    )
    print("Store results...", flush=True)
    store_results("gfre_mat-", experiment_id, FRE_matrix)
    store_results("gfrd_mat-", experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

    return {
        'hash': experiment_id,
        'fre': FRE_matrix,
        'frd': FRD_matrix,
    }


# This experiment produces gfre and gfrd matrices for all pairs from features_hypers
def gfr_all_pairs_experiment(
    dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal, regularizer
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal, regularizer
    )
    frames = read_dataset(dataset_name, nb_samples)
    feature_spaces = compute_representations(features_hypers, frames)
    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(
        two_split, seed, noise_removal, regularizer, feature_spaces
    )

    print("Store results...")
    store_results("gfre_mat-", experiment_id, FRE_matrix)
    store_results("gfrd_mat-", experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

def lfre_pairwise_experiment(
    dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, regularizer
):
    metadata, experiment_id = store_metadata(
        dataset_name,
        nb_samples,
        list(zip(features_hypers1, features_hypers2)),
        two_split,
        seed,
        noise_removal=False,
        regularizer=regularizer,
        nb_local_envs = nb_local_envs
    )
    frames = read_dataset(dataset_name, nb_samples)
    feature_spaces1 = compute_representations(features_hypers1, frames)
    feature_spaces2 = compute_representations(features_hypers2, frames)
    print("Compute local feature reconstruction errors...", flush=True)
    lfre_mat = compute_local_feature_reconstruction_error_for_pairwise_feature_spaces(
        feature_spaces1, feature_spaces2, nb_local_envs, two_split, seed, regularizer
    )
    print("Computation local feature reconstruction errors finished", flush=True)

    print("Store results...")
    store_results("lfre_mat-", experiment_id, lfre_mat)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

def lfre_all_pairs_experiment(
    dataset_name, nb_samples, features_hypers, nb_local_envs, two_split, seed, regularizer
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, noise_removal=False, regularizer=regularizer, nb_local_envs=nb_local_envs
    )
    frames = read_dataset(dataset_name, nb_samples)
    feature_spaces = compute_representations(features_hypers, frames)
    lfre_mat = compute_local_feature_reconstruction_error_for_all_feature_spaces_pairs(
        feature_spaces, nb_local_envs, two_split, seed, regularizer
    )

    print("Store results...")
    store_results("lfre_mat-", experiment_id, lfre_mat)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)


def ghfre_experiment(
    dataset_name, nb_samples, features_hypers, two_split, seed, regularizer, hidden_feature_name,
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal=False, regularizer=regularizer, hidden_feature_name=hidden_feature_name)
    frames = read_dataset(dataset_name, nb_samples)
    hidden_feature = np.array([frame.info[hidden_feature_name] for frame in frames])[:,np.newaxis]
    feature_spaces = compute_representations(features_hypers, frames)
    print("Compute hidden feature reconstruction errors...", flush=True)

    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(
        two_split, seed, False, regularizer, feature_spaces, [hidden_feature for _ in range(len(feature_spaces))]
    )

    print("Store results...", flush=True)
    store_results("ghfre_mat-", experiment_id, FRE_matrix)
    store_results("ghfrd_mat-", experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

def hfre_experiment(
    dataset_name, nb_samples, features_hypers, two_split, seed, regularizer, hidden_feature_name,
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal=False, regularizer=regularizer, hidden_feature_name=hidden_feature_name)
    frames = read_dataset(dataset_name, nb_samples)
    hidden_feature = np.array([frame.info[hidden_feature_name] for frame in frames])[:,np.newaxis]
    feature_spaces = compute_representations(features_hypers, frames)
    print("Compute hidden feature reconstruction errors...", flush=True)

    FRE_vector = feature_spaces_hidden_feature_reconstruction_errors(
        feature_spaces, hidden_feature, two_split, seed, regularizer)
    print("Compute hidden feature reconstruction errors finished.", flush=True)
    print("Store results...")
    store_results("hfre_features-", experiment_id, np.array(feature_spaces))
    store_results("hfre_vec-", experiment_id, FRE_vector)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

def store_metadata(
    dataset_name, nb_samples, features_hypers, two_split, seed, noise_removal, regularizer, hidden_feature_name = None, nb_local_envs = None
):
    metadata = {
        # Methane
        # datasets "selection-10k.extxyz" -  random-methane
        # datasets "manif-minus.extxyz" + "manif-plus.extxyz" -  degenerated manifold
        # datasets "dragged_methane.extxyz" - methane with one hydrogen dragged away from the center
        # Carbon
        # datasets "C-VII-pp-wrapped.xyz" -
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
        # regularizer to compute weights for reconstruction weights
        "regularizer": regularizer,
        # if something general in the procedure is changed which cannot be captured with the above hyperparameters
        "git_last_commit_id": subprocess.check_output(["git", "describe", "--always"])
        .strip()
        .decode("utf-8"),
        "additional_info": "CH4 environments",
    }

    # only relevant for hidden feature reconstruction error experiments
    if hidden_feature_name is not None:
        metadata["hidden_feature_name"] = hidden_feature_name
    if nb_local_envs is not None:
        metadata["nb_local_envs"] = nb_local_envs

    sha = hashlib.sha1(json.dumps(metadata).encode("utf8")).hexdigest()[:8]
    output_hash = f"{sha}"

    with open(RESULTS_FOLDER + "metadata-" + output_hash + ".json", "w") as fd:
        json.dump(metadata, fd, indent=2)
    return metadata, output_hash


def read_dataset(dataset_name, nb_samples):
    print("Load data...", flush=True)
    frames = ase.io.read(DATASET_FOLDER + dataset_name, ":" + str(nb_samples))
    for i in range(len(frames)):
        frames[i].cell = np.eye(3) * 15
        frames[i].center()
        frames[i].wrap(eps=1e-11)
    print("Load data finished.", flush=True)
    return frames

def compute_feature_space_reconstruction_measures(
    two_split, seed, noise_removal, regularizer, feature_spaces1, feature_spaces2=None 
):
    print("Compute feature space reconstruction measures...", flush=True)
    if two_split:
        if feature_spaces2 is None:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_all_pairs(
                feature_spaces1, seed=seed, noise_removal=noise_removal, regularizer=regularizer
            )
        else:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_pairwise(
                feature_spaces1, feature_spaces2, seed=seed, noise_removal=noise_removal, regularizer=regularizer
            )
    else:
        if feature_spaces2 is None:
            FRE_matrix, FRD_matrix = reconstruction_measure_all_pairs(
                feature_spaces1, noise_removal=noise_removal, regularizer=regularizer
            )
        else:
            FRE_matrix, FRD_matrix = reconstruction_measure_pairwise(
                feature_spaces1, feature_spaces2, noise_removal=noise_removal, regularizer=regularizer
            )
    print("Compute feature space reconstruction measures finished.", flush=True)
    return FRE_matrix, FRD_matrix


def store_results(prefix, experiment_id, result_np_array):
    # Store experiment results
    np.save(RESULTS_FOLDER + prefix + experiment_id, result_np_array)
