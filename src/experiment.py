#!/usr/bin/env python3
# coding: utf-8
import hashlib
import json

from src.feature_space_measures import (
   two_split_reconstruction_measure_all_pairs,
    reconstruction_measure_all_pairs,
    two_split_reconstruction_measure_pairwise,
    reconstruction_measure_pairwise,
    compute_local_feature_reconstruction_error_for_pairwise_feature_spaces,
    compute_local_feature_reconstruction_error_for_all_feature_spaces_pairs,
    feature_spaces_hidden_feature_reconstruction_errors,
    train_test_gfrm_pairwise
)

from src.scalers import NormalizeScaler
from src.select_features import select_features

from src.representation import compute_representations
import numpy as np

import ase.io

import subprocess

DATASET_FOLDER = "data/"
RESULTS_FOLDER = "results/"

def generate_two_split_idx(nb_samples, train_ratio=0.5, seed=0x5F3759DF, frames=None, center_atom_id_mask_description="all environments"):
    """
    Computes the FRE and FRD of features2 from features1 with a two-split

    Parameters:
    ----------
    features1 (array): feature space X_F as in the paper, samples x features
    features2 (array): feature space X_{F'} as in the paper, samples x features

    Returns:
    --------
    double: FRE(X_{F},X_{F'}) scalar value
    double: FRD(X_{F},X_{F'}) scalar value
    """
    if type(train_ratio) == str:
        train_ratio_strings = train_ratio.split(" ")
        if train_ratio_strings[0] == "equispaced":
            shift = int(train_ratio_strings[1])
            idx = np.arange(nb_samples)
            #print(idx[::shift].shape)
            return idx[::shift], idx
        else:
            raise ValueError(f"train_ration {train_ratio} is not known.")
    else:
        if (frames is None):
            np.random.seed(seed)
            idx = np.arange(nb_samples)
            np.random.shuffle(idx)
            split_id = int(len(idx) * train_ratio)
            return idx[:split_id], idx[split_id:]
        # choose structures instead of environments
        else:
            np.random.seed(seed)
            struc_idx = np.arange(len(frames))
            np.random.shuffle(struc_idx)
            split_id = int(len(struc_idx) * train_ratio)
            train_struc_idx, test_struc_idx = struc_idx[:split_id], struc_idx[split_id:]
            if (center_atom_id_mask_description == "first environment"):
                train_env_idx = np.copy(train_struc_idx) 
                test_env_idx = np.copy(test_struc_idx)
                train_test_struc_idx = {'train':train_struc_idx, 'test':test_struc_idx}
            elif (center_atom_id_mask_description == "all environments"):
                cumulative_env_idx = np.hstack( (0, np.cumsum([len(frame) for frame in frames])) )
                train_env_idx = np.concatenate([np.arange(len(frames[idx])) + cumulative_env_idx[idx] for idx in train_struc_idx])
                test_env_idx = np.concatenate([np.arange(len(frames[idx])) + cumulative_env_idx[idx] for idx in test_struc_idx])
                train_test_struc_idx = {'train':train_struc_idx, 'test':test_struc_idx}
            else:
                raise ValueError(f"center_atom_id_mask_description  {center_atom_id_mask_description } is not known.")
            return train_env_idx, test_env_idx, train_test_struc_idx

def standardize_features(features, train_idx=None):
    if train_idx is None:
        return NormalizeScaler().fit(features).transform(features)
    return NormalizeScaler().fit(features[train_idx]).transform(features)

def split_features(features, train_idx, test_idx):
    if train_idx is None:
        return (features, features)
    return features[train_idx], features[test_idx]

def postprocess_features(features, feature_hypers, train_idx, test_idx=None):
    standardized_features = standardize_features(features, train_idx)
    standardization_error = np.linalg.norm(standardized_features-features)/len(features) 
    print("Standardization error:",standardization_error)
    features = standardized_features
    if "feature_selection_parameters" in feature_hypers:
        features = select_features(features, features[train_idx], feature_hypers["feature_selection_parameters"])
    return (features[train_idx], features[test_idx])

# TODO rename nb_samples to nb_structures as input, to rename
# This experiment produces GFR(features_hypers1_i, features_hypers2_i) pairs
def gfr_pairwise_experiment(
    dataset_name,
    nb_samples,
    features_hypers1,
    features_hypers2,
    two_split,
    train_ratio,
    seed,
    noise_removal,
    regularizer,
    one_direction=False,
    compute_distortion=True,
    train_test_gfrm=False,
    set_methane_dataset_to_same_species=True,
    center_atom_id_mask_description="first environment"
):
    metadata, experiment_id = store_metadata(
        dataset_name,
        nb_samples,
        list(zip(features_hypers1, features_hypers2)),
        two_split,
        train_ratio,
        seed,
        noise_removal,
        regularizer,
        one_direction=one_direction,
        compute_distortion=compute_distortion,
        train_test_gfrm=train_test_gfrm,
        center_atom_id_mask_description=center_atom_id_mask_description
    )
    frames = read_dataset(dataset_name, nb_samples, set_methane_dataset_to_same_species)

    if center_atom_id_mask_description == "first environment":
        nb_samples_ = len(frames)
    elif center_atom_id_mask_description == "all environments":
        nb_samples_ = sum([frame.get_global_number_of_atoms() for frame in frames])


    train_idx, test_idx, train_test_structures_idx = generate_two_split_idx(nb_samples_, train_ratio, seed, frames, center_atom_id_mask_description )
    print("np.max(train_idx)", np.max(train_idx))
    print("np.max(test_idx)", np.max(test_idx))

    feature_spaces1 = compute_representations(features_hypers1, frames, train_idx, center_atom_id_mask_description, train_test_structures_idx)
    print("feature_spaces1[0].shape",feature_spaces1[0].shape)
    feature_spaces2 = compute_representations(features_hypers2, frames, train_idx, center_atom_id_mask_description, train_test_structures_idx)

    for i in range(len(feature_spaces1)):
        feature_spaces1[i] = postprocess_features(feature_spaces1[i], features_hypers1[i], train_idx, test_idx)
        feature_spaces2[i] = postprocess_features(feature_spaces2[i], features_hypers2[i], train_idx, test_idx)

    print("Compute feature space reconstruction measures...", flush=True)
    if train_test_gfrm:
        FRE_train_matrix, FRE_test_matrix, FRD_matrix = train_test_gfrm_pairwise(feature_spaces1, feature_spaces2, noise_removal=noise_removal, regularizer=regularizer, one_direction=one_direction, compute_distortion=compute_distortion)
        print("Compute feature space ßeconstruction measures finished.", flush=True)

        print("Store results...", flush=True)
        store_results("gfre_train_mat-", experiment_id, FRE_train_matrix)
        store_results("gfre_test_mat-", experiment_id, FRE_test_matrix)
        store_results("gfrd_train_test_mat-", experiment_id, FRD_matrix)
        print(f"Store results finished. Hash value {experiment_id}", flush=True)
        return experiment_id, FRE_train_matrix
    else:
        FRE_matrix, FRD_matrix = two_split_reconstruction_measure_pairwise(
                    feature_spaces1, feature_spaces2, noise_removal=noise_removal, regularizer=regularizer, one_direction=one_direction, compute_distortion=compute_distortion)
        print("Compute feature space ßeconstruction measures finished.", flush=True)

        print("Store results...", flush=True)
        store_results("gfre_mat-", experiment_id, FRE_matrix)
        store_results("gfrd_mat-", experiment_id, FRD_matrix)
        print(f"Store results finished. Hash value {experiment_id}", flush=True)
        return experiment_id, FRE_matrix


# This experiment produces gfre and gfrd matrices for all pairs from features_hypers
def gfr_all_pairs_experiment(
    dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer, compute_distortion=True
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer
    )
    frames = read_dataset(dataset_name, nb_samples)
    train_idx, test_idx = generate_two_split_idx(nb_samples, train_ratio, seed)
    feature_spaces = compute_representations(features_hypers, frames, train_idx)
    for i in range(len(feature_spaces)):
        feature_spaces[i] = postprocess_features(feature_spaces[i], features_hypers[i], train_idx, test_idx)
    FRE_matrix, FRD_matrix = two_split_reconstruction_measure_all_pairs(
                    feature_spaces,  noise_removal=noise_removal, regularizer=regularizer, compute_distortion=compute_distortion)


    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(
        two_split, train_ratio, seed, noise_removal, regularizer, feature_spaces
    )

    print("Store results...")
    store_results("gfre_mat-", experiment_id, FRE_matrix)
    store_results("gfrd_mat-", experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)
    return experiment_id

def lfre_pairwise_experiment(
        dataset_name, nb_samples, features_hypers1, features_hypers2, nb_local_envs, two_split, seed, train_ratio, regularizer, inner_epsilon=None, outer_epsilon=None, one_direction=False, set_methane_dataset_to_same_species=True, center_atom_id_mask_description="first environment"):

    metadata, experiment_id = store_metadata(
        dataset_name,
        nb_samples,
        list(zip(features_hypers1, features_hypers2)),
        two_split,
        train_ratio,
        seed,
        noise_removal=False,
        regularizer=regularizer,
        nb_local_envs = nb_local_envs,
        inner_epsilon=inner_epsilon,
        outer_epsilon=outer_epsilon,
        one_direction=one_direction,
        center_atom_id_mask_description=center_atom_id_mask_description
    )


    frames = read_dataset(dataset_name, nb_samples, set_methane_dataset_to_same_species)
    feature_spaces1 = compute_representations(features_hypers1, frames, center_atom_id_mask_description=center_atom_id_mask_description)
    feature_spaces2 = compute_representations(features_hypers2, frames, center_atom_id_mask_description=center_atom_id_mask_description)

    for i in range(len(feature_spaces1)):
        feature_spaces1[i] = postprocess_features(feature_spaces1[i], features_hypers1[i], np.arange(feature_spaces1[i].shape[0]), [])[0]
        feature_spaces2[i] = postprocess_features(feature_spaces2[i], features_hypers2[i], np.arange(feature_spaces2[i].shape[0]), [])[0]
    print("Compute local feature reconstruction errors...", flush=True)

    lfre_mat, lfrd_mat = compute_local_feature_reconstruction_error_for_pairwise_feature_spaces(
        feature_spaces1, feature_spaces2, nb_local_envs, two_split, train_ratio, seed, regularizer, inner_epsilon, outer_epsilon, one_direction
    )
    print("Computation local feature reconstruction errors finished", flush=True)

    print("Store results...")
    store_results("lfre_mat-", experiment_id, lfre_mat)
    store_results("lfrd_mat-", experiment_id, lfrd_mat)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)
    return experiment_id, lfre_mat

def lfre_all_pairs_experiment(
    dataset_name, nb_samples, features_hypers, nb_local_envs, two_split, train_ratio, seed, regularizer
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal=False, regularizer=regularizer, nb_local_envs=nb_local_envs
    )
    frames = read_dataset(dataset_name, nb_samples)
    feature_spaces = compute_representations(features_hypers, frames)
    lfre_mat, lfrd_mat = compute_local_feature_reconstruction_error_for_all_feature_spaces_pairs(
        feature_spaces, nb_local_envs, two_split, train_ratio, seed, regularizer
    )

    print("Store results...")
    store_results("lfre_mat-", experiment_id, lfre_mat)
    store_results("lfrd_mat-", experiment_id, lfrd_mat)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)


def ghfre_experiment(
    dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, regularizer, hidden_feature_name,
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal=False, regularizer=regularizer, hidden_feature_name=hidden_feature_name)
    frames = read_dataset(dataset_name, nb_samples)
    hidden_feature = np.array([frame.info[hidden_feature_name] for frame in frames])[:,np.newaxis]
    feature_spaces1 = compute_representations(features_hypers, frames)

    train_idx, test_idx = generate_two_split_idx(nb_samples, train_ratio, seed)

    for i in range(len(feature_spaces1)):
        feature_spaces1[i] = postprocess_features(feature_spaces1[i], features_hypers[i], train_idx, test_idx)
        feature_spaces2[i] = postprocess_features(feature_spaces2[i], features_hypers[i], train_idx, test_idx)

    print("Compute feature space reconstruction measures...", flush=True)
    FRE_matrix, FRD_matrix = two_split_reconstruction_measure_pairwise(
                feature_spaces1, feature_spaces2, noise_removal=False, regularizer=regularizer )
    print("Compute feature space reconstruction measures finished.", flush=True)
    

    print("Compute hidden feature reconstruction errors...", flush=True)

    FRE_matrix, FRD_matrix = compute_feature_space_reconstruction_measures(
        two_split, train_ratio, seed, False, regularizer, feature_spaces1, [hidden_feature for _ in range(len(feature_spaces1))]
    )

    print("Store results...", flush=True)
    store_results("ghfre_mat-", experiment_id, FRE_matrix)
    store_results("ghfrd_mat-", experiment_id, FRD_matrix)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)
    return experiment_id, FRE_matrix

def hfre_experiment(
    dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, regularizer, hidden_feature_name,
):
    metadata, experiment_id = store_metadata(
        dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal=False, regularizer=regularizer, hidden_feature_name=hidden_feature_name)
    frames = read_dataset(dataset_name, nb_samples)
    hidden_feature = np.array([frame.info[hidden_feature_name] for frame in frames])[:,np.newaxis]
    feature_spaces = compute_representations(features_hypers, frames)
    print("Compute hidden feature reconstruction errors...", flush=True)

    FRE_vector = feature_spaces_hidden_feature_reconstruction_errors(
        feature_spaces, hidden_feature, two_split, train_ratio, seed, regularizer)
    print("Compute hidden feature reconstruction errors finished.", flush=True)
    print("Store results...")
    store_results("hfre_features-", experiment_id, np.array(feature_spaces))
    store_results("hfre_vec-", experiment_id, FRE_vector)
    print(f"Store results finished. Hash value {experiment_id}", flush=True)

def store_metadata(
    dataset_name, nb_samples, features_hypers, two_split, train_ratio, seed, noise_removal, regularizer, hidden_feature_name = None, nb_local_envs = None, inner_epsilon=None, outer_epsilon=None, one_direction=None, compute_distortion=None, train_test_gfrm=None, center_atom_id_mask_description=None
):
    metadata = {
        # Methane
        # datasets `selection-10k.extxyz` -  random methane
        # datasets `manif-minus-plus.extxyz` -  degenerate manifold
        # datasets `displaced-methane-step_size=PLACE_HOLDER-range=PLACE_HOLDER-seed=PLACE_HOLDER.extxyz` - displaced methane
        # Carbon structures
        # datasets "C-VII-pp-wrapped.xyz" - carbon dataset
        "dataset": dataset_name,
        # the hypers of targeted features spaces for the experiment
        "features_hypers": features_hypers,
        "two_split": two_split,
        "train_ratio": train_ratio,
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
        "additional_info": "H5 environments, no lfrd computed, truncation->extension",
    }

    # only relevant for hidden feature reconstruction error experiments
    if hidden_feature_name is not None:
        metadata["hidden_feature_name"] = hidden_feature_name
    if nb_local_envs is not None:
        metadata["nb_local_envs"] = nb_local_envs
    if inner_epsilon is not None:
        metadata["inner_epsilon"] = inner_epsilon
    if outer_epsilon is not None:
        metadata["outer_epsilon"] = outer_epsilon
    if one_direction is not None:
        metadata["one_direction"] = one_direction
    if compute_distortion is not None:
        metadata["compute_distortion"] = compute_distortion
    if train_test_gfrm is not None:
        metadata["train_test_gfrm"] = train_test_gfrm 
    if center_atom_id_mask_description is not None:
        metadata["center_atom_id_mask_description "] = "first environment"
        


    sha = hashlib.sha1(json.dumps(metadata).encode("utf8")).hexdigest()[:8]
    output_hash = f"{sha}"

    with open(RESULTS_FOLDER + "metadata-" + output_hash + ".json", "w") as fd:
        json.dump(metadata, fd, indent=2)
    return metadata, output_hash


def read_dataset(dataset_name, nb_samples, set_methane_dataset_to_same_species=True):
    print("Load data...", flush=True)
    frames = ase.io.read(DATASET_FOLDER + dataset_name, ":" + str(nb_samples))
    if dataset_name == "C-VII-pp-wrapped.xyz":
        for i in range(len(frames)):
            frames[i].pbc=True
            frames[i].wrap(eps=1e-11)
    elif dataset_name == "random-ch4-10k.extxyz" or dataset_name == "selection-10k.extxyz" or dataset_name == 'qm9.db':
        for i in range(len(frames)):
            frames[i].cell = np.eye(3) * 20
            frames[i].center()
            frames[i].wrap(eps=1e-11)
            if (set_methane_dataset_to_same_species):
                frames[i].numbers = np.ones(len(frames[i]))
    print("Load data finished.", flush=True)
    return frames

def compute_feature_space_reconstruction_measures(
    two_split, train_ratio, seed, noise_removal, regularizer, feature_spaces1, feature_spaces2=None 
):
    print("Compute feature space reconstruction measures...", flush=True)
    if two_split:
        if feature_spaces2 is None:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_all_pairs(
                feature_spaces1, train_ratio=train_ratio, seed=seed, noise_removal=noise_removal, regularizer=regularizer
            )
        else:
            FRE_matrix, FRD_matrix = two_split_reconstruction_measure_pairwise(
                feature_spaces1, feature_spaces2, train_ratio=train_ratio, seed=seed, noise_removal=noise_removal, regularizer=regularizer
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
