import numpy as np
import scipy

from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
from src.wasserstein import compute_squared_wasserstein_distance, compute_radial_spectrum_wasserstein_features
from src.sorted_distances import compute_sorted_distances
from src.scalers import standardize_features

FEATURES_ROOT ="features"

def compute_representations(features_hypers, frames, train_idx=None, center_atom_id_mask_description="first environment"):
    if center_atom_id_mask_description == "first environment":
        print("WARNING only the first environment of all structures is computed. Please use center_atom_id_mask_description='all environments' if you want to use all environments")
        center_atom_id_mask = [[0] for frame in frames]
    elif center_atom_id_mask_description  == "all environments":
        center_atom_id_mask = [list(range(len(frame))) for frame in frames]
    else:
        raise ValueError("The center_atom_id_mask_description "+center_atom_id_mask_description+" is not available")
    for i in range(len(frames)):
        # masks the atom such that only the representation of the first environment is computed
        mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        if "hilbert_space_parameters" in feature_hypers:
            features = compute_hilbert_space_features(feature_hypers, frames, train_idx, center_atom_id_mask)
        else:
            features = compute_representation(feature_hypers, frames, center_atom_id_mask)
        feature_spaces.append(features)
    print("Compute representations finished", flush=True)
    return feature_spaces


def compute_representation(feature_hypers, frames, center_atom_id_mask):
    if feature_hypers["feature_type"] == "soap":
        representation = SphericalInvariants(**feature_hypers["feature_parameters"])
        return representation.transform(frames).get_features(representation)
    elif feature_hypers["feature_type"] == "wasserstein":
        return compute_radial_spectrum_wasserstein_features(feature_hypers["feature_parameters"], frames)
    elif feature_hypers["feature_type"] == "sorted_distances":
        features = compute_sorted_distances(feature_hypers["feature_parameters"], frames, center_atom_id_mask)
        return features
    elif feature_hypers["feature_type"] == "precomputed":
        parameters = feature_hypers['feature_parameters']
        # TODO parameter case seems to useless, probably can be removed 
        nb_samples = sum([len(structure_mask) for structure_mask in center_atom_id_mask]) if "nb_samples" not in parameters.keys() else parameters["nb_samples"]
        if parameters["filetype"] == "npy":
            pathname = f"{FEATURES_ROOT}/{parameters['feature_name']}/{parameters['filename']}"
            return np.load(pathname)[:nb_samples]
        elif parameters["filetype"] == "txt":
            pathname = f"{FEATURES_ROOT}/{parameters['feature_name']}/{parameters['filename']}"
            return np.loadtxt(pathname)[:nb_samples]
        elif parameters["filetype"] == "frame_info":
            return np.array([frame.info[parameters['feature_name']] for frame in frames])[:,np.newaxis][:nb_samples]
        # TODO is deprecated??
        # hardcoded case
        elif parameters["feature_name"] == "displaced_hydrogen_distance": 
            return load_hydrogen_distance_dataset(frames)[:nb_samples]


def compute_hilbert_space_features(feature_hypers, frames, train_idx, center_atom_id_mask):
    computation_type = feature_hypers["hilbert_space_parameters"]["computation_type"]
    if computation_type == "implicit_distance":
        features = compute_features_from_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames, train_idx, center_atom_id_mask), feature_hypers["hilbert_space_parameters"]["kernel_parameters"]))
    elif computation_type == "explicit":
        features = compute_explicit_features(feature_hypers, frames, train_idx, center_atom_id_mask)
    return features

def compute_explicit_features(feature_hypers, frames, train_idx, center_atom_id_mask):
    features = compute_representation(feature_hypers, frames, center_atom_id_mask)
    kernel_parameters = feature_hypers["hilbert_space_parameters"]["kernel_parameters"]
    kernel_type = kernel_parameters["kernel_type"]
    if kernel_type == "polynomial":
        return compute_explicit_polynomial_features(features, kernel_parameters["degree"])
    else:
        raise ValueError("The kernel_type=" + kernel_type + " is not known.")

def compute_explicit_polynomial_features(features, degree):
    # Based on https://en.wikipedia.org/wiki/Multinomial_theorem#Theorem 
    # degree = n
    # nb_features = m 
    features = np.hstack( (features, np.ones(len(features))[:,np.newaxis]) )
    nb_features = features.shape[1]
    # https://en.wikipedia.org/wiki/Multinomial_theorem#Number_of_multinomial_coefficients
    nb_polynomial_features = int(scipy.special.comb(degree+nb_features-1, nb_features-1))
    polynomial_features = np.zeros( (features.shape[0], nb_polynomial_features) )
    #from sympy.ntheory import multinomial_coefficients
    #from sympy.ntheory.multinomial import multinomial_coefficients
    from sympy.ntheory.multinomial import multinomial_coefficients_iterator
    i = 0
    for k_indices, multinomial_coeff in multinomial_coefficients_iterator(nb_features, degree):
        np.sqrt(multinomial_coeff)
        for t in range(len(k_indices)):
            polynomial_features[:,i] *= features[:,t]**k_indices[t]
        i += 1
    return polynomial_features


def compute_squared_distance(feature_hypers, frames, train_idx, center_atom_id_mask):
    print("Compute distance.")
    distance_type = feature_hypers["hilbert_space_parameters"]["distance_parameters"]["distance_type"]
    if distance_type == "euclidean":
        features = compute_representation(feature_hypers, frames, center_atom_id_mask)
    elif distance_type == "wasserstein":
        raise ValueError("The distance_type='wasserstein' is not fully implemented yet.")
        features = compute_squared_wasserstein_distance(feature_hypers, frames)
    else:
        raise ValueError("The distance_type='" + distance_type + "' is not known.")
    features = standardize_features(features, train_idx)
    # D(A,B)**2 = K(A,A) + K(B,B) - 2*K(A,B)
    return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)


def compute_features_from_kernel(kernel):
    print("Compute features from kernel...")
    # SVD is more numerical stable than eigh
    U, s, _ = scipy.linalg.svd(kernel)
    # reorder eigvals and eigvectors such that largest eigvenvectors and eigvals start in the fist column
    print("Compute features from kernel finished.")
    return np.flip(U, axis=1).dot(np.diag(np.flip(s)))

# use this implementation to check for negative eigenvalues for debugging
# def compute_features_from_kernel(kernel):
#    d, U = scipy.linalg.eigh(kernel)
#    if np.min(d) < 0:
#        print('Negative eigenvalue encounterd',np.min(d))
#        d[d < 0] = 0
#    # flip such that largest eigvals are on the left
#    return np.flip(U, axis=1).dot(np.diag(np.sqrt(np.flip(d))))


def compute_kernel_from_squared_distance(squared_distance, kernel_parameters):
    print("Compute kernel.")
    kernel_type = kernel_parameters["kernel_type"]
    if kernel_type == "center":
        H = np.eye(len(squared_distance)) - np.ones((len(squared_distance), len(squared_distance))) / len(squared_distance)
        return -H.dot(squared_distance).dot(H) / 2
    elif kernel_type == "polynomial":
        H = np.eye(len(squared_distance)) - np.ones((len(squared_distance), len(squared_distance))) / len(squared_distance)
        return (1 + (-H.dot(squared_distance).dot(H) * kernel_parameters["gamma"] ))**kernel_parameters["degree"]
    elif kernel_type == "negative_distance":
        return -squared_distance ** (kernel_parameters["degree"] / 2)
    elif kernel_type == "rbf":
        kernel = np.exp(-kernel_parameters["gamma"] * squared_distance)
        return kernel
    elif kernel_type == "laplacian":
        return np.exp(-kernel_parameters["gamma"] * np.sqrt(squared_distance))
    else:
        raise ValueError("The kernel_type=" + kernel_type + " is not known.")

def load_hydrogen_distance_dataset(frames):
    return np.array([frame.info["hydrogen_distance"] for frame in frames])[:,np.newaxis]
