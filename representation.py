import numpy as np
import scipy

from rascal.representations import SphericalInvariants
from wasserstein import compute_squared_wasserstein_distance, compute_radial_spectrum_wasserstein_features

def compute_representations(features_hypers, frames):
    cumulative_nb_atoms = np.cumsum([frame.get_global_number_of_atoms() for frame in frames])
    first_atom_idx_for_each_frame = cumulative_nb_atoms - frames[0].get_global_number_of_atoms()
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        if "hilbert_space_parameters" in feature_hypers:
            features = compute_hilbert_space_features(feature_hypers, frames, first_atom_idx_for_each_frame)
        else:
            features = compute_representation(feature_hypers, frames, first_atom_idx_for_each_frame)
        feature_spaces.append(features)
    print("Compute representations finished", flush=True)
    return feature_spaces

def compute_representation(feature_hypers, frames, evironment_idx):
    if feature_hypers["feature_type"] == "soap":
        representation = SphericalInvariants(**feature_hypers["feature_parameters"])
        return representation.transform(frames).get_features(representation)[evironment_idx]
    if feature_hypers["feature_type"] == "wasserstein":
        return compute_radial_spectrum_wasserstein_features(feature_hypers["feature_parameters"], frames, evironment_idx)
    else:
        raise ValueError("The feature_type="+feature_hypers["feature_type"]+" is not known.")

def compute_hilbert_space_features(feature_hypers, frames, environment_idx):
    return compute_features_from_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames, environment_idx), feature_hypers["hilbert_space_parameters"]["kernel_parameters"]))

def compute_squared_distance(feature_hypers, frames, environment_idx):
    if feature_hypers["feature_type"] != "soap":
        raise ValueError("Hilbert space features only work with soap features.")
    print("Compute distance.")
    distance_type = feature_hypers["hilbert_space_parameters"]["distance_parameters"]["distance_type"]
    if distance_type == "euclidean":
        representation = SphericalInvariants(**feature_hypers)
        features = representation.transform(frames).get_features(representation)[environment_idx]
        # D(A,B)**2 = K(A,A) + K(B,B) - 2*K(A,B)
        return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)
    elif distance_type == "wasserstein":
        return compute_squared_wasserstein_distance(feature_hypers, frames, environment_idx)
    else:
        raise ValueError("The distance_type="+distance_type+" is not known.")

def compute_features_from_kernel(kernel):
    print("Compute features from kernel...")
    # SVD is more numerical stable than eigh
    U, s, _ = scipy.linalg.svd(kernel)
    # reorder eigvals and eigvectors such that largest eigvenvectors and eigvals start in the fist column
    print("Compute features from kernel finished.")
    return np.flip(U, axis=1).dot(np.diag(np.flip(s)))

# use this implementation to check for negative eigenvalues for debugging
#def compute_features_from_kernel(kernel):
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
        return (1 + kernel_parameters["gamma"] * (-H.dot(squared_distance).dot(H) / 2))**kernel_parameters["degree"]
    elif kernel_type == "negative_distance":
        return -squared_distance ** (kernel_parameters["degree"]/2)
    elif kernel_type == "rbf":
        return np.exp(-kernel_parameters["gamma"]*squared_distance)
    elif kernel_type == "laplacian":
        return np.exp(-kernel_parameters["gamma"]*np.sqrt(squared_distance))
    else:
        raise ValueError("The kernel_type="+kernel_type +" is not known.")
