import numpy as np
import scipy

from rascal.representations import SphericalInvariants
from wasserstein import compute_squared_wasserstein_distance

# example
#feature_hypers = {
#    "soap_type": "PowerSpectrum",
#    "radial_basis": radial_basis,
#    "interaction_cutoff": cutoff,
#    "max_radial": 10,
#    "max_angular": 6,
#    "gaussian_sigma_constant": sigma,
#    "gaussian_sigma_type": "Constant",
#    "cutoff_smooth_width": cutoff_smooth_width,
#    "normalize": False,
#    "cutoff_function_type": "RadialScaling",
#    # if kernel type is given then the 
#    "hilbert_space_parameters": {
#        "distance_type": "euclidean"/"wasserstein",
#        rbf: "kernel_parameters": {"kernel_type": "rbf"/"center"/"polynomial", "sigma": 1},
#        polynomial: "kernel_parameters": {"degree": 1},
#    }
#}

def compute_representations(features_hypers, frames):
    cumulative_nb_atoms = np.cumsum([frame.get_global_number_of_atoms() for frame in frames])
    first_atom_idx_for_each_frame = cumulative_nb_atoms - frames[0].get_global_number_of_atoms()
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        if "hilbert_space_parameters" in feature_hypers:
            features = compute_hilbert_space_features(feature_hypers, frames)
        else:
            representation = SphericalInvariants(**feature_hypers)
            features = representation.transform(frames).get_features(representation)[first_atom_idx_for_each_frame]
        feature_spaces.append(features)
    print("Compute representations finished", flush=True)
    return feature_spaces

def compute_hilbert_space_features(feature_hypers):
    return compute_features_from_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames), feature_hypers["kernel_parameters"]))

def compute_squared_distance(feature_hypers, frames):
    if distance_type == "euclidean":
        representation = SphericalInvariants(**feature_hypers)
        features = representation.transform(frames).get_features(representation)[first_atom_idx_for_each_frame]
        # D(A,B)**2 = K(A,A) + K(B,B) - 2*K(A,B)
        return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)
    elif distance_type == "wasserstein":
        return compute_wasserstein_distance(feature_hypers)
    else:
        raise ValueError("The distance_type="+distance_type+" is not known.")

def compute_features_from_kernel(kernel):
    # SVD is more numerical stable than eigh
    U, s, _ = scipy.linalg.svd(kernel)
    # reorder eigvals and eigvectors such that largest eigvenvectors and eigvals start in the fist column
    return np.flip(U, axis=1).dot(np.diag(np.flip(s)))

# use this implementation to check for negative eigenvalues for debugging
#def compute_features_from_kernel(kernel):
#    d, U = scipy.linalg.eigh(kernel)
#    if np.min(d) < 0:
#        print('Negative eigenvalue encounterd',np.min(d))
#        d[d < 0] = 0
#    # flip such that largest eigvals are on the left
#    return np.flip(U, axis=1).dot(np.diag(np.sqrt(np.flip(d))))

def compute_kernel(squared_distance, kernel_parameters):
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
