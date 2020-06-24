import numpy as np
import scipy

from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
from wasserstein import compute_squared_wasserstein_distance, compute_radial_spectrum_wasserstein_features


def compute_representations(features_hypers, frames, center_atom_id_mask=None):
    if center_atom_id_mask is None:
        center_atom_id_mask = [[0] for frame in frames]
    for i in range(len(frames)):
        # masks the atom such that only the representation of the first environment is computed
        mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        if "hilbert_space_parameters" in feature_hypers:
            features = compute_hilbert_space_features(feature_hypers, frames)
        else:
            features = compute_representation(feature_hypers, frames, center_atom_id_mask)
        feature_spaces.append(features)
    if "feature_selection_parameters" in feature_hypers:
        raise ValueError("The feature selection methods are not implemented yet.")
        for feature_space in feature_spaces:
            # TODO Guillaume: apply feature selection method on all features in list feature_spaces
            print(feature_hypers["feature_selection_parameters"]["nb_features"])
    print("Compute representations finished", flush=True)
    return feature_spaces


def compute_representation(feature_hypers, frames, center_atom_id_mask):
    if feature_hypers["feature_type"] == "soap":
        representation = SphericalInvariants(**feature_hypers["feature_parameters"])
        return representation.transform(frames).get_features(representation)
    elif feature_hypers["feature_type"] == "wasserstein":
        return compute_radial_spectrum_wasserstein_features(feature_hypers["feature_parameters"], frames)
    elif feature_hypers["feature_type"] == "precomputed_BP":
        nb_envs = sum([len(structure_mask) for structure_mask in center_atom_id_mask])

        parameters = feature_hypers["feature_parameters"]
        path = f"{parameters['file_root']}/{parameters['dataset']}/compute_features/{parameters['dataset']}_{parameters['SF_count']}SF"
        data = np.zeros((nb_envs, parameters["SF_count"]))

        if parameters['dataset'] == "methane":
            # only C-centered envs are included in the BF feature, so we have
            # env per frame
            with open(path) as fd:
                for i, line in enumerate(fd):
                    # only the first 4000 frames are used
                    if i >= nb_envs:
                        break
                    data[i, :] = list(map(float, line.split()))
                assert(i == data.shape[0])
        elif parameters['dataset'] == "carbon":
            # all envs are included in BP, but SOAP only compute the env for the
            # first atom of the frame, so we need to skip some of them
            count = 0
            included_envs = []
            for frame in frames:
                included_envs.append(count)
                count += len(frame)

            with open(path) as fd:
                i = 0
                for line_i, line in enumerate(fd):
                    if line_i in included_envs:
                        data[i, :] = list(map(float, line.split()))
                        i += 1
                    if i >= nb_envs:
                        break

                assert(i == data.shape[0])
        else:
            raise ValueError("unknown dataset " + parameters['dataset'])
        return data
    else:
        raise ValueError("The feature_type=" + feature_hypers["feature_type"] + " is not known.")


def compute_hilbert_space_features(feature_hypers, frames):
    return compute_features_from_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames), feature_hypers["hilbert_space_parameters"]["kernel_parameters"]))


def compute_squared_distance(feature_hypers, frames):
    if feature_hypers["feature_type"] != "soap":
        raise ValueError("Hilbert space features only work with soap features.")
    print("Compute distance.")
    distance_type = feature_hypers["hilbert_space_parameters"]["distance_parameters"]["distance_type"]
    if distance_type == "euclidean":
        representation = SphericalInvariants(**feature_hypers["feature_parameters"])
        features = representation.transform(frames).get_features(representation)
        # D(A,B)**2 = K(A,A) + K(B,B) - 2*K(A,B)
        return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)
    elif distance_type == "wasserstein":
        return compute_squared_wasserstein_distance(feature_hypers, frames)
    else:
        raise ValueError("The distance_type=" + distance_type + " is not known.")


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
        return (1 + kernel_parameters["gamma"] * (-H.dot(squared_distance).dot(H) / 2))**kernel_parameters["degree"]
    elif kernel_type == "negative_distance":
        return -squared_distance ** (kernel_parameters["degree"] / 2)
    elif kernel_type == "rbf":
        kernel = np.exp(-kernel_parameters["gamma"] * 1/np.mean(squared_distance) * squared_distance)
        plt.imshow(kernel)
        plt.colorbar()
        plt.show()
        return kernel
    elif kernel_type == "laplacian":
        return np.exp(-kernel_parameters["gamma"] * 1/np.mean(squared_distance) * np.sqrt(squared_distance))
    else:
        raise ValueError("The kernel_type=" + kernel_type + " is not known.")
