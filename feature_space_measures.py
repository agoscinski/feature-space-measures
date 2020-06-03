import numpy as np
import scipy


def feature_space_reconstruction_weights(features1, features2, svd_method='gesdd'):
    """
    Computes the minimal weights reconstructing features2 from features1

    Parameters:
    ----------
    features1 (array): feature space X_F as in the paper
    features2 (array): feature space X_{F'} as in the paper

    Returns:
    array : weights P = argmin_{P'} \| X_{F'} - (X_F)P' \|
    """
    return np.linalg.lstsq(features1, features2, rcond=None)[0]


def feature_space_reconstruction_measures(features1, features2, svd_method='gesdd'):
    """
    Computes the FRE and FRD of features2 from features1

    Parameters:
    ----------
    features1 (array): feature space X_F as in the paper
    features2 (array): feature space X_{F'} as in the paper

    Returns:
    array: FRE(X_{F},X_{F'})
    array: FRD(X_{F},X_{F'})
    """

    # P = argmin_{P'} \| X_{F'} - (X_F)P' \|
    P = feature_space_reconstruction_weights(features1, features2, svd_method)

    # (\|X_{F'} - (X_F)P \|) / (\|X_F\|)
    FRE = np.linalg.norm(features1.dot(P)-features2)/np.linalg.norm(features2)

    # P = U S V, we use svd because it is more stable than eigendecomposition
    U, S, V = scipy.linalg.svd(P, lapack_driver=svd_method)

    # Remove noise, TODO this is still an absolute method which needs to be made relative
    S[S < 1e-9] = 0

    # The reconstruction \tilde{X}_{F'} = X_F P = X_F U S V
    # => \tilde{X}_{F'} V.T = X_F U S
    # TODO here I am lacking a bit an intuitive reason why we do this, obviously keeping U S V on the right side will completely ignore the contribution of V when applying it in the procrustes problem, but this is does not explain why putting V.T on the right side is the way to go. Maybe this can be more explored in the paper supplementary

    # X_F U
    features1_U = features1.dot(U)[:, :len(S)]
    # \tilde{X}_{F'} V.T = X_F U S
    reconstructed_features2_VT = features1_U.dot(np.diag(S))

    # Solve procrusets problem see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    U2, S2, V2 = scipy.linalg.svd(features1_U.T.dot(
        reconstructed_features2_VT), lapack_driver=svd_method)
    Q = U2.dot(V2)
    # see paper for derivation of alpha
    alpha = np.trace(features1_U.dot(Q).T.dot(
        reconstructed_features2_VT))/np.trace(features1_U.dot(features1_U.T))
    FRD = np.linalg.norm(alpha*features1_U.dot(Q)-reconstructed_features2_VT) / \
        np.linalg.norm(reconstructed_features2_VT)
    return FRE, FRD


def reconstruction_measure_matrix(feature_spaces):
    """
    Computes the FRE and FRD of features2 from features1

    Parameters:
    ----------
    feature_spaces (list): a list of feature spaces [X_{H_1}, ..., X_{H_n}]

    Returns:
    array: a matrix containing the FRE(X_{H_i},X_{H_j})
    array: a matrix containing the FRD(X_{H_i},X_{H_j})
    """

    FRE_matrix = np.zeros((len(feature_spaces), len(feature_spaces)))
    FRD_matrix = np.zeros((len(feature_spaces), len(feature_spaces)))
    for i in range(len(feature_spaces)):
        for j in range(len(feature_spaces)):
            FRE_matrix[i, j], FRD_matrix[i, j] = feature_space_reconstruction_measures(
                feature_spaces[i], feature_spaces[j])
    return FRE_matrix, FRD_matrix


# Construction side ahead

# TODO tools for computing features from different kernels

def center_features(features):
    H = np.eye(len(features)) - \
        np.ones((len(features), len(features)))/len(features)
    return H.dot(features)


def distance2_from_features(features):
    distmat = np.sum(features**2, axis=1)[:, np.newaxis] + np.sum(
        features**2, axis=1)[np.newaxis, :] - 2*features.dot(features.T)
    distmat[distmat < 0] = 0
    return distmat


def centered_kernel_from_distance2(distance):
    H = np.eye(len(distance)) - \
        np.ones((len(distance), len(distance)))/len(distance)
    return -H.dot(distance).dot(H)/2


def rbf_kernel_from_distance2(distance2, sigma=1):
    return np.exp(-distance2/sigma)


def features_from_kernel(K):
    D, U = scipy.linalg.eigh(K)
    if np.min(D) < 0:
        #print('negative eigenvalue encounterd',np.min(D))
        D[D < 0] = 0
    # flip such that largest eigvals are on the left
    return np.flip(U, axis=1).dot(np.diag(np.sqrt(np.flip(D))))


def features_from_distance2(distance):
    D, U = np.linalg.eigh(-(np.eye(len(distance))-np.ones((len(distance), len(distance)))/len(distance)).dot(
        distance).dot(np.eye(len(distance))-np.ones((len(distance), len(distance)))/len(distance))/2)
    if np.min(D) < 0:
        #print('negative eigenvalue encounterd',np.min(D))
        D[D < 0] = 0
    return np.flip(U, axis=1).dot(np.diag(np.sqrt(np.flip(D))))


def feature2_reconstruction_error(features1, features2):
    W = np.linalg.pinv(features1.T.dot(features1)).dot(
        features1.T.dot(features2))
    features2_reconstructed = features1.dot(W)
    return np.linalg.norm(features2-features2_reconstructed)/np.linalg.norm(features2), W


def feature2_reconstruction_error_vector(features1, features2):
    W = np.linalg.pinv(features1.T.dot(features1)).dot(
        features1.T.dot(features2))
    features2_reconstructed = features1.dot(W)
    return np.linalg.norm(features2-features2_reconstructed, axis=1)/np.linalg.norm(features2, axis=1)

# TODO global and local feature space reconstruction measures


def local_truncated_reconstruction(features1, features2, nb_of_local_env=10):
    distance2 = distance2_from_features(features1)
    reconstruction_error = np.zeros((len(features1), len(features1)))
    for local_env in range(len(features1)):
        local_env_idx = np.random.randint(len(features1), size=nb_of_local_env)
        truncated_features1 = features_from_kernel(centered_kernel_from_distance2(
            distance2_from_features(features1[local_env_idx])))
        truncated_features2 = features_from_kernel(centered_kernel_from_distance2(
            distance2_from_features(features2[local_env_idx])))
        reconstruction_error[local_env] = feature2_reconstruction_error_vector(
            truncated_features1, truncated_features2)[0]
    return reconstruction_error


def local_reconstruction(features1, features2, nb_of_local_env=100):
    distance2 = distance2_from_features(features1)
    reconstruction_error = np.zeros((len(features1), len(features1)))
    for local_env in range(len(features1)):
        #local_env_idx = np.random.randint(len(features1),size=nb_of_local_env)
        local_env_idx = np.argsort(distance2[local_env])[
            :nb_of_local_env][::-1]
        truncated_features1 = features1[local_env_idx]
        truncated_features2 = features2[local_env_idx]
        reconstruction_error[local_env] = feature2_reconstruction_error_vector(
            truncated_features1, truncated_features2)[0]
    return reconstruction_error


def global_truncated_reconstruction(features1, features2):
    distance2 = distance2_from_features(features1)
    truncated_features1 = features_from_kernel(
        centered_kernel_from_distance2(distance2_from_features(features1)))
    truncated_features2 = features_from_kernel(
        centered_kernel_from_distance2(distance2_from_features(features2)))
    return feature2_reconstruction_error_vector(truncated_features1, truncated_features2)


def global_reconstruction(features1, features2):
    distance2 = distance2_from_features(features1)
    truncated_features1 = features1
    truncated_features2 = features2
    return feature2_reconstruction_error_vector(truncated_features1, truncated_features2)
