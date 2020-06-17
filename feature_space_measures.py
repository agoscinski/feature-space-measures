import numpy as np
import scipy
from scalers import NormalizeScaler


def feature_space_reconstruction_weights(features1, features2, svd_method="gesdd"):
    """
    Computes the minimal weights reconstructing features2 from features1

    Parameters:
    ----------
    features1 (array): feature space X_F as in the paper, samples x features
    features2 (array): feature space X_{F'} as in the paper, samples x features

    Returns:
    --------
    array : weights P = argmin_{P'} | X_{F'} - (X_F)P' |
    """
    return np.linalg.lstsq(features1, features2, rcond=None)[0]

def feature_space_reconstruction_measures(
    features1,
    features2,
    reconstruction_weights=None,
    svd_method="gesdd",
    noise_removal=False,
):
    """
    Computes the FRE and FRD of features2 from features1

    Parameters:
    ----------
    features1 (array): feature space X_F as in the paper, samples x features
    features2 (array): feature space X_{F'} as in the paper, samples x features
    reconstruction_weights (array):  weights defined by P = argmin_{P'} || X_{F'} - (X_F)P' ||

    Returns:
    --------
    double: FRE(X_{F},X_{F'}) scalar value
    double: FRD(X_{F},X_{F'}) scalar value
    """
    if reconstruction_weights is None:
        features1 = NormalizeScaler().fit(features1).transform(features1)
        features2 = NormalizeScaler().fit(features2).transform(features2)
        reconstruction_weights = feature_space_reconstruction_weights(
            features1, features2
        )
    # (\|X_{F'} - (X_F)P \|) / (\|X_F\|)
    FRE = np.linalg.norm(
        features1.dot(reconstruction_weights) - features2
    )  # / np.linalg.norm(features2)

    # P = U S V, we use svd because it is more stable than eigendecomposition
    U, S, V = scipy.linalg.svd(reconstruction_weights, lapack_driver=svd_method)

    if noise_removal:
        S[S < 1e-9] = 0

    # The reconstruction \tilde{X}_{F'} = X_F P = X_F U S V
    # => \tilde{X}_{F'} V.T = X_F U S
    # TODO here I am lacking a bit an intuitive reason why we do this, obviously keeping U S V on the right side will completely ignore the contribution of V when applying it in the procrustes problem, but this is does not explain why putting V.T on the right side is the way to go. Maybe this can be more explored in the paper supplementary

    # X_F U
    features1_U = features1.dot(U)[:, :len(S)]
    # \tilde{X}_{F'} V.T = X_F U S
    reconstructed_features2_VT = features1_U.dot(np.diag(S))

    # Solve procrustes problem see https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    U2, S2, V2 = scipy.linalg.svd(
        features1_U.T.dot(reconstructed_features2_VT),
        lapack_driver=svd_method
    )
    Q = U2.dot(V2)
    # see paper for derivation of alpha
    alpha = np.trace(features1_U.dot(Q).T.dot(reconstructed_features2_VT)) / np.trace(
        features1_U.dot(features1_U.T)
    )
    FRD = np.linalg.norm(alpha * features1_U.dot(Q) - reconstructed_features2_VT)
    FRD /= np.linalg.norm(reconstructed_features2_VT)
    return FRE, FRD


def two_split_feature_space_reconstruction_measures(
    features1, features2, svd_method="gesdd", seed=0x5F3759DF, noise_removal=False
):
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
    np.random.seed(seed)
    idx = np.arange(len(features1))
    np.random.shuffle(idx)
    split_id = int(len(idx) / 2)

    features1 = NormalizeScaler().fit(features1[idx[:split_id]]).transform(features1)
    features1_train = features1[idx[:split_id]]
    features1_test = features1[idx[split_id:]]

    features2 = NormalizeScaler().fit(features2[idx[:split_id]]).transform(features2)
    features2_train = features2[idx[:split_id]]
    features2_test = features2[idx[split_id:]]

    reconstruction_weights = feature_space_reconstruction_weights(
        features1_train, features2_train, svd_method
    )
    return feature_space_reconstruction_measures(
        features1_test,
        features2_test,
        reconstruction_weights,
        svd_method,
        noise_removal,
    )


def two_split_reconstruction_measure_all_pairs(
    feature_spaces, svd_method="gesdd", seed=0x5F3759DF, noise_removal=False
):
    """
    Computes the FRE and FRD of features2 from features1

    Parameters:
    ----------
    feature_spaces (list): a list of feature spaces [X_{H_1}, ..., X_{H_n}]

    Returns:
    --------
    array: a matrix containing the FRE(X_{H_i},X_{H_j})
    array: a matrix containing the FRD(X_{H_i},X_{H_j})
    """

    FRE_matrix = np.zeros((len(feature_spaces), len(feature_spaces)))
    FRD_matrix = np.zeros((len(feature_spaces), len(feature_spaces)))
    for i in range(len(feature_spaces)):
        for j in range(len(feature_spaces)):
            FRE_matrix[i, j], FRD_matrix[
                i, j
            ] = two_split_feature_space_reconstruction_measures(
                feature_spaces[i], feature_spaces[j], svd_method, seed, noise_removal
            )
    return FRE_matrix, FRD_matrix


def reconstruction_measure_all_pairs(
    feature_spaces, svd_method="gesdd", noise_removal=False
):
    """
    Computes the FRE and FRD for all feature_spaces pairs

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
                feature_spaces[i],
                feature_spaces[j],
                svd_method=svd_method,
                noise_removal=noise_removal,
            )
    return FRE_matrix, FRD_matrix


def two_split_reconstruction_measure_pairwise(
    feature_spaces1,
    feature_spaces2,
    svd_method="gesdd",
    seed=0x5F3759DF,
    noise_removal=False,
):
    """
    Computes the FRE and FRD of (feature_spaces1[i], feature_spaces2[i]) and (feature_spaces2[i], feature_spaces1[i]) pairs

    Parameters:
    ----------

    Returns:
    --------
    array: a (2, len(feature_spaces1)) matrix containing the FRE(X_{H_i},Y_{H_i})
    array: a (2, len(feature_spaces1)) matrix containing the FRD(X_{H_i},Y_{H_i})
    """
    assert len(feature_spaces1) == len(feature_spaces2)
    FRE_matrix = np.zeros((2, len(feature_spaces1)))
    FRD_matrix = np.zeros((2, len(feature_spaces1)))
    for i in range(len(feature_spaces1)):
        FRE_matrix[0, i], FRD_matrix[0, i] = two_split_feature_space_reconstruction_measures(
            feature_spaces1[i], feature_spaces2[i], svd_method, seed, noise_removal
        )
        FRE_matrix[1, i], FRD_matrix[1, i] = two_split_feature_space_reconstruction_measures(
            feature_spaces2[i], feature_spaces1[i], svd_method, seed, noise_removal
        )
    return FRE_matrix, FRD_matrix


def reconstruction_measure_pairwise(
    feature_spaces1, feature_spaces2, svd_method="gesdd", noise_removal=False
):
    assert len(feature_spaces1) == len(feature_spaces2)
    FRE_matrix = np.zeros((2, len(feature_spaces1)))
    FRD_matrix = np.zeros((2, len(feature_spaces1)))
    for i in range(len(feature_spaces1)):
        FRE_matrix[0, i], FRD_matrix[
            0, i
        ] = two_split_feature_space_reconstruction_measures(
            feature_spaces1[i], feature_spaces2[i], svd_method, noise_removal
        )
        FRE_matrix[1, i], FRD_matrix[
            1, i
        ] = two_split_feature_space_reconstruction_measures(
            feature_spaces2[i], feature_spaces1[i], svd_method, noise_removal
        )
    return FRE_matrix, FRD_matrix

def hidden_feature_reconstruction_errors(
    features, hidden_feature
):
    FRE_vector = np.zeros(features.shape[1])
    for i in range (features.shape[1]): # nb features
        features = NormalizeScaler().fit(features).transform(features)
        hidden_feature = NormalizeScaler().fit(hidden_feature).transform(hidden_feature)
        reconstruction_weights = feature_space_reconstruction_weights(
                features[:,i][:,np.newaxis], hidden_feature
        )
        # (\|X_{F'} - (X_F)P \|) / (\|X_F\|)
        FRE_vector[i] = np.linalg.norm(
                features[:,i][:,np.newaxis].dot(reconstruction_weights) - hidden_feature
        )  # / np.linalg.norm(features2)
    return FRE_vector

def feature_spaces_hidden_feature_reconstruction_errors(
    feature_spaces, hidden_feature, noise_removal=False
):
    # we assert that only feature_spaces with the same number of features are used to simplify storage
    for i in range(len(feature_spaces)):
        assert( feature_spaces[i].shape[1] == feature_spaces[0].shape[1] )
    # for each feature space a fre vector exist, computing the error for each feature
    FRE_vectors = np.zeros((len(feature_spaces), feature_spaces[0].shape[1]))
    for i in range(len(feature_spaces)):
        FRE_vectors[i] = hidden_feature_reconstruction_errors(
           feature_spaces[i], hidden_feature
        )
    return FRE_vectors


# ----- Construction side ahead
# TODO tools for computing features from different kernels

# TODO global and local feature space reconstruction measures


def local_truncated_reconstruction(features1, features2, nb_of_local_env=10):
    distance2 = distance2_from_features(features1)
    reconstruction_error = np.zeros((len(features1), len(features1)))
    for local_env in range(len(features1)):
        local_env_idx = np.random.randint(len(features1), size=nb_of_local_env)
        truncated_features1 = features_from_kernel(
            centered_kernel_from_distance2(
                distance2_from_features(features1[local_env_idx])
            )
        )
        truncated_features2 = features_from_kernel(
            centered_kernel_from_distance2(
                distance2_from_features(features2[local_env_idx])
            )
        )
        reconstruction_error[local_env] = feature2_reconstruction_error_vector(
            truncated_features1, truncated_features2
        )[0]
    return reconstruction_error


def local_reconstruction(features1, features2, nb_of_local_env=100):
    distance2 = distance2_from_features(features1)
    reconstruction_error = np.zeros((len(features1), len(features1)))
    for local_env in range(len(features1)):
        # local_env_idx = np.random.randint(len(features1),size=nb_of_local_env)
        local_env_idx = np.argsort(distance2[local_env])[:nb_of_local_env][::-1]
        truncated_features1 = features1[local_env_idx]
        truncated_features2 = features2[local_env_idx]
        reconstruction_error[local_env] = feature2_reconstruction_error_vector(
            truncated_features1, truncated_features2
        )[0]
    return reconstruction_error
