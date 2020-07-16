import numpy as np
import scipy
from scipy.spatial.distance import cdist
from scalers import NormalizeScaler
from sklearn import linear_model
import warnings


def global_embed_cv(x1, x2):
    # features1 = x1
    # features2 = x1
    n = len(x1)
    # two-way split of the dataset
    x1a = x1[:n//2]; x1b = x1[n//2:]
    x2a = x2[:n//2]; x2b = x2[n//2:]
    # svd of the two halves
    k = min(x1a.shape[1], x2a.shape[1])
    ua, sa, vha = np.linalg.svd(x1a, full_matrices = False)
    ub, sb, vhb = np.linalg.svd(x1b, full_matrices = False)
    # computes intermediates
    uat_x2a = ua.T @ x2a
    ubt_x2b = ub.T @ x2b
    x1b_va = x1b @ vha.T
    x1a_vb = x1a @ vhb.T
    def thresh_cv_loss(lthr):
        thr = np.exp(lthr)
        na = len(np.where(sa>thr)[0])
        nb = len(np.where(sb>thr)[0])
        # error approximating x2b a-fitted model and vice versa
        loss_ab = (( (x1b_va[:,:na]*(1/sa[:na]))@uat_x2a[:na] - x2b)**2).sum()
        loss_ba = (( (x1a_vb[:,:nb]*(1/sb[:nb]))@ubt_x2b[:nb] - x2a)**2).sum()
        return loss_ab+loss_ba
    res = scipy.optimize.minimize_scalar(thresh_cv_loss, method='Bounded', bounds=[-50,0])
    return np.exp(res.x)*2/(sa[0]+sb[0])



def feature_space_reconstruction_weights(features1, features2, regularizer=1e-6):
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
    

    ##print("Computing weights...")
    #W = np.zeros((features1.shape[1],features2.shape[1]))
    #for i in range( features2.shape[1] ):
    #    if i % int(features2.shape[1]/5) == 0:
    #        print("weight step "+str(i)+"")
    #    reg = linear_model.BayesianRidge(alpha_1=regularizer, alpha_2=regularizer, lambda_1=regularizer, lambda_2=regularizer)
    #    reg.fit(features1, features2[:,i])
    #    W[:,i] = reg.coef_
    ##print("Computing weights finished.")
    #return W
    #regs = [linear_model.BayesianRidge(alpha_1=regularizer, alpha_2=regularizer, lambda_1=regularizer, lambda_2=regularizer) for i in range(features2.shape[1])]
    #[regs[i].fit(features1, features2[:,i]) for i in range(features2.shape[1])]
    #return np.array( [regs[i].coef_ for i in range(features2.shape[1])] ).T
    if regularizer == "CV":
        regularizer = global_embed_cv(features1, features2)
        print(regularizer)
    W = np.linalg.lstsq(features1, features2, rcond=regularizer)[0]
    if np.linalg.norm(W) > 1e4:
        warnings.warn("Reconstruction weight matrix very large "+ str(np.linalg.norm(W)) +". Results could be misleading.", Warning)
    return W

def standardize_features(features, train_idx=None):
    if train_idx is None:
        return NormalizeScaler().fit(features).transform(features)
    return NormalizeScaler().fit(features[train_idx]).transform(features)

def feature_space_reconstruction_measures(
    features1,
    features2,
    svd_method="gesdd",
    noise_removal=False,
    reconstruction_weights=None,
    regularizer=np.nan,
    n_test=None
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
        if regularizer is np.nan:
            raise ValueError("If no reconstruction weights is given a regularizer has to be given.")
        features1 = standardize_features(features1)
        features2 = standardize_features(features2)
        reconstruction_weights = feature_space_reconstruction_weights(
            features1, features2, regularizer
        )
    if n_test is None:
        n_test = features2.shape[0]

    # (\|X_{F'} - (X_F)P \|) / (\|X_F\|)
    FRE = np.linalg.norm(
        features1.dot(reconstruction_weights) - features2
    ) / np.sqrt(n_test)

    # P = U S V, we use svd because it is more stable than eigendecomposition
    U, S, V = scipy.linalg.svd(reconstruction_weights, lapack_driver="gesvd")

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
    # alpha is not important anymore because features are standardized 
    ## see paper for derivation of alpha
    #alpha = np.trace(features1_U.dot(Q).T.dot(reconstructed_features2_VT)) / np.trace(
    #    features1_U.dot(features1_U.T)
    #)
    alpha = 1
    FRD = np.linalg.norm(alpha * features1_U.dot(Q) - reconstructed_features2_VT) / np.sqrt(n_test)
    return FRE, FRD


def generate_two_split_idx(nb_samples, train_ratio=0.5, seed=0x5F3759DF):
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
    idx = np.arange(nb_samples)
    np.random.shuffle(idx)
    split_id = int(len(idx) * train_ratio)
    return idx[:split_id], idx[split_id:]

def split_in_two(features1, features2, train_idx, test_idx):
    features1_train = features1[train_idx]
    features1_test = features1[test_idx]

    features2_train = features2[train_idx]
    features2_test = features2[test_idx]

    return features1_train, features2_train, features1_test, features2_test

def two_split_reconstruction_measure_all_pairs(
    feature_spaces, svd_method="gesdd", train_ratio=0.5, seed=0x5F3759DF, noise_removal=False, regularizer=np.nan
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
    nb_samples = len(feature_spaces[0])
    train_idx, test_idx = generate_two_split_idx(nb_samples, train_ratio, seed)
    for i in range(len(feature_spaces)):
        for j in range(len(feature_spaces)):
            features1 = standardize_features(feature_spaces[i], train_idx)
            features2 = standardize_features(feature_spaces[j], train_idx)
            features1_train, features2_train, features1_test, features2_test = split_in_two(
                    features1, features2, train_idx, test_idx)
            reconstruction_weights = feature_space_reconstruction_weights(
                features1_train, features2_train, regularizer
            )
            FRE_matrix[i, j], FRD_matrix[
                i, j
            ] = feature_space_reconstruction_measures(
                    features1_test,
                    features2_test,
                    reconstruction_weights=reconstruction_weights,
                    regularizer=regularizer,
                    svd_method=svd_method,
                    noise_removal=noise_removal,
                )
    return FRE_matrix, FRD_matrix


def reconstruction_measure_all_pairs(
    feature_spaces, svd_method="gesdd", noise_removal=False, regularizer=np.nan
):
    """
    Computes the FRE and FRD for all feature_spaces pairs

 a   Parameters:
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
                regularizer=regularizer
            )
    return FRE_matrix, FRD_matrix


def two_split_reconstruction_measure_pairwise(
    feature_spaces1,
    feature_spaces2,
    svd_method="gesdd",
    train_ratio=0.5,
    seed=0x5F3759DF,
    noise_removal=False,
    regularizer=np.nan,
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


    nb_samples = len(feature_spaces1[0])
    train_idx, test_idx = generate_two_split_idx(nb_samples, train_ratio, seed)

    for i in range(len(feature_spaces1)):
        features1 = standardize_features(feature_spaces1[i], train_idx)
        features2 = standardize_features(feature_spaces2[i], train_idx)
        features1_train, features2_train, features1_test, features2_test = split_in_two(
                features1, features2, train_idx, test_idx)
        reconstruction_weights = feature_space_reconstruction_weights(
            features1_train, features2_train, regularizer
        )
        FRE_matrix[0, i], FRD_matrix[0, i] = feature_space_reconstruction_measures(
            features1_test, features2_test, reconstruction_weights=reconstruction_weights,
            svd_method=svd_method, noise_removal=noise_removal
        )
        reconstruction_weights = feature_space_reconstruction_weights(
            features2_train, features1_train, regularizer
        )
        FRE_matrix[1, i], FRD_matrix[1, i] = feature_space_reconstruction_measures(
            features2_test, features1_test, reconstruction_weights=reconstruction_weights,
            svd_method=svd_method, noise_removal=noise_removal
        )

    return FRE_matrix, FRD_matrix


def reconstruction_measure_pairwise(
    feature_spaces1, feature_spaces2, svd_method="gesdd", noise_removal=False, regularizer=np.nan
):
    assert len(feature_spaces1) == len(feature_spaces2)
    FRE_matrix = np.zeros((2, len(feature_spaces1)))
    FRD_matrix = np.zeros((2, len(feature_spaces1)))
    for i in range(len(feature_spaces1)):
        FRE_matrix[0, i], FRD_matrix[
            0, i
        ] = feature_space_reconstruction_measures(
            feature_spaces1[i], feature_spaces2[i], svd_method=svd_method, noise_removal=noise_removal, regularizer=regularizer
        )
        FRE_matrix[1, i], FRD_matrix[
            1, i
        ] = feature_space_reconstruction_measures(
            feature_spaces2[i], feature_spaces1[i], svd_method=svd_method, noise_removal=noise_removal, regularizer=regularizer
        )
    return FRE_matrix, FRD_matrix

def hidden_feature_reconstruction_errors(
    features_train, hidden_feature_train, features_test=None, hidden_feature_test=None, regularizer=np.nan):
    if features_test is None:
        features_test = features_train
    if hidden_feature_test is None:
        hidden_feature_test = hidden_feature_train
    n_test = features_train.shape[0]
    FRE_vector = np.zeros(features_train.shape[1])
    for i in range (features_train.shape[1]): # nb features
        reconstruction_weights = feature_space_reconstruction_weights(
                features_train[:,i][:,np.newaxis], hidden_feature_train, regularizer
        )
        # (\|X_{F'} - (X_F)P \|) / (\|X_F\|)
        FRE_vector[i] = np.linalg.norm(
                features_test[:,i][:,np.newaxis].dot(reconstruction_weights) - hidden_feature_test
        )  / np.sqrt(n_test)
    return FRE_vector

def feature_spaces_hidden_feature_reconstruction_errors(
    feature_spaces, hidden_feature, two_split=False, train_ratio=None, seed=None, regularizer=np.nan):
    # we assert that only feature_spaces with the same number of features are used to simplify storage
    for i in range(len(feature_spaces)):
        assert( feature_spaces[i].shape[1] == feature_spaces[0].shape[1] )
    # for each feature space a fre vector exist, computing the error for each feature
    FRE_vectors = np.zeros((len(feature_spaces), feature_spaces[0].shape[1]))
    if two_split:
        # generate idx beforehand
        nb_samples = len(feature_spaces[0])
        train_idx, test_idx = generate_two_split_idx(nb_samples, train_ratio, seed)
    else:
        train_idx, test_idx = (None, None)
    for i in range(len(feature_spaces)):
        features = feature_spaces[i]
        features = standardize_features(features, train_idx)
        hidden_feature = standardize_features(hidden_feature, train_idx)
        if two_split:
            features_train, hidden_feature_train, features_test, hidden_feature_test = split_in_two(
                    features, hidden_feature, train_idx, test_idx)
            FRE_vectors[i] = hidden_feature_reconstruction_errors(
               features_train, hidden_feature_train, features_test, hidden_feature_test, regularizer=regularizer
            )
        else:
            FRE_vectors[i] = hidden_feature_reconstruction_errors(
               features, hidden_feature, regularizer=regularizer)
    return FRE_vectors

def local_feature_reconstruction_error(nb_local_envs, features1_train, features2_train, features1_test = None, features2_test = None, regularizer=np.nan):
    if features1_test is None:
        features1_test = features1_train
    if features2_test is None:
        features2_test = features2_train

    n_test = features2_test.shape[0]
    lfre_vec = np.zeros(n_test)
    lfrd_vec = np.zeros(n_test)
    # D(A,B)^2 = K(A,A) + K(B,B) - 2 * K(A,B)
    #features2_test_sq_sum = np.sum(features2_test**2, axis=1)
    #squared_dist = features2_test_sq_sum[:,np.newaxis] + features2_test_sq_sum - 2 * features2_test.dot(features2_test.T)
    squared_dist = np.sum(features1_train**2, axis=1) + np.sum(features1_test**2, axis=1)[:,np.newaxis] - 2 * features1_test.dot(features1_train.T)
    for i in range(n_test):
        if i % int(n_test/10) == 0:
            print("step "+str(i)+"")
        #local_env_idx = np.argsort(squared_dist[i])[:nb_local_envs]
        #local_features1 = standardize_features(features1_train[local_env_idx])
        #local_features2 = standardize_features(features2_train[local_env_idx])

        #reconstruction_weights = feature_space_reconstruction_weights(
        #    local_features1, local_features2, regularizer
        #)
        #lfre_vec[i] = np.linalg.norm(local_features1.dot(reconstruction_weights)  - local_features2 ) / np.sqrt(len(local_env_idx))
        #lfre_vec[i] = lfre_vec[i]**2/ n_test

        local_env_idx = np.argsort(squared_dist[i])[:nb_local_envs]
        local_features1_train = features1_train[local_env_idx]
        local_features1_train_mean = np.mean(features1_train[local_env_idx], axis=0)
        local_features2_train = features2_train[local_env_idx]
        local_features2_train_mean = np.mean(features2_train[local_env_idx], axis=0)
        # standardize
        reconstruction_weights = feature_space_reconstruction_weights(
            local_features1_train - local_features1_train_mean, local_features2_train - local_features2_train_mean, regularizer
        )
        # \|x_i' - \tilde{x}_i' \|^2 / n_test
        print(np.linalg.norm(features1_test[i,:][np.newaxis,:]))
        lfre_vec[i] = np.linalg.norm(
            (features1_test[i,:][np.newaxis,:] - local_features1_train_mean).dot(reconstruction_weights) + local_features2_train_mean
            - features2_test[i,:][np.newaxis,:]
        )**2
        print(lfre_vec[i])
        # \|x_i' - \tilde{x}_i' \|^2 / n_test
        # not evactly _i_tilde but: (x_F^{(i)}-\bar{x}_F)P_{FF'}^{(i)}
        #features1_i = features1_test[local_env_idx] - local_features1_train_mean
        #features1_i = features1_test[i,:][np.newaxis,:] - local_features1_train_mean
        # not exactly _i but x_{F'}^{(i)}-\bar{x}_{F'}^{(i)}
        #features2_i = features2_test[local_env_idx]
        #features2_i = features2_test[i,:][np.newaxis,:] - local_features2_train_mean
        #lfre_vec[i], lfrd_vec[i] = feature_space_reconstruction_measures(features1_i, features2_i, reconstruction_weights=reconstruction_weights, n_test=1)
        #lfrd_vec[i] = lfrd_vec[i]**2/ np.sqrt(n_test)
    return lfre_vec, lfrd_vec

def compute_local_feature_reconstruction_error_for_pairwise_feature_spaces(
        feature_spaces1, feature_spaces2, nb_local_envs, two_split, train_ratio, seed, regularizer):
    assert( len(feature_spaces1) == len(feature_spaces2) )
    for i in range(len(feature_spaces1)):
        assert( feature_spaces1[i].shape[0] == feature_spaces2[i].shape[0] )

    n_test = feature_spaces2[0].shape[0]
    if two_split:
        train_idx, test_idx = generate_two_split_idx(n_test, train_ratio, seed)
    lfre_mat = np.zeros((len(feature_spaces1)*2, n_test))
    lfrd_mat = np.zeros((len(feature_spaces1)*2, n_test))
    for i in range(len(feature_spaces1)):
        features1 = standardize_features(feature_spaces1[i])
        features2 = standardize_features(feature_spaces2[i])
        if two_split:
            features1_train, features2_train, _, _ = split_in_two(
                    features1, features2, train_idx, test_idx)
            features1_test = features1
            features2_test = features2
            lfre_mat[2*i], lfrd_mat[2*i] = local_feature_reconstruction_error(nb_local_envs, features1_train, features2_train, features1_test, features2_test, regularizer=regularizer)
            lfre_mat[2*i+1], lfrd_mat[2*i+1] = local_feature_reconstruction_error(nb_local_envs, features2_train, features1_train, features2_test, features1_test, regularizer=regularizer)
        else:
            lfre_mat[2*i], lfrd_mat[2*i] = local_feature_reconstruction_error(nb_local_envs, features1, features2, regularizer=regularizer)
            lfre_mat[2*i+1], lfrd_mat[2*i+1] = local_feature_reconstruction_error(nb_local_envs, features2, features1, regularizer=regularizer)
    return lfre_mat, lfrd_mat

def compute_local_feature_reconstruction_error_for_all_feature_spaces_pairs(
        feature_spaces, nb_local_envs, two_split, train_ratio, seed, regularizer):
    for i in range(len(feature_spaces)):
        for j in range(i+1,len(feature_spaces)):
            assert( feature_spaces1[i].shape[0] == feature_spaces2[j].shape[0] )
    n_test = feature_spaces[0].shape[0]
    if two_split:
        train_idx, test_idx = generate_two_split_idx(n_test, train_ratio, seed)
    lfre_mat = np.zeros((len(feature_spaces), len(feature_spaces), n_test))
    lfrd_mat = np.zeros((len(feature_spaces), len(feature_spaces), n_test))
    for i in range(len(feature_spaces)):
        for j in range(len(feature_spaces)):
            features1 = feature_spaces[i]
            features2 = feature_spaces[j]
            if two_split:
                features1_train, features2_train, _, _ = split_in_two(
                        features1, features2, train_idx, test_idx)
                features1_test = features1
                features2_test = features2
                lfre_mat[i,j], lfrd_mat[i,j] = local_feature_reconstruction_error(nb_local_envs, features1_train, features2_train, features1_test, features2_test, nb_local_envs, regularizer=regularizer)
            else:
                lfre_mat[i,j], lfrd_mat[i,j] = local_feature_reconstruction_error(nb_local_envs, features1, features2, regularizer=regularizer)
    return lfre_mat, lfrd_mat
