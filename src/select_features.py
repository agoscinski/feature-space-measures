import numpy as np
import scipy
from src.CUR import CUR
#from src.kpcovr import KPCovR


def select_features(features, features_train, hypers, Y=None):
    if 'n_features' in hypers:
        if hypers['n_features'] >= features_train.shape[1]:
            print("Warning: n_features >= nb_samples")
            features_idx = np.arange(features_train.shape[1])
            return features[:, features_idx]
    if hypers['type'] == 'FPS':
        features_idx = select_fps(features_train.T, hypers)
        return features[:, features_idx]
    elif hypers['type'] == 'CUR':
        features_idx = select_cur(features_train, hypers)
        return features[:, features_idx]
    elif hypers['type'] == 'PCA':
        pca = select_pca(features_train, hypers)
        print("pca.explained_variance_ratio_", np.sum(pca.explained_variance_ratio_))
        return pca.transform(features)
    elif hypers['type'] == 'KPCA':
        kpcovr = select_kpca(features_train, Y, hypers)
        return kpcovr.transform(features)
    else:
        raise ValueError('unknown feature selection type ' + hypers['type'])


def select_fps(X, hypers, seed=0x5f3759df):
    requested = hypers['n_features']
    idx = np.zeros(requested, dtype=np.int)
    idx[:] = np.nan
    # Pick first point at random
    np.random.seed(seed)
    idx[0] = np.random.randint(0, X.shape[0])
    # Compute distance from all points to the first point

    # Loop over the remaining points...
    for i in range(requested-1):
        # Get maximum distance and corresponding point

        # Compute distance from all points to the selected point
        #d2 = np.linalg.norm(X - X[idx[i]], axis=1)**2
        d2 = np.sum( scipy.spatial.distance.cdist(X, X[idx[:i+1]]), axis=1)

        # Set distances to minimum among the last two selected points
        for k in np.argsort(d2)[::-1]:
            if not(k in idx):
                new_id = k
                break;

        idx[i+1] = new_id
        #if np.sort(d2, order='descending')[new_id] == 0.0:
        #    print("WARNING ---- {} points requested, but only {} are availabe in FPS".format(requested, i))
        #    return X[idx[:i], :]

    idx[-1] = new_id
    return idx


def select_cur(X, hypers):
    requested = hypers['n_features']

    cur = CUR(X, feature_select=True)
    cur.compute(requested)

    #print(X[:, cur.idx_c].shape)
    return cur.idx_c

def select_pca(X, hypers):
    import sklearn.decomposition
    if 'n_features' in hypers:
        return sklearn.decomposition.PCA(n_components=hypers['n_features']).fit(X)
    elif 'explained_variance_ratio' in hypers:
        # increase n_components by 1000 up to 10000 until  
        # TODO make max number of features hyper
        for i in range(19): 
            pca = sklearn.decomposition.PCA(n_components=min(1000+(500*i),X.shape[1])).fit(X)
            if ( np.sum(pca.explained_variance_ratio_) >= hypers['explained_variance_ratio'] ):
                n_components_fulfilling_ratio = np.argmax(np.cumsum(pca.explained_variance_ratio_) > hypers['explained_variance_ratio'])
                pca.n_components_ = n_components_fulfilling_ratio
                break;
        if ( np.sum(pca.explained_variance_ratio_) < hypers['explained_variance_ratio'] ):
            print("WARNING: explained_variance_ratio was not reached in feature selection, continue with 10000 features")
        print("pca.n_components_:",pca.n_components_, flush=True)
        return pca
    else:
        raise ValueError(f"Missing PCA parameters, PCA hypers:", hypers)


def select_kpca(X, Y, hypers):
    print("NOT FINISHED")
    kpcovr_calculators = KPCovR(alpha=a, n_PC=hypers['n_features'],
                                           kernel_type="linear",
                                           regularization=1e-6)
    kpcovr_calculators.fit(X, Y)
    return kpcovr_calculators

