import numpy as np
import scipy
from CUR import CUR


def select_features(features, hypers):
    if hypers['n_features'] >= features.shape[1]:
        print("Warning: n_features >= nb_samples")
        return np.arange(features.shape[1])
    if hypers['type'] == 'FPS':
        return select_fps(features.T, hypers)
    if hypers['type'] == 'CUR':
        return select_cur(features, hypers)
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
