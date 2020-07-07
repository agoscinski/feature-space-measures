import numpy as np

from CUR import CUR


def select_features(features, hypers):
    if hypers['type'] == 'FPS':
        return select_fps(features.T, hypers).T
    if hypers['type'] == 'CUR':
        return select_cur(features, hypers)
    else:
        raise ValueError('unknown feature selection type ' + hypers['type'])


def select_fps(X, hypers):
    requested = hypers['n_features']

    if requested > X.shape[0]:
        return X

    idx = np.zeros(requested, dtype=np.int)
    # Pick first point at random
    idx[0] = np.random.randint(0, X.shape[0])

    # Compute distance from all points to the first point
    d1 = np.linalg.norm(X - X[idx[0]], axis=1)**2

    # Loop over the remaining points...
    for i in range(1, requested):
        # Get maximum distance and corresponding point
        idx[i] = np.argmax(d1)

        # Compute distance from all points to the selected point
        d2 = np.linalg.norm(X - X[idx[i]], axis=1)**2

        # Set distances to minimum among the last two selected points
        d1 = np.minimum(d1, d2)

        if d1.max() == 0.0:
            print("WARNING ---- {} points requested, but only {} are availabe in FPS".format(requested, i))
            return X[idx[:i]]

    return X[idx]


def select_cur(X, hypers):
    requested = hypers['n_features']

    if requested > X.shape[0]:
        return X

    cur = CUR(X, feature_select=True)
    cur.compute(requested)

    print(X[:, cur.idx_c].shape)

    return X[:, cur.idx_c]
