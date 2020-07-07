import numpy as np
import time
from scipy.sparse.linalg import svds as svd


def svd_select(A, n, k=1, idxs=None, **kwargs):
    """
    Selection function which computes the CUR indices using the SVD Decomposition
    """

    if idxs is None:
        idxs = []
    else:
        idxs = list(idxs)
    Acopy = A.copy()

    for nn in range(n):
        if len(idxs) <= n:
            try:
                (S, v, D) = np.linalg.svd(Acopy)
            except np.linalg.LinAlgError:
                print("WARNING ---- {} points requested, but only {} are availabe in CUR-SVD".format(n, len(idxs)))
                return idxs
            pi = (D[:k] ** 2.0).sum(axis=0)
            pi[idxs] = 0  # eliminate possibility of selecting same column twice
            i = pi.argmax()
            idxs.append(i)

        v = Acopy[:, idxs[nn]] / np.sqrt(
            np.matmul(Acopy[:, idxs[nn]], Acopy[:, idxs[nn]])
        )

        for i in range(Acopy.shape[1]):
            Acopy[:, i] -= v * np.dot(v, Acopy[:, i])

    return idxs


class CUR:
    """
    Performs CUR Decomposition on a Supplied Matrix

    ---Arguments---
    matrix: matrix to be decomposed
    precompute: (int, tuple, Nonetype) number of columns, rows to be computed
                upon instantiation. Defaults to None.
    feature_select: (bool) whether to compute only column indices
    pi_function: (<func>) Importance metric and selection for the matrix
    symmetry_tolerance: (float) Tolerance by which a matrix is symmetric
    params: (dict) Dictionary of additional parameters to be passed to the
            pi function

    ---References---
    1.  G.  Imbalzano,  A.  Anelli,  D.  Giofre,  S.  Klees,  J.  Behler,
        and M. Ceriotti, J. Chem. Phys.148, 241730 (2018)
    """

    def __init__(
        self, matrix, feature_select=False, symmetry_tolerance=1e-4, **svd_kwargs
    ):
        self.matrix = matrix
        self.symmetric = self.matrix.shape == self.matrix.T.shape and np.all(np.abs(self.matrix - self.matrix.T)) < symmetry_tolerance
        self.feature_select = feature_select
        self.svd_kwargs = dict(**svd_kwargs)

        self.idx_c = None
        self.idx_r = None

    def compute_idx(self, n_c, n_r):
        idx_c = svd_select(self.matrix, n_c, idxs=self.idx_c, **self.svd_kwargs)
        if self.feature_select:
            idx_r = np.asarray(range(self.matrix.shape[1]))
        elif not self.symmetric:
            idx_r = svd_select(self.matrix.T, n_r, idxs=self.idx_r, **self.svd_kwargs)
        else:
            idx_r = idx_c
        return idx_c, idx_r

    def compute(self, n_c, n_r=None):
        """Compute the n_c selected columns and n_r selected rows"""
        if self.feature_select:
            n_r = self.matrix.shape[1]
        elif self.symmetric and n_r is None:
            n_r = n_c
        elif n_r is None:
            print("You must specify a n_r for non-symmetric matrices.")

        if self.idx_c is None or (not self.feature_select and self.idx_r is None):
            idx_c, idx_r = self.compute_idx(n_c, n_r)
            self.idx_c, self.idx_r = idx_c, idx_r
        elif len(self.idx_c) < n_c or len(self.idx_r) < n_r:
            idx_c, idx_r = self.compute_idx(n_c, n_r)
            self.idx_c, self.idx_r = idx_c, idx_r
        else:
            idx_c = self.idx_c[:n_c]
            idx_r = self.idx_r[:n_r]

        # The CUR Algorithm
        A_c = self.matrix[:, idx_c]
        if self.symmetric and not self.feature_select:
            A_r = A_c.T
        elif self.feature_select:
            A_r = self.matrix.copy()
        else:
            A_r = self.matrix[idx_r, :]

        # Compute S.
        S = np.matmul(np.matmul(np.linalg.pinv(A_c), self.matrix), np.linalg.pinv(A_r))
        return A_c, S, A_r

    def compute_P(self, n_c, thresh=1e-12):
        """Computes the projector into latent-space for ML models"""

        A_c, S, A_r = self.compute(n_c)
        SA = np.matmul(S, A_r)
        SA = np.matmul(SA, SA.T)

        v_SA, U_SA = np.linalg.eigh(SA)
        v_SA[v_SA < thresh] = 0

        self.P = np.matmul(U_SA, np.diagflat(np.sqrt(v_SA)))
        return self.P
