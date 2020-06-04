import sys

sys.path.insert(0, "/local/scratch/goscinsk/lib/librascal/build")
from rascal.representations import SphericalInvariants
import numpy as np
import scipy
import ase.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


# Reconstructs the manifold as in Fig. 2 a) https://arxiv.org/abs/2001.11696

# XYZ of manifold, plus corresponds to the C^+ manifold and minus to the C^- manifold
frames_minus = ase.io.read("data/manif-minus.extxyz", ":")
frames_plus = ase.io.read("data/manif-plus.extxyz", ":")
frames = frames_minus + frames_plus
for frame in frames:
    frame.cell = np.eye(3) * 5
    frame.center()
    frame.wrap(eps=1e-11)

cutoff = 4
max_radial = 5
max_angular = 4
normalize = False

hypers_ps_gto = {
    "soap_type": "PowerSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}
hypers_bs_gto = {
    "soap_type": "BiSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}


print("Computing representations, should take around 1 minute")
# H-H species are only considered around the central C atom environment
representation = SphericalInvariants(**hypers_ps_gto)
print("Start computing power spectrum...")
soap_ps_gto_features = representation.transform(frames).get_features_by_species(representation)[(1, 1)][::5]
print("Finished computing power spectrum")
# H-H-H species are only considered around the central C atom environment
representation = SphericalInvariants(**hypers_bs_gto)
print("Start computing bispectrum...")
soap_bs_gto_features = representation.transform(frames).get_features_by_species(representation)[(1, 1, 1)][::5]
print("Finished computing bispectrum")

# Extract first 3 PCA components


def extract_principal_compononents(features):
    cov = features.T.dot(features)
    D, U = scipy.linalg.eigh(cov)
    return features.dot(U[:, -3:])


def plot_first_3_features(features, name):
    X, Y, Z = features[:, 0], features[:, 1], features[:, 2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(X[:80] * 500, Y[:80] * 500, Z[:80] * 500)
    ax.scatter(X[80:] * 500, Y[80:] * 500, Z[80:] * 500)
    plt.title(name)
    plt.show()
    plt.close()


print("Starting eigendecomposition of power spectrum features...")
truncated_soap_bs_gto_features = extract_principal_compononents(soap_bs_gto_features)
print("Finished eigendecomposition of power spectrum features")
print("Starting eigendecomposition of bispectrum features...")
truncated_soap_ps_gto_features = extract_principal_compononents(soap_ps_gto_features)
print("Finished eigendecomposition of bispectrum features")
plot_first_3_features(truncated_soap_ps_gto_features, "Power spectrum first 3 principal components")
plot_first_3_features(truncated_soap_bs_gto_features, "Bispectrum first 3 principal components")
