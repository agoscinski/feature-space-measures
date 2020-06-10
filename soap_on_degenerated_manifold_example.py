import sys
sys.path.insert(0, "/home/alexgo/lib/librascal/build")

from feature_space_measures import reconstruction_measure_matrix
from rascal.representations import SphericalInvariants
import numpy as np
import scipy

import ase.io


frames_minus = ase.io.read("data/manif-minus.extxyz", ":")
frames_plus = ase.io.read("data/manif-plus.extxyz", ":")
frames = frames_minus + frames_plus
for frame in frames:
    frame.cell = np.eye(3) * 5
    frame.center()
    frame.wrap(eps=1e-11)

cutoff = 4
max_radial = 7
max_angular = 3
normalize = False

hypers_rs_dvr = {
    "soap_type": "RadialSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_rs_gto = {
    "soap_type": "RadialSpectrum",
    "radial_basis": "GTO",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

hypers_ps_dvr = {
    "soap_type": "PowerSpectrum",
    "radial_basis": "DVR",
    "interaction_cutoff": cutoff,
    "max_radial": max_radial,
    "max_angular": max_angular,
    "gaussian_sigma_constant": 0.3,
    "gaussian_sigma_type": "Constant",
    "cutoff_smooth_width": 0.01,
    "normalize": normalize,
}

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

representation = SphericalInvariants(**hypers_rs_dvr)
soap_rs_dvr_features = representation.transform(frames).get_features_by_species(representation)[(1,)][::5]
representation = SphericalInvariants(**hypers_rs_gto)
soap_rs_gto_features = representation.transform(frames).get_features_by_species(representation)[(1,)][::5]
representation = SphericalInvariants(**hypers_ps_dvr)
soap_ps_dvr_features = representation.transform(frames).get_features_by_species(representation)[(1, 1)][::5]
representation = SphericalInvariants(**hypers_ps_gto)
soap_ps_gto_features = representation.transform(frames).get_features_by_species(representation)[(1, 1)][::5]

feature_spaces = [soap_rs_dvr_features, soap_rs_gto_features, soap_ps_dvr_features, soap_ps_gto_features]
FRE_matrix, FRD_matrix = reconstruction_measure_matrix(feature_spaces)
