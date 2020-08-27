from representation import compute_kernel_from_squared_distance, compute_squared_distance, compute_representation
from experiment import read_dataset
import numpy as np
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
from scalers import standardize_features
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rc('font',**{'family':'sans-serif','sans-serif':['Ubuntu'],'size':6})
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.size'] = 4

mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 4

mpl.rcParams['xtick.labelsize'] = 24
mpl.rcParams['ytick.labelsize'] = 24

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.labelsize'] = 27
mpl.rcParams['legend.fontsize'] = 24
mpl.rcParams['axes.titlesize'] = 20

def compute_squared_distance_matrix(features, train_idx):
    features = standardize_features(features, train_idx)
    return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)


def plot(densitymat, title, file_name):
    densitymat[densitymat<0] = 0
    densitymat = np.sqrt(densitymat)
    plt.imshow(densitymat, vmin=0, vmax=3)
    plt.xlabel("$z_{\\text{H}}$ / \AA")
    plt.ylabel("$z_{\\text{H}}$ / \AA")
    locs, labels = plt.xticks()
    #print(Hdist)
    locs = [np.where(Hdist==1)[0][0], np.where(Hdist==2)[0][0], np.where(Hdist==3)[0][0], np.where(Hdist==4)[0][0]]
    #print(locs)
    plt.xticks(locs, [1,2,3,4])
    plt.yticks(locs, [1,2,3,4])
    plt.title(title)
    cbarticks = [0,1,2,3]
    cbar = plt.colorbar(ticks=cbarticks)
    cbar.ax.set_ylim(0,3)
    cbar.set_ticks(cbarticks)
    cbar.set_ticklabels(cbarticks)
    plt.savefig(file_name+".png", bbox_inches='tight')
    plt.savefig(file_name+".svg", bbox_inches='tight')
    plt.savefig(file_name+".pdf", bbox_inches='tight')
    plt.show()

frames = read_dataset("displaced-methane-step_size=0.01-range=[0.5,4.5]-seed=None.extxyz", "")
nb_samples = len(frames)
train_idx = np.arange(nb_samples)
args = np.argsort( [frame.info["hydrogen_distance"] for frame in frames] )
Hdist = np.sort( [frame.info["hydrogen_distance"] for frame in frames] )
frames = [frames[args[i]] for i in range(len(frames))]
center_atom_id_mask = [[0] for frame in frames]
for i in range(len(frames)):
    mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
for delta_normalization in [True, False]:
    features_hyper = {
        "feature_type": "wasserstein",
        "feature_parameters": {
            "nb_basis_functions": 500,
            "grid_type": "equispaced",
            "delta_normalization": delta_normalization,
            "delta_sigma": None, # 0.5
            "delta_offset_percentage": 0, # 0.1
            "soap_parameters": {
                "soap_type": "RadialSpectrum",
                "radial_basis": "DVR",
                "interaction_cutoff": 4,
                "max_radial": 1000,
                "max_angular": 0,
                "gaussian_sigma_constant": 0.5,
                "gaussian_sigma_type": "Constant",
                "cutoff_smooth_width": 0.5,
                "normalize": False
            },
        },
    }

    features = compute_representation(features_hyper, frames, center_atom_id_mask)
    dmat = compute_squared_distance_matrix(features, train_idx)
    if delta_normalization:
        title = "$d_W^2$ with $\delta$ normalization"
    else:
        title = "$d_W^2$ with scaling normalization"
    if delta_normalization:
        file_name = "dmat-delta_normalization"
    else:
        file_name = "dmat-scaling_normalization"

    plot(dmat, title, file_name)


features_hyper = {
    "feature_type": "sorted_distances",
    "feature_parameters": {
        "interaction_cutoff": 4,
        "padding_type": "max"
    }
}

features = compute_representation(features_hyper, frames, center_atom_id_mask)
print(features.shape)
dmat = compute_squared_distance_matrix(features, train_idx)
title = "$d_{SD}^2$ with cutoff padding"
file_name = "dmat-sorted_distance"
plot(dmat, title, file_name)
