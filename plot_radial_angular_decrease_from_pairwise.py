import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

RESULTS_FOLDER = "results/"
PLOTS_FOLDER = "plots/"

def plot_radial_angular_trajectories(gto_measure_matrix, dvr_measure_matrix, title, name, gto_metadata, dvr_metadata):
    rc("text", usetex=True)
    nb_features_hypers = len(gto_metadata["features_hypers"])
    max_radials = np.array([gto_metadata["features_hypers"][i][0]["max_radial"]
         for i in range(nb_features_hypers)])
    max_angulars = np.array([gto_metadata["features_hypers"][i][0]["max_angular"]
         for i in range(nb_features_hypers)])

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(max_radials, gto_measure_matrix[0], label="GTO")
    ax1.plot(max_radials, dvr_measure_matrix[0], label="DVR")
    #plt.plot(max_angulars, gto_measure_matrix[np.tril_indices(len(max_angulars),-1)], label="DVR")
    # add x axis for max radial
    ax1.set_xlabel("max radial $n$", fontsize=18)
    ax1.set_xticks(max_radials)
    ax1.set_xticklabels(max_radials, fontsize=18)

    ax1.set_ylabel(title, fontsize=18)
    locs, labels = plt.yticks()
    ax1.set_yticks(locs)
    ax1.set_yticks(np.round(locs,2)[::2])
    ax1.set_yticklabels(np.round(locs,2)[::2], fontsize=18)
    ax1.tick_params(labelsize=15)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xlabel("max angular $l$", fontsize=18)
    ax2.set_xticks(max_radials)
    ax2.set_xticklabels(max_angulars, fontsize=18)

    ax1.legend(fontsize=18)
    plt.title(title, fontsize=22)
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER+name+".svg", bbox_inches='tight')
    plt.show()
    plt.close()


gto_hash_value = sys.argv[1]
dvr_hash_value = sys.argv[2]
gto_gfre_matrix = np.load(RESULTS_FOLDER+"fre_mat-"+ gto_hash_value + ".npy")
gto_gfrd_matrix = np.load(RESULTS_FOLDER+"frd_mat-"+ gto_hash_value + ".npy")

dvr_gfre_matrix = np.load(RESULTS_FOLDER+"fre_mat-"+ dvr_hash_value + ".npy")
dvr_gfrd_matrix = np.load(RESULTS_FOLDER+"frd_mat-"+ dvr_hash_value + ".npy")

with open(f"{RESULTS_FOLDER}metadata-{gto_hash_value}.json", "r") as gto_metadata_file:
    gto_metadata = json.load(gto_metadata_file)
with open(f"{RESULTS_FOLDER}metadata-{dvr_hash_value}.json", "r") as dvr_metadata_file:
    dvr_metadata = json.load(dvr_metadata_file)
plot_radial_angular_trajectories(gto_gfre_matrix, dvr_gfre_matrix, "$\\textrm{GRFE}((n,l),(n+4,l+1))$", "gfre_radial_angular-gto-"+gto_hash_value+"-dvr-"+dvr_hash_value, gto_metadata, dvr_metadata)
plot_radial_angular_trajectories(gto_gfrd_matrix, dvr_gfrd_matrix, "$\\textrm{GRFD}((n,l),(n+4,l+1))$", "gfrd_radial_angular-gto-"+gto_hash_value+"-dvr-"+dvr_hash_value, gto_metadata, dvr_metadata)
