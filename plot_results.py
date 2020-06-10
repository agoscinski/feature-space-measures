import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


RESULTS_FOLDER = "results/"
PLOTS_FOLDER = "plots/"

def plot_feature_space_measure_matrix(measure_matrix, title, name, metadata):
    rc("text", usetex=True)
    ticks_table = {
            ("RadialSpectrum","DVR"): r"$\textrm{RS}_{\textrm{DVR}}$",
            ("RadialSpectrum","GTO"): r"$\textrm{RS}_{\textrm{GTO}}$",
            ("PowerSpectrum","DVR"): r"$\textrm{PS}_{\textrm{DVR}}$",
            ("PowerSpectrum","GTO"): r"$\textrm{PS}_{\textrm{GTO}}$",
            ("BiSpectrum","DVR"): r"$\textrm{BS}_{\textrm{DVR}}$",
            ("BiSpectrum","GTO"): r"$\textrm{BS}_{\textrm{GTO}}$"
    }
    nb_features_hypers = len(metadata["features_hypers"])
    ticks = [ticks_table[(metadata["features_hypers"][i]["soap_type"],
        metadata["features_hypers"][i]["radial_basis"])]
         for i in range(nb_features_hypers)]

    plt.figure(figsize=(8, 8))
    plt.imshow(measure_matrix)
    plt.yticks(np.arange(nb_features_hypers), ticks, fontsize=19)
    plt.xticks(np.arange(nb_features_hypers), ticks, fontsize=19)
    plt.ylim(-0.5, nb_features_hypers-0.5)
    plt.title(title, fontsize=22)
    plt.xlabel("$X_{F'}$", fontsize=22)
    plt.ylabel("$X_F$", fontsize=22)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize="x-large")
    plt.tight_layout()
    plt.savefig(PLOTS_FOLDER+name+".svg", bbox_inches='tight')
    plt.show()
    plt.close()

def plot_feature_space_measure_matrix_plain(measure_matrix, name):
    plt.figure(figsize=(8, 8))
    plt.imshow(measure_matrix)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(name+".svg", bbox_inches='tight')
    plt.show()
    plt.close()

try:
    hash_value = sys.argv[1]
    FRE_matrix = np.load(RESULTS_FOLDER+"fre_mat-"+ hash_value + ".npy")
    FRD_matrix = np.load(RESULTS_FOLDER+"frd_mat-"+ hash_value +".npy")
    with open(f"{RESULTS_FOLDER}metadata-{hash_value}.json", "r") as metadata_file:
        metadata = json.load(metadata_file)
    plot_feature_space_measure_matrix(FRE_matrix, "Error $\|X_{F'}-X_F P\|/\|\\tilde{X}_{F'}'\|$", "fre_mat-"+hash_value, metadata)
    plot_feature_space_measure_matrix(FRD_matrix, "Distortion $\|\\tilde{X}_{F'}-X_F Q\|/\|\\tilde{X}_{F'}'\|$", "frd_mat-"+hash_value, metadata)
except IndexError:
    print("Please give the hash value of the experiment as input.")

