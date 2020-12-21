import schnetpack as spk
import torch
import numpy as np
import ase.io

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=4
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=4

import resource
def memory_limit(nb_bytes):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (nb_bytes, hard))
# limits memory usage to 700 GB
memory_limit(int(7e+11))

DEVICE = 'cpu'
DATA_PATH = '../../data'
MODEL_PATH = 'trained_schnet_models'
FEATURE_SIZE = 128

def compute_schnet_features_for_qm9(nb_structures, structure_property = 'energy_U0', target="Atom", layers=None):
    model_filename = MODEL_PATH+'/qm9_'+structure_property+'/best_model'
    qm9_db_filename = DATA_PATH+'/qm9.db'
    qm9_extxyz_filename = DATA_PATH+'/qm9.extxyz'

    model = torch.load(model_filename, map_location=torch.device(DEVICE))
    print("Number of interaction blocks:", len(model.representation.interactions), flush=True)
    print("Gaussian width used for expanison is ", list(model.representation.distance_expansion.named_buffers())[0][1][0], flush=True)
    for interaction_layer in range(len(model.representation.interactions)):
        print( "Interaction layer "+str(interaction_layer)+". has cutoff type " + str(model.representation.interactions[0].cutoff_network) +
               " cutoff "+ str(list(model.representation.interactions[4].cutoff_network.named_buffers())[0][1][0]) +" AA" , flush=True)
    model.representation.return_intermediate = True

    # schnet increases the number of environments to the structure with the maximum number of environments.
    # for structures with less environments null representation are added into the array which we do want to store
    # to prevent a postfiltering, we compute the representation structure wise such that this padding does not happen 

    
    layers = range(1+len(model.representation.interactions)) if layers is None else layers
    if target == "Atom":
        frames = ase.io.read(qm9_extxyz_filename , ':'+str(nb_structures))
        cumulative_nb_envs = np.cumsum([frame.get_global_number_of_atoms() for frame in frames])
        struc_to_env_idx = np.hstack(([0], cumulative_nb_envs))
        nb_envs = cumulative_nb_envs[-1]
        # 1 embedding layer + 6 interaction layers times 128 hardcoded features size
        features = torch.zeros([nb_envs, len(layers), FEATURE_SIZE])
        # we compute structure-wise, this way we dont have to filter representations of null environments
        for i in range(nb_structures):
            if (i % (nb_structures//10) == 0):
                print(f"{i}/{nb_structures} structures", flush=True)
            data = spk.datasets.QM9(qm9_db_filename, subset=list(range(i,i+1)), download=False, load_only=[structure_property], remove_uncharacterized=True)
            loader = spk.AtomsLoader(data, batch_size=1, shuffle=False, num_workers=0)
            loaded_data = list(loader)[0]
            # shape: num_structures x num_envs x num_features
            features_i = model.representation(loaded_data)[1]
            for layer in range(len(layers)):
                # reshape (num_structures * num_envs) x num_features
                features[struc_to_env_idx[i]:struc_to_env_idx[i+1], layer] = features_i[layer].reshape(-1, FEATURE_SIZE)
    elif target == "Structure":
        features = torch.zeros([nb_structures, len(layers), FEATURE_SIZE])
        # we compute structure-wise, this way we dont have to filter representations of null environments
        for i in range(nb_structures):
            if (i % (nb_structures//10) == 0):
                print(f"{i}/{nb_structures} structures", flush=True)
            data = spk.datasets.QM9(qm9_db_filename, subset=list(range(i,i+1)), download=False, load_only=[], remove_uncharacterized=True)
            loader = spk.AtomsLoader(data, batch_size=1, shuffle=False, num_workers=0)
            loaded_data = list(loader)[0]
            # shape: num_structures x num_envs x num_features
            features_i = model.representation(loaded_data)[1]
            for layer_idx in range(len(layers)):
                # reshape (num_structures * num_envs) x num_features
                features[i, layer_idx] = torch.mean(features_i[layers[layer_idx]].reshape(-1, FEATURE_SIZE), 0)
    else:
        raise ValueError(f"target {target} is not supported")
    return features.detach().numpy()

def main():
    # schnet qm9 U0 has 6 interaction blocks
    nb_structures = 130831
    #property_keys = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
    property_keys = ["soap_876b10bc"]
    target = "Structure"
    layers = [6]
    for structure_property in property_keys:
        features = compute_schnet_features_for_qm9(nb_structures, structure_property, target, layers)
        layers = range(features.shape[1]) if layers is None else layers
        for layer_idx in range(len(layers)):
            np.save('schnet_qm9_'+structure_property+'_nb_structures='+str(nb_structures)+'_layer='+str(layers[layer_idx])+'_target='+target+'.npy', features[:, layer_idx])

if __name__ == "__main__":
    # execute only if run as a script
    main()
