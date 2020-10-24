import schnetpack as spk
import torch
import numpy as np
import ase.io

DEVICE = 'cpu'
DATA_PATH = '../../data'
MODEL_PATH = 'trained_schnet_models'
FEATURE_SIZE = 128

def compute_schnet_features_for_qm9(nb_structures, structure_property = 'energy_U0'):
    model_filename = MODEL_PATH+'/qm9_'+structure_property+'/best_model'
    qm9_filename = DATA_PATH+'/qm9.db'

    frames = ase.io.read(qm9_filename, ':'+str(nb_structures))
    cumulative_nb_envs = np.cumsum([frame.get_global_number_of_atoms() for frame in frames])
    struc_to_env_idx = np.hstack(([0], cumulative_nb_envs))
    nb_envs = cumulative_nb_envs[-1]

    model = torch.load(model_filename, map_location=torch.device(DEVICE))
    print("Number of interaction blocks:", len(model.representation.interactions))
    print("Gaussian width used for expanison is ", list(model.representation.distance_expansion.named_buffers())[0][1][0])
    for interaction_layer in range(len(model.representation.interactions)):
        print( "Interaction layer "+str(interaction_layer)+". has cutoff type " + str(model.representation.interactions[0].cutoff_network) +
               " cutoff "+ str(list(model.representation.interactions[4].cutoff_network.named_buffers())[0][1][0]) +" AA" )
    model.representation.return_intermediate = True

    # schnet increases the number of environments to the structure with the maximum number of environments.
    # for structures with less environments null representation are added into the array which we do want to store
    # to prevent a postfiltering, we compute the representation structure wise such that this padding does not happen 

    # 1 embedding layer + 6 interaction layers times 128 hardcoded features size
    features = torch.zeros([nb_envs, 1+len(model.representation.interactions), FEATURE_SIZE ])
    # we compute structure-wise, this way we dont have to filter representations of null environments
    for i in range(nb_structures):
        data = spk.datasets.QM9(qm9_filename, subset=list(range(i,i+1)), download=False, load_only=[structure_property], remove_uncharacterized=True)
        loader = spk.AtomsLoader(data, batch_size=1, shuffle=False, num_workers=0)
        loaded_data = list(loader)[0]
        # shape: num_structures x num_envs x num_features
        features_i = model.representation(loaded_data)[1]
        for layer in range(1+len(model.representation.interactions)):
            # reshape (num_structures * num_envs) x num_features 
            features[struc_to_env_idx[i]:struc_to_env_idx[i+1], layer] = features_i[layer].reshape(-1, FEATURE_SIZE)
    return features.detach().numpy()

def main():
    # schnet qm9 U0 has 6 interaction blocks
    nb_structures = 500
    property_keys = ["dipole_moment", "isotropic_polarizability", "homo", "lumo", "electronic_spatial_extent", "zpve", "energy_U0", "energy_U", "enthalpy_H", "free_energy", "heat_capacity"]
    for structure_property in property_keys:
        features = compute_schnet_features_for_qm9(nb_structures)
        for layer in range(features.shape[1]):
            np.save('schnet_'+structure_property+'_U0_nb_structures='+str(nb_structures)+'_layer='+str(layer)+'.npy', features[:, layer])

if __name__ == "__main__":
    # execute only if run as a script
    main()
