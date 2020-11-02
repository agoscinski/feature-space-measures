import tensorflow as tf
import yaml
import numpy as np
import os
import sys
from multiprocessing import Process, Lock

sys.path.insert(0,'./dimenet')
from dimenet.model.dimenet import DimeNet
from dimenet.model.dimenet_pp import DimeNetPP
from dimenet.model.activations import swish
from dimenet.training.data_container import DataContainer
from dimenet.training.data_provider import DataProvider


DIMENET_FOLDER = 'dimenet/'

def compute_dimenet_features_for_qm9(nb_structures, structure_property_key, model_name):
    # config.yaml for DimeNet, config_pp.yaml for DimeNet
    if model_name == "dimenet":
        with open(DIMENET_FOLDER+'config.yaml', 'r') as c:
            config = yaml.safe_load(c)
    elif model_name == "dimenet++":
        with open(DIMENET_FOLDER+'config_pp.yaml', 'r') as c:
            config = yaml.safe_load(c)
    else:
        raise ValueError(f"Dimenet model name {model_name} not known")
            

    if model_name == "dimenet":
        num_bilinear = config['num_bilinear']
        feature_size = 128
    elif model_name == "dimenet++":
        out_emb_size = config['out_emb_size']
        int_emb_size = config['int_emb_size']
        basis_emb_size = config['basis_emb_size']
        feature_size = 128
        
    emb_size = config['emb_size']
    num_blocks = config['num_blocks']

    num_spherical = config['num_spherical']
    num_radial = config['num_radial']

    # 'zeros' for Mu, HOMO, LUMO, and ZPVE; 'GlorotOrthogonal' for alpha, R2, U0, U, H, G, and Cv
    if structure_property_key in ['mu', 'homo', 'lumo', 'zpve']:
        output_init = 'zeros'
    elif structure_property_key in ['alpha', 'r2', 'U0', 'U', 'H', 'G', 'Cv']:
        output_init = 'GlorotOrthogonal'  
    else:
        raise ValueError("Wrong property key "+structure_property_key)

    cutoff = config['cutoff']
    print("Cutoff used",cutoff, flush=True)
    envelope_exponent = config['envelope_exponent']

    num_before_skip = config['num_before_skip']
    num_after_skip = config['num_after_skip']
    num_dense_output = config['num_dense_output']

    num_train = config['num_train']
    num_valid = config['num_valid']
    data_seed = config['data_seed']
    dataset_path = DIMENET_FOLDER+config['dataset']

    batch_size = config['batch_size']

    #####################################################################
    # Change this if you want to predict a different target, e.g. to ['U0']
    # (but don't forget to change eoutput_init as well)
    targets = [structure_property_key]
    #####################################################################5

    if model_name == "dimenet":
        model = DimeNet(
                emb_size=emb_size, num_blocks=num_blocks, num_bilinear=num_bilinear,
                num_spherical=num_spherical, num_radial=num_radial,
                cutoff=cutoff, envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                num_dense_output=num_dense_output, num_targets=len(targets),
                activation=swish, output_init=output_init)
        best_ckpt_file = DIMENET_FOLDER+'pretrained/dimenet/'+structure_property_key+'/ckpt'
    elif model_name == "dimenet++":
        model = DimeNetPP(
                emb_size=emb_size, out_emb_size=out_emb_size,
                int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
                num_blocks=num_blocks, num_spherical=num_spherical, num_radial=num_radial,
                cutoff=cutoff, envelope_exponent=envelope_exponent,
                num_before_skip=num_before_skip, num_after_skip=num_after_skip,
                num_dense_output=num_dense_output, num_targets=len(targets),
                activation=swish, output_init=output_init)
        best_ckpt_file = DIMENET_FOLDER+'pretrained/dimenet_pp/'+structure_property_key+'/ckpt'
    else:
        raise ValueError(f"Unknown model name: '{model_name}'")

    model.load_weights(best_ckpt_file)

    data_container = DataContainer(dataset_path, cutoff=cutoff, target_keys=targets)

    # randomized attribute does not matter if idx_to_data() is used
    # Initialize DataProvider (splits dataset into training, validation and test set based on data_seed)
    data_provider = DataProvider(data_container, num_train, num_valid, 1,
                                 randomized=True)

    cumulative_nb_envs = np.cumsum(np.load(DIMENET_FOLDER+"data/qm9_eV.npz")['N'][:nb_structures])
    struc_to_env_idx = np.hstack(([0], cumulative_nb_envs))
    nb_envs = cumulative_nb_envs[-1]

    dimenet_features = np.zeros( (nb_envs, len(model.output_blocks), feature_size) )
    for i in range(nb_structures):
        if (i % 500 == 0):
            print(structure_property_key,i,flush=True)
        model(data_provider.idx_to_data(i)[0])
        for layer in range(len(model.output_blocks)):
            dimenet_features[struc_to_env_idx[i]:struc_to_env_idx[i+1], layer, :] = model.output_blocks[layer].representation.numpy()
    return dimenet_features



def compute_smooth_cutoff(cutoff=5, p=6, grid_size=100): 
    d = np.linspace(0,cutoff,grid_size)
    d_dimenet = 1 - (p+1)*(p+2)/2 * (d/cutoff)**(p-1) + p*(p+2)*(d/cutoff)**(p) - p*(p+1)/2 * (d/cutoff)**(p+1)

    err = np.zeros(grid_size)
    smooth_widths = np.linspace(cutoff/grid_size, cutoff, grid_size)
    for i in range(grid_size):
        smooth_width = smooth_widths[i]
        d_scaled = np.pi * (d - cutoff + smooth_width) / smooth_width;
        d_soap = (0.5 * (1. + np.cos(d_scaled)))
        d_soap[d <= cutoff-smooth_width] = 1
        err[i] = np.linalg.norm(d_dimenet - d_soap)
    print("The smooth cutoff width most similar to the one of dimenet", smooth_widths[np.argmin(err)])

def compute_and_store_dimenet_features_for_qm9(nb_structures, structure_property_key, model_name):
    print("structure_property_key", structure_property_key, flush=True)
    features = compute_dimenet_features_for_qm9(nb_structures, structure_property_key, model_name)
    for layer in range(features.shape[1]):
        np.save('dimenet_qm9_'+structure_property_key+'_nb_structures='+str(nb_structures)+'_layer='+str(layer)+'.npy', features[:, layer])

def main():
    nb_structures = 1000
    #structure_properties_key = ['U0', 'mu', 'alpha', 'homo', 'lumo', 'r2', 'zpve', 'U', 'H', 'G', 'Cv']
    #structure_properties_key = ['U0', 'mu', 'alpha', 'homo']
    structure_properties_key = ['U0']
    model_name = "dimenet++"
    for structure_property_key in structure_properties_key:
        Process(target=compute_and_store_dimenet_features_for_qm9, args=(nb_structures, structure_property_key, model_name)).start()

if __name__ == "__main__":
    # execute only if run as a script
    main()
