import numpy as np
import copy
import ase.io
from nice.blocks import (StandardSequence,
                         StandardBlock,
                         ThresholdExpansioner,
                         CovariantsPurifierBoth,
                         IndividualLambdaPCAsBoth,
                         ThresholdExpansioner,
                         InvariantsPurifier,
                         InvariantsPCA,
                         InitialScaler)
from nice.utilities import get_spherical_expansion

def compute_nice_features(feature_hypers, frames, train_idx, center_atom_id_mask):
    nb_blocks = feature_hypers["nb_blocks"]
    for nu in feature_hypers["nus"]:
        if nu not in range(1, nb_blocks+2):
            raise ValueError(f"nu={nu} should be in range [1, nb_blocks+1] with nb_blocks={nb_blocks}")

    if (nb_blocks == 1):
        blocks = [
                StandardBlock(None, None, None,
                      ThresholdExpansioner(num_expand=300, mode='invariants'),
                      InvariantsPurifier(max_take=10),
                      InvariantsPCA(n_components=200))
            ]
    elif (nb_blocks == 2):
        blocks = [
            StandardBlock(ThresholdExpansioner(num_expand=300),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=100),
                          ThresholdExpansioner(num_expand=300, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200)),
            StandardBlock(None, None, None,
                          ThresholdExpansioner(num_expand=300, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200))
            ]
    elif (nb_blocks == 3):
        blocks = [
            StandardBlock(ThresholdExpansioner(num_expand=800),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=800),
                          ThresholdExpansioner(num_expand=800, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=800)),
            StandardBlock(ThresholdExpansioner(num_expand=600),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=600),
                          ThresholdExpansioner(num_expand=800, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=800)),
            StandardBlock(None, None, None,
                          ThresholdExpansioner(num_expand=200, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200))
            ]
    else:
        raise ValueError("nb_blocks > 3 is not supported")

    invariant_nice_calculator = StandardSequence(blocks,
                        initial_scaler=InitialScaler(
                            mode='variance', individually=False))

    all_species = np.unique(np.concatenate([frame.numbers for frame in frames]))
    train_coefficients = get_spherical_expansion([frames[idx] for idx in train_idx], feature_hypers['spherical_coeffs'], all_species)
    train_coefficients = np.concatenate([train_coefficients[key] for key in train_coefficients.keys()], axis=0)
    invariant_nice_calculator.fit(train_coefficients)
    coefficients = get_spherical_expansion(frames, feature_hypers['spherical_coeffs'], all_species)
    nice_calculator = {}
    features_sp = {}
    for species in all_species:
        nice_calculator[species] = copy.deepcopy(invariant_nice_calculator)
        features_sp_block = nice_calculator[species].transform(coefficients[species], return_only_invariants=True)
        features_sp[species] = np.hstack( [features_sp_block[block] for block in feature_hypers["nus"]] )
    
    # envs to struc for all strucs
    nb_envs = sum([features_sp[species].shape[0] for species in all_species])
    nb_features = features_sp[all_species[0]].shape[1]
    features = np.zeros((nb_envs, nb_features))
    for species in all_species:
        sp_mask = np.concatenate( [frame.numbers==species for frame in frames] )
        features[np.arange(nb_envs)[sp_mask]] = features_sp[species]
    print("nice features.shape", features.shape)

    cumulative_env_idx = np.hstack( (0, np.cumsum([len(frame) for frame in frames])) )
    sample_idx = np.concatenate( [np.array(center_atom_id_mask[idx]) + cumulative_env_idx[idx] for idx in range(len(center_atom_id_mask))] )
    return features[sample_idx]

feature_hypers = {
        "nb_blocks": 3,
        "nus": [1,2,3,4],
        "spherical_coeffs": {
            "radial_basis": "GTO",
            "interaction_cutoff": 4.0,
            "max_radial": 6,
            "max_angular": 4,
            "gaussian_sigma_constant": 0.5,
            "gaussian_sigma_type": "Constant",
            "cutoff_smooth_width": 0.5,
        }
    }
frames = ase.io.read("selection-10k.extxyz", ":4000")
print(len(frames))
for i in range(len(frames)):
    frames[i].cell = np.eye(3) * 15
    frames[i].center()
    frames[i].wrap(eps=1e-11)
    frames[i].numbers = np.ones(len(frames[i]))

np.random.seed(0x5f3759df)
idx = np.arange(len(frames))
np.random.shuffle(idx)
split_id = int(len(idx) * 0.5)
train_idx = idx[:split_id]

center_atom_id_mask = [[0] for frame in frames]

features = compute_nice_features(feature_hypers, frames, train_idx, center_atom_id_mask)
print("features.shape", features.shape)
np.save("methane-allc", features)
