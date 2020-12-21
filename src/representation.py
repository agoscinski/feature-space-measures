import numpy as np
import scipy
import scipy.special

from rascal.representations import SphericalInvariants
from rascal.neighbourlist.structure_manager import mask_center_atoms_by_id
from src.wasserstein import compute_squared_wasserstein_distance, compute_radial_spectrum_wasserstein_features
from src.sorted_distances import compute_sorted_distances
from src.scalers import standardize_features, standardize_kernel

FEATURES_ROOT ="features"

# computes mean features TODO generalize this to general atom computation
def compute_structure_features_from_atom_features(features, center_atom_id_mask):
    atom_to_struc_idx = np.hstack( (0, np.cumsum([len(center_mask) for center_mask in center_atom_id_mask])) )
    return np.vstack( [np.mean(features[atom_to_struc_idx[i]:atom_to_struc_idx[i+1]], axis=0) for i in range(len(center_atom_id_mask))] )

def compute_representations(features_hypers, frames, target="Atom", environments_train_idx=None, center_atom_id_mask_description="first environment"):
    if center_atom_id_mask_description == "first environment":
        print("WARNING only the first environment of all structures is computed. Please use center_atom_id_mask_description='all environments' if you want to use all environments")
        center_atom_id_mask = [[0] for frame in frames]
    elif center_atom_id_mask_description  == "all environments":
        center_atom_id_mask = [list(range(len(frame))) for frame in frames]
    else:
        raise ValueError("The center_atom_id_mask_description "+center_atom_id_mask_description+" is not available")
    print("Compute representations...", flush=True)
    feature_spaces = []
    for feature_hypers in features_hypers:
        if "hilbert_space_parameters" in feature_hypers:
            features = compute_hilbert_space_features(feature_hypers, frames, target, environments_train_idx, center_atom_id_mask)
        else:
            features = compute_representation(feature_hypers, frames, environments_train_idx, center_atom_id_mask)
            if target == "Structure" and ("target" not in feature_hypers["feature_parameters"] or feature_hypers["feature_parameters"]["target"] == "Atom"):
                features = compute_structure_features_from_atom_features(features, center_atom_id_mask)
        print("np.sum(np.isnan(features))",np.sum(np.isnan(features)))
        feature_spaces.append(features)

    print("Compute representations finished", flush=True)
    return feature_spaces


def compute_representation(feature_hypers, frames, train_idx, center_atom_id_mask):
    if feature_hypers["feature_type"] == "soap":
        for i in range(len(frames)):
            mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
        representation = SphericalInvariants(**feature_hypers["feature_parameters"])
        return representation.transform(frames).get_features(representation)
    elif feature_hypers["feature_type"] == "nice":
        # nice cannot deal with masked central atoms
        return compute_nice_features(feature_hypers["feature_parameters"], frames, train_idx, center_atom_id_mask)
    elif feature_hypers["feature_type"] == "wasserstein":
        for i in range(len(frames)):
            mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
        return compute_radial_spectrum_wasserstein_features(feature_hypers["feature_parameters"], frames)
    elif feature_hypers["feature_type"] == "sorted_distances":
        for i in range(len(frames)):
            mask_center_atoms_by_id(frames[i], id_select=center_atom_id_mask[i])
        features = compute_sorted_distances(feature_hypers["feature_parameters"], frames, center_atom_id_mask)
        return features
    elif feature_hypers["feature_type"] == "precomputed":
        parameters = feature_hypers['feature_parameters']
        # TODO parameter case seems to useless, probably can be removed 
        if ("target" not in feature_hypers["feature_parameters"]) or (feature_hypers["feature_parameters"]["target"] == "Atom"):
            nb_samples = sum([len(structure_mask) for structure_mask in center_atom_id_mask]) if "nb_samples" not in parameters.keys() else parameters["nb_samples"]
        elif (feature_hypers["feature_parameters"]["target"] == "Structure"):
            nb_samples = len(frames)
        if parameters["filetype"] == "npy":
            pathname = f"{FEATURES_ROOT}/{parameters['feature_name']}/{parameters['filename']}"
            return np.load(pathname)[:nb_samples]
        elif parameters["filetype"] == "txt":
            pathname = f"{FEATURES_ROOT}/{parameters['feature_name']}/{parameters['filename']}"
            return np.loadtxt(pathname)[:nb_samples]
        elif parameters["filetype"] == "frame_info":
            return np.array([frame.info[parameters['feature_name']] for frame in frames])[:,np.newaxis][:nb_samples]
        # TODO is deprecated??
        # hardcoded case
        elif parameters["feature_name"] == "displaced_hydrogen_distance": 
            return load_hydrogen_distance_dataset(frames)[:nb_samples]

def compute_nice_features(feature_hypers, frames, train_idx, center_atom_id_mask):
    import copy
    from nice.blocks import StandardSequence, StandardBlock, ThresholdExpansioner, CovariantsPurifierBoth, IndividualLambdaPCAsBoth, ThresholdExpansioner, InvariantsPurifier, InvariantsPCA, InitialScaler

    from nice.utilities import get_spherical_expansion
    #print(np.max(train_idx), len(frames))
    #print("len(frames)",len(frames))
    #print("frames[0]", frames[0])
    
    nb_blocks = feature_hypers["nb_blocks"]
    for nu in feature_hypers["nus"]:
        if nu not in range(1, nb_blocks+2):
            raise ValueError(f"nu={nu} should be in range [1, nb_blocks+1] with nb_blocks={nb_blocks}")

    if (nb_blocks == 1):
        blocks = [
                StandardBlock(None, None, None,
                      ThresholdExpansioner(num_expand=1000, mode='invariants'),
                      InvariantsPurifier(max_take=10),
                      InvariantsPCA(n_components=200))
            ]
    elif (nb_blocks == 2):
        blocks = [
            StandardBlock(ThresholdExpansioner(num_expand=300),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=100),
                          ThresholdExpansioner(num_expand=1000, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200)),
            StandardBlock(None, None, None,
                          ThresholdExpansioner(num_expand=1000, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200))
            ]
    elif (nb_blocks == 3):
        blocks = [
            StandardBlock(ThresholdExpansioner(num_expand=300),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=100),
                          ThresholdExpansioner(num_expand=1000, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200)),
            StandardBlock(ThresholdExpansioner(num_expand=300),
                          CovariantsPurifierBoth(max_take=10),
                          IndividualLambdaPCAsBoth(n_components=100),
                          ThresholdExpansioner(num_expand=1000, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=200)),
            StandardBlock(None, None, None,
                          ThresholdExpansioner(num_expand=1000, mode='invariants'),
                          InvariantsPurifier(max_take=10),
                          InvariantsPCA(n_components=100))
            ]
    else:
        raise ValueError("nb_blocks > 3 is not supported")

    invariant_nice_calculator = StandardSequence(blocks,
                        initial_scaler=InitialScaler(
                            mode='variance', individually=False))

    all_species = np.unique(np.concatenate([frame.numbers for frame in frames]))
    train_coefficients = get_spherical_expansion([frames[idx] for idx in train_idx], feature_hypers['spherical_coeffs'], all_species)
    train_coefficients = np.concatenate([train_coefficients[key] for key in train_coefficients.keys()], axis=0)
    #print("train_coefficients.shape", train_coefficients.shape)
    # we fit all environments, not species-wise
    invariant_nice_calculator.fit(train_coefficients)

    # does not work because `transform_sequentially` includes already
    #from nice.utilities import transform_sequentially
    #nice_calculator = {specie: invariant_nice_calculator for specie in all_species}
    #features = transform_sequentially(nice_calculator, frames, feature_hypers['spherical_coeffs'], all_species)
    #print("type(features)",type(features))
    #print("len(features)",len(features))
    #print("features min max isnan", np.min(features), np.max(features), np.sum(np.isnan(features)))
    #return features

    #print("features.shape", features.shape)

    coefficients = get_spherical_expansion(frames, feature_hypers['spherical_coeffs'], all_species)
    nice_calculator = {}
    features_sp = {}
    for species in all_species:
        #print("species ",species)
        #print(train_coefficients[species].shape)
        nice_calculator[species] = copy.deepcopy(invariant_nice_calculator)
        #print("train_coefficients[species] min max isnan", np.min(train_coefficients[species]), np.max(train_coefficients[species]), np.sum(np.isnan(train_coefficients[species])))

        features_sp_block = nice_calculator[species].transform(coefficients[species], return_only_invariants=True)

        # species-wise fitting
        #nice_calculator[species].fit(train_coefficients[species])
        #features_sp_block = nice_calculator[species].transform(coefficients[species], return_only_invariants=True)
        #features_sp[species] = np.concatenate([features_sp_block[block] for block in features_sp_block], axis=1)

        #print("len(features_sp_block)",len(features_sp_block))
        #print("features_sp_block.keys", features_sp_block.keys())
        #print("[block for block in features_sp_block]", [block for block in features_sp_block])
        #print("[features_sp_block[block].shape for block in features_sp_block]", [features_sp_block[block].shape for block in features_sp_block])
        #print("features_sp_block[1].shape", features_sp_block[1].shape)
        #print("features_sp_block[2].shape", features_sp_block[2].shape)
        features_sp[species] = np.hstack( [features_sp_block[block] for block in feature_hypers["nus"]] )
        #print("features_sp[species].shape",features_sp[species].shape)
        #print("features_sp[species].shape",features_sp[species].shape)
    
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

def compute_hilbert_space_features(feature_hypers, frames, target, train_idx, center_atom_id_mask):
    computation_type = feature_hypers["hilbert_space_parameters"]["computation_type"]
    if computation_type == "implicit_distance":
        features = compute_features_from_kernel(standardize_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames, target, train_idx, center_atom_id_mask), feature_hypers["hilbert_space_parameters"]["kernel_parameters"]), train_idx))
    elif computation_type == "sparse_implicit_distance":
        features = compute_sparse_features_from_kernel(standardize_kernel(compute_kernel_from_squared_distance(compute_squared_distance(feature_hypers, frames, target, train_idx, center_atom_id_mask), feature_hypers["hilbert_space_parameters"]["kernel_parameters"]), train_idx))
    elif computation_type == "implicit_kernel":
        features = compute_features_from_kernel(standardize_kernel(compute_kernel(feature_hypers, frames, target, train_idx, center_atom_id_mask), train_idx))
    elif computation_type == "explicit":
        assert target in ['Atom'] # Structure not yet implemented
        features = standardize_features(compute_explicit_features(feature_hypers, frames, train_idx, center_atom_id_mask), train_idx)
    else:
        raise ValueError("The computation_type=" + computation_type + " is not known.")
    return features

def compute_kernel(feature_hypers, frames, target, train_idx, center_atom_id_mask):
    features = compute_representation(feature_hypers, frames, train_idx, center_atom_id_mask)
    if target == "Structure":
        features = compute_structure_features_from_atom_features(features, center_atom_id_mask)
    features = standardize_features(features, train_idx)
    kernel_parameters = feature_hypers["hilbert_space_parameters"]["kernel_parameters"]
    kernel_type = kernel_parameters["kernel_type"]
    if kernel_type == "polynomial":
        return (kernel_parameters["c"] + features.dot(features.T))**kernel_parameters["degree"]


def compute_explicit_features(feature_hypers, frames, target, train_idx, center_atom_id_mask):
    features = compute_representation(feature_hypers, frames, train_idx, center_atom_id_mask)
    if target == "Structure":
        features = compute_structure_features_from_atom_features(features, center_atom_id_mask)
    kernel_parameters = feature_hypers["hilbert_space_parameters"]["kernel_parameters"]
    kernel_type = kernel_parameters["kernel_type"]
    if kernel_type == "polynomial":
        return standardize_features(compute_explicit_polynomial_features(features, kernel_parameters["degree"]), train_idx)
    else:
        raise ValueError("The kernel_type=" + kernel_type + " is not known.")

def compute_explicit_polynomial_features(features, degree):
    # Based on https://en.wikipedia.org/wiki/Multinomial_theorem#Theorem 
    # degree = n
    # nb_features = m 
    features = np.hstack( (features, np.ones(len(features))[:,np.newaxis]) )
    nb_features = features.shape[1]
    # https://en.wikipedia.org/wiki/Multinomial_theorem#Number_of_multinomial_coefficients
    nb_polynomial_features = int(scipy.special.comb(degree+nb_features-1, nb_features-1))
    polynomial_features = np.zeros( (features.shape[0], nb_polynomial_features) )
    #from sympy.ntheory import multinomial_coefficients
    #from sympy.ntheory.multinomial import multinomial_coefficients
    from sympy.ntheory.multinomial import multinomial_coefficients_iterator
    i = 0
    print(f"Computation of explicit features nb_features={nb_features} degree={degree}...", flush=True)
    for k_indices, multinomial_coeff in multinomial_coefficients_iterator(nb_features, degree): 
        polynomial_features[:,i] = np.sqrt(multinomial_coeff)
        for t in range(len(k_indices)):
            polynomial_features[:,i] *= features[:,t]**k_indices[t]
        i += 1
    print("Computation of explicit features finished", flush=True)
    return polynomial_features


def compute_squared_distance(feature_hypers, frames, target, train_idx, center_atom_id_mask):
    print("Compute distance.")
    distance_type = feature_hypers["hilbert_space_parameters"]["distance_parameters"]["distance_type"]
    if distance_type == "euclidean":
        features = compute_representation(feature_hypers, frames, train_idx, center_atom_id_mask)
    elif distance_type == "wasserstein":
        raise ValueError("The distance_type='wasserstein' is not fully implemented yet.")
        features = compute_squared_wasserstein_distance(feature_hypers, frames)
    else:
        raise ValueError("The distance_type='" + distance_type + "' is not known.")
    if target == "Structure":
        features = compute_structure_features_from_atom_features(features, center_atom_id_mask)
    features = standardize_features(features, train_idx)
    # D(A,B)**2 = K(A,A) + K(B,B) - 2*K(A,B)
    return np.sum(features ** 2, axis=1)[:, np.newaxis] + np.sum(features ** 2, axis=1)[np.newaxis, :] - 2 * features.dot(features.T)


def compute_features_from_kernel(kernel):
    print("Compute features from kernel...", flush=True)
    # SVD is more numerical stable than eigh
    #U, s, _ = scipy.linalg.svd(kernel)
    #return np.flip(U, axis=1).dot(np.diag(np.flip(s)))
    # reorder eigvals and eigvectors such that largest eigvenvectors and eigvals start in the fist column
    d, A = scipy.linalg.eigh(kernel)
    print("Compute features from kernel finished.", flush=True)
    if np.min(d) < 0:
        print('Warning: Negative eigenvalue encountered ',np.min(d),' If small value, it could be numerical error', flush=True)
        d[d < 0] = 0
    return A.dot(np.diag(np.sqrt(d)))

def compute_sparse_features_from_kernel(kernel):
    print("Compute features from kernel...", flush=True)
    from sklearn.utils.extmath import randomized_svd
    from sklearn.utils.validation import check_random_state
    i = 0
    total_explained_variance_ratio_  = 0
    # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/decomposition/_pca.py#L552-L556
    # this is the original total variance estimation of the code
    #total_var = np.var(kernel, ddof=1, axis=0)
    # but it is only accurate if features << samples which is not the case for the kernel
    # therefore we estimate a lower bound by taking the currently smallest eigval and
    n_samples = len(kernel)
    while(total_explained_variance_ratio_ < 0.99):
        random_state = check_random_state(None)
        iterated_power='auto'
        # TODO the 1000+(500*i) is hacked for experiments, make this hyperparameters
        U, S, _ = randomized_svd(kernel, n_components=min(1000+(500*i),len(kernel)),
                                 n_iter=iterated_power,
                                 flip_sign=True,
                                 random_state=random_state)
        # altered copy from
        # https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/decomposition/_pca.py#L552-L556
        # this is the original total variance estimation of the code
        #total_var = np.var(kernel, ddof=1, axis=0).sum()
        # but it is only accurate if features << samples which is not the case for the kernel
        # therefore we estimate a lower bound by taking the currently smallest eigval and
        total_var = ( np.sum(S**2) + (n_samples-len(S))* np.min(S)**2 ) / (n_samples - 1)
        explained_variance_ = (S ** 2) / (n_samples - 1)
        total_explained_variance_ratio_ = np.sum(explained_variance_ / total_var)
        i += 1
    print("number of kernel features:", 1000*500*(i-1), flush=True)

    print("Compute features from kernel finished.", flush=True)
    # if one wants to sort features according to the eigvals descending
    #idx = np.argsort(S)[::-1]
    #U = U[:,idx]
    #S = S[idx]
    return U.dot(np.diag(np.sqrt(S)))

# use this implementation to check for negative eigenvalues for debugging
# def compute_features_from_kernel(kernel):
#    d, U = scipy.linalg.eigh(kernel)
#    if np.min(d) < 0:
#        print('Negative eigenvalue encounterd',np.min(d))
#        d[d < 0] = 0
#    # flip such that largest eigvals are on the left
#    return np.flip(U, axis=1).dot(np.diag(np.sqrt(np.flip(d))))


def compute_kernel_from_squared_distance(squared_distance, kernel_parameters):
    print("Compute kernel...", flush=True)
    kernel_type = kernel_parameters["kernel_type"]
    if kernel_type == "center":
        H = np.eye(len(squared_distance)) - np.ones((len(squared_distance), len(squared_distance))) / len(squared_distance)
        return -H.dot(squared_distance).dot(H) / 2
    elif kernel_type == "polynomial":
        H = np.eye(len(squared_distance)) - np.ones((len(squared_distance), len(squared_distance))) / len(squared_distance)
        return (kernel_parameters["c"] + (-H.dot(squared_distance).dot(H)/2 * kernel_parameters["gamma"]))**kernel_parameters["degree"]
    elif kernel_type == "negative_distance":
        return -squared_distance ** (kernel_parameters["degree"] / 2)
    elif kernel_type == "rbf":
        kernel = np.exp(-kernel_parameters["gamma"] * squared_distance)
        return kernel
    elif kernel_type == "laplacian":
        return np.exp(-kernel_parameters["gamma"] * np.sqrt(squared_distance))
    else:
        raise ValueError("The kernel_type=" + kernel_type + " is not known.")

# TODO deprecated
def load_hydrogen_distance_dataset(frames):
    return np.array([frame.info["hydrogen_distance"] for frame in frames])[:,np.newaxis]
