# coding: utf-8
import numpy as np
from ase import neighborlist


def compute_sorted_distances(feature_parameters, frames, center_atom_id_mask):
    print("Warning: Sorted distances only works for one species")
    cutoff = feature_parameters["interaction_cutoff"]

    #white_ids = np.array(center_atom_id_mask)
    #white_ids[1:] += np.array([len(env_ids) for env_ids in center_atom_id_mask])[:-1,np.newaxis]
    #white_ids = white_ids.flatten()

    max_neighbors = 0
    for frame_id in range(len(frames)):
        frame = frames[frame_id]
        atom_i = neighborlist.neighbor_list('i', frame, cutoff)
        for atom_id in center_atom_id_mask[frame_id]:
            max_neighbors = max(max_neighbors, np.max( np.sum(atom_i == atom_id) ))

    padding_type = feature_parameters["padding_type"]
    if padding_type == "max":
        max_distance = 0
        for frame_id in range(len(frames)):
            frame = frames[frame_id]
            atom_i, distances = neighborlist.neighbor_list('id', frame, cutoff)
            for atom_id in center_atom_id_mask[frame_id]:
                max_neighbors = max(max_neighbors, np.max(distances[atom_i == atom_id]))
        padding = max_distance
    elif padding_type == "zero":
        padding = 0
    else:
        raise("Error padding_type "+padding_type+" is not available.")

    sorted_distances = np.ones( (sum([len(env_idx) for env_idx in center_atom_id_mask]), max_neighbors) ) * padding

    print(max_neighbors)
    k = 0
    for frame_id in range(len(frames)):
        frame = frames[frame_id]
        atom_i, distances = neighborlist.neighbor_list('id', frame, cutoff)
        for atom_id in center_atom_id_mask[frame_id]:
            # solution to extract distances from structure to env assumes atom_i is sorted
            sorted_distances_env = np.sort(distances[atom_i == atom_id])
            sorted_distances[k, :len(sorted_distances_env)] = sorted_distances_env
        k += 1
    return sorted_distances

#        per_atom = True
#        environments = sum((len(frame) for frame in frames), 0)
#
#        all_distances = np.zeros((environments, len(species_pairs), max_neighbors))
#        actual_max_neighbors = 0
#        current_center = 0
#        ave_neighbors = 0
#        ave_neighbors_count = 0
#        for frame_id in white_ids:
#            frame = frames[frame_id]
#            atom_i, atom_j, distances = neighborlist.neighbor_list('ijd', frame, cutoff)
#            idx = neighborlist.first_neighbors(len(frame), atom_i)
#
#            for i in range(len(frame)):
#                # fill with cutoff to be ensure when sorting it is at the end
#                # This will later result in that the cutoff is strict
#                tmp_distances = np.full((len(species_pairs), max_neighbors), cutoff)
#
#                start = idx[i]
#                stop = idx[i + 1]
#
#                neighbors_count = [0] * len(species_pairs)
#
#                for j, d in zip(atom_j[start:stop], distances[start:stop]):
#                    pair = sort_pair(frame[i].symbol, frame[j].symbol)
#                    pair_id = species_pairs[pair]
#
#                    tmp_distances[pair_id, neighbors_count[pair_id]] = d
#                    neighbors_count[pair_id] += 1
#
#                actual_max_neighbors = max(actual_max_neighbors, max(neighbors_count))
#
#                # if padding 0 then no radial scaling
#                for k in range(len(species_pairs)):
#                    # fill with zeros if the pair was NOT found
#                    if tmp_distances[k, 0] == cutoff:
#                        tmp_distances[k, :] = 0
#                    else:
#                        tmp_distances[k, :].sort()
#                        ave_neighbors += neighbors_count[k]
#                        ave_neighbors_count += 1
#
#                padding_idx = tmp_distances == cutoff
#                tmp_distances[padding_idx] = padding
#
#                pair = sort_pair(frame[i].symbol, frame[i].symbol)
#                pair_id = species_pairs[pair]
#
#                all_distances[current_center, :, :] += tmp_distances
#
#                if per_atom:
#                    current_center += 1
#
#            if not per_atom:
#                all_distances[current_center, :, :] /= len(frame)
#                current_center += 1
#
#        # TODO: all_distances is very sparse. use sparse matrix storage?
#        #
#        # a = all_distances.reshape((environments, len(all_pairs) * self.max_neighbors))
#        # import matplotlib.pyplot as plt
#        # plt.spy(a)
#        # plt.show()
#
#        ave_neighbors /= ave_neighbors_count
#        assert current_center == environments
#        # print(f'actual_max_neighbors={actual_max_neighbors}')
#        return all_distances
