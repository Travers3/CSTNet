from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
import pickle
import sys


def get_adjacency_matrix(distance_df, sensor_ids):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            print(row[0])
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = 1
    dist_mx = dist_mx.astype(np.float64)
    return sensor_ids, sensor_id_to_ind, dist_mx


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_ids_filename', type=str, default='graph_grid_ids.txt',
                        help='File containing sensor ids separated by comma.')
    parser.add_argument('--distances_filename', type=str, default='distance.csv',
                        help='CSV file containing sensor distances with three columns: [from, to, distance].')
    parser.add_argument('--output_pkl_filename', type=str, default='adj_mat.pkl',
                        help='Path of the output file.')
    args = parser.parse_args()

    from_to_df = pd.read_csv('from_to_data.csv',dtype={'from': 'int', 'to': 'int'})
    distance_df = from_to_df[['from','to']]
    sensor_ids = np.loadtxt(args.sensor_ids_filename,delimiter=',')
    sensor_ids=np.int32(sensor_ids)

    # distance_df = pd.read_csv(args.distances_filename, dtype={'from': 'int', 'to': 'int'})
    _, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)

    #Save to pickle file.
  
    np.save("adj_mat.npy",adj_mx)
    with open(args.output_pkl_filename, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f, protocol=2)