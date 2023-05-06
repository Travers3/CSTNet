import torch
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from tqdm import tqdm

import logging
from math import radians, cos, sin, asin, atan2, sqrt, degrees

logger = logging.getLogger()

def haversine_distance(lon1, lat1, lon2, lat2):
    """[This function is used to calculate the distance between two GPS points (unit: meter)]

    Args:
        lon1 ([float]): [longitude 1]
        lat1 ([float]): [latitude 1]
        lon2 ([float]): [longitude 2]
        lat2 ([float]): [latitude 2]

    Returns:
        [type]: [Distance between two points]
    """
    lon1, lat1, lon2, lat2 = map(
        radians, [lon1, lat1, lon2, lat2])  # Convert decimal degrees to radians
    c = 2 * asin(sqrt(sin((lat2 - lat1)/2)**2 + cos(lat1) *
                 cos(lat2) * sin((lon2 - lon1)/2)**2))  # Haversine formula
    r = 6371.393  # Average radius of the earth in kilometers
    return c * r * 1000

def conut_gird_num(tracks_data, grid_distance):
    """[This function is used to generate the number of lattice length and width according to the given lattice size]

    Args:
        tracks_data ([object]): [Original timectory data]
        grid_distance ([int]): [Division distance]

    Returns:
        [type]: [description]
    """
    Lon1 = tracks_data['Lon'].min()
    Lat1 = tracks_data['Lat'].min()
    Lon2 = tracks_data['Lon'].max()
    Lat2 = tracks_data['Lat'].max()
    low = haversine_distance(Lon1, Lat1, Lon2, Lat1)
    high = haversine_distance(Lon1, Lat2, Lon2, Lat2)
    left = haversine_distance(Lon1, Lat1, Lon1, Lat2)
    right = haversine_distance(Lon2, Lat1, Lon2, Lat2)
    lon_grid_num = int((low + high) / 2 / grid_distance)
    lat_grid_num = int((left + right) / 2 / grid_distance)
    logger.info("After division, the whole map is:", lon_grid_num, '*',
          lat_grid_num, '=', lon_grid_num * lat_grid_num, 'grids')
    return lon_grid_num, lat_grid_num, Lon1, Lat1, Lon2, Lat2

def grid_process(tracks_data, grid_distance,is_delete):
    """[This function is used to map each GPS point to a fixed grid]

    Args:
        tracks_data ([type]): [description]
        grid_distance ([type]): [description]

    Returns:
        [type]: [description]
    """


    lon_grid_num, lat_grid_num, Lon1, Lat1, Lon2, Lat2 = conut_gird_num(
        tracks_data, grid_distance)
    Lon_gap = (Lon2 - Lon1)/lon_grid_num
    Lat_gap = (Lat2 - Lat1)/lat_grid_num
    # Get the two-dimensional matrix coordinate index and convert it to one-dimensional ID
    tracks_data['grid_ID'] = tracks_data.apply(lambda x: int(
        (x['Lat']-Lat1)/Lat_gap) * lon_grid_num + int((x['Lon']-Lon1)/Lon_gap) + 1, axis=1)

    def delete_columns_num_less(source_data, column_name):
        d = pd.DataFrame(source_data.grid_ID.value_counts())
        d.columns = ['nums']

        # 出现一次的全部删除
        d = d[d['nums'] < 10]
        delindexs = d.index
        print(len(delindexs))

        #找到待删除数据的index 主数据里也要删除
        delete_data = source_data[source_data[column_name].isin(delindexs)]
        delete_index_list = delete_data.index.to_list()
        l = source_data.shape[0]//2
        for index,val in enumerate(delete_index_list):
            if val >= l:
                delete_index_list[index] = val - l
        num_set = set(delete_index_list)

        source_data = source_data[~source_data[column_name].isin(delindexs)]
        grid_list = sorted(set(source_data[column_name]))

        source_data[column_name] = [grid_list.index(
            num) for num in tqdm(source_data[column_name])]

        return source_data, num_set

    if is_delete:
        tracks_data, num_set = delete_columns_num_less(tracks_data,'grid_ID')
        grid_list = sorted(set(tracks_data['grid_ID']))

        tracks_data['grid_ID'] = [grid_list.index(
            num) for num in tqdm(tracks_data['grid_ID'])]
        grid_list = sorted(set(tracks_data['grid_ID']))
        logger.info('After removing the invalid grid, there are', len(grid_list), 'grids')
        return tracks_data, grid_list, num_set

    else:
        grid_list = sorted(set(tracks_data['grid_ID']))

        tracks_data['grid_ID'] = [grid_list.index(
            num) for num in tqdm(tracks_data['grid_ID'])]
        grid_list = sorted(set(tracks_data['grid_ID']))
        logger.info('After removing the invalid grid, there are', len(grid_list), 'grids')
        return tracks_data, grid_list

def generate_dataset(tracks_data, split_ratio):
    """[This function is used to generate data set, train set and test set]

    Args:
        tracks_data ([object]): [timectory data after discretization 3ws 3ws grid]
        split_ratio ([float]): [split ratio]

    Returns:
        [type]: [Track list, user list, data set, training set and test set, number of test sets]
    """
    user_list = tracks_data['ObjectID'].drop_duplicates().values.tolist()
    user_time_dict = {key: [] for key in user_list}

    for user_id in tqdm(tracks_data['ObjectID'].drop_duplicates().values.tolist()):
        one_user_data = tracks_data.loc[tracks_data.ObjectID == user_id, :]
        for time_id in one_user_data['time'].drop_duplicates().values.tolist():
            # one_time_data = one_user_data.lo  c[tracks_data.timNumber == time_id, 'grid_ID'].drop_duplicates().values.tolist()
            one_time_data = one_user_data.loc[tracks_data.time ==
                                              time_id, 'grid_ID'].values.tolist()
            user_time_dict[user_id].append(
                (time_id, one_time_data))
    time_list = list(range(time_id+1))

    test_nums = 0
    user_time_train, user_time_test = {
        key: [] for key in user_list}, {key: [] for key in user_list}
    for key in user_time_dict:
        time_num = len(user_time_dict[key])
        test_nums += time_num - int(time_num*split_ratio)
        for idx in list(range(time_num))[:int(time_num*split_ratio)]:
            user_time_train[key].append(user_time_dict[key][idx])
        for idx in list(range(time_num))[int(time_num*split_ratio):]:
            user_time_test[key].append(user_time_dict[key][idx])

    return time_list, user_list, user_time_dict, user_time_train, user_time_test, test_nums
def preprocess_adj(adj):
    """[A^~ = A + I]

    Args:
        adj ([type]): [adjacency matrix]

    Returns:
        [type]: [A^~]
    """
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def normalize_adj(adj):
    """[Fourier transform]

    Args:
        adj ([type]): [adjacency matrix A^~]

    Returns:
        [type]: [Matrix after Fourier transform]
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))  # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()  # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)  # D^-0.5
    # D^-0.5AD^0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """[Convert the matrix into sparse matrix and save it]

    Args:
        sparse_mx ([type]): [description]

    Returns:
        [type]: [sparse matrix]
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def generate_graph(grid_list, time_list, user_list, user_time_dict, user_time_train):
    """[This function is used to generate local graph, local graph node feature, global graph, global graph node feature]

    Args:
        grid_list ([list]): [grid list]
        time_list ([list]): [timectory list]
        user_list ([list]): [user list]
        user_time_dict ([dict]): [all timectory data]
        user_time_train ([dict]): [timectory data in training set]

    Returns:
        [type]: [local_feature, local_adj , local_feature, local_adj]
    """
    local_feature = np.eye(len(grid_list))

    local_graph = nx.Graph()
    local_graph.add_nodes_from(grid_list)
    local_edge_dict, local_edge_list = {}, []
    for key in user_time_dict:
        for one_time in user_time_dict[key]:
            one_time_list = one_time[1]
            for idx in range(1, len(one_time_list)):
                node1, node2 = sorted(
                    [one_time_list[idx-1], one_time_list[idx]])
                # if node1 != node2:
                if node1 != node2:
                    edge = str(node1) + ' ' + str(node2)
                    if edge not in local_edge_dict:
                        local_edge_dict[edge] = 1
                    else:
                        local_edge_dict[edge] += 1
    for key in local_edge_dict:
        local_edge_list.append(
            list(map(int, key.split()))+[local_edge_dict[key]])
    local_graph.add_weighted_edges_from(local_edge_list)
    local_adj = sparse_mx_to_torch_sparse_tensor(preprocess_adj(
        nx.to_scipy_sparse_matrix(local_graph, dtype=np.float32)))

    return torch.FloatTensor(local_feature), local_adj


def get_data_and_graph(raw_path,grid_size):
    """[Functions for processing data and generating local and global graphs]

    Args:
        raw_path ([str]): [Path of data file to be processed]
        read_pkl ([bool]): [If the value is false, the preprocessed data will be saved for direct use next time]
        grid_size ([type]): [Size of a single grid]

    Returns:
        [type]: [Processed timectory data and graphs data]
    """
    grid_distance = grid_size
    split_ratio = 0.6

    tracks_data = pd.read_csv(raw_path)
    tracks_data, grid_list = grid_process(tracks_data, grid_distance)

    time_list, user_list, user_time_dict, user_time_train, user_time_test, test_nums = generate_dataset(
        tracks_data, split_ratio)
    local_feature, local_adj = generate_graph(
        grid_list, time_list, user_list, user_time_dict, user_time_train)
    grid_nums, time_nums, user_nums = len(
        grid_list), len(time_list), len(user_list)

    return local_feature, local_adj, user_time_train, user_time_test, grid_nums, time_nums, user_nums, test_nums
