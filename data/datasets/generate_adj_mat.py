"""
Script for generating adjacency matrix.
Author: ChunWei Shen
"""
from typing import List, Dict, Tuple

import csv
import pickle
import argparse
import numpy as np
import pandas as pd
from metadata import N_SERIES

def gen_distance_adj_mat(
    distance_df: pd.DataFrame, sensor_ids: List[str], normalized_k: float = 0.1,
) -> Tuple[List[str], Dict[str, int], np.ndarray]:
    """
    Args:
        distance_df: data frame with three columns: [from, to, distance]
        sensor_ids: list of sensor ids
        normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity
    """
    n_sensors = len(sensor_ids)
    dist_mat = np.zeros((n_sensors, n_sensors), dtype=np.float32)
    dist_mat[:] = np.inf

    # Builds sensor id to index map
    sensor_id_to_idx = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_idx[sensor_id] = i

    # Fills cells in the matrix with distances
    for row in distance_df.values:
        if row[0] not in sensor_id_to_idx or row[1] not in sensor_id_to_idx:
            continue
        dist_mat[sensor_id_to_idx[row[0]], sensor_id_to_idx[row[1]]] = row[2]

    # Calculates the standard deviation as theta
    distances = dist_mat[~np.isinf(dist_mat)].flatten()
    std = distances.std()
    adj_mat = np.exp(-np.square(dist_mat / std))

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity
    adj_mat[adj_mat < normalized_k] = 0

    return sensor_ids, sensor_id_to_idx, adj_mat

def gen_binary_adj_mat(
    distance_df_filepath: str, n_series: int, sensor_ids_filepath: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        distance_df_filepath: path of the csv file contains edges information
        n_series: number of series
        sensor_ids_filepath: path of the txt file contains list of sensor ids
    """
    binary_adj = np.zeros((int(n_series), int(n_series)), dtype=np.float32)
    dist_adj = np.zeros((int(n_series), int(n_series)), dtype=np.float32)

    # Builds sensor id to index map
    if sensor_ids_filepath:
        with open(sensor_ids_filepath, "r") as f:
            sensor_id_to_idx = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

        with open(distance_df_filepath, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                binary_adj[sensor_id_to_idx[i], sensor_id_to_idx[j]] = 1
                binary_adj[sensor_id_to_idx[j], sensor_id_to_idx[i]] = 1
                dist_adj[sensor_id_to_idx[i], sensor_id_to_idx[j]] = distance
                dist_adj[sensor_id_to_idx[j], sensor_id_to_idx[i]] = distance

        return binary_adj, dist_adj

    else:
        with open(distance_df_filepath, "r") as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                binary_adj[i, j] = 1
                binary_adj[j, i] = 1
                dist_adj[i, j] = distance
                dist_adj[j, i] = distance

        return binary_adj, dist_adj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="metr_la")
    parser.add_argument("--sensor_ids_filepath", type=str, default="./data/raw/metr_la/graph_sensor_ids.txt", 
                        help="file containing sensor ids separated by comma")
    parser.add_argument("--distances_filepath", type=str, default="./data/raw/metr_la/distances_la_2012.csv", 
                        help="csv file containing sensor distances")
    parser.add_argument("--normalized_k", type=float, default=0.1, help="normalization for sparsity")
    parser.add_argument("--output_pkl_filepath", type=str, default="./data/raw/metr_la/metr_la_adj_test.pkl", 
                        help="path of the output file")
    args = parser.parse_args()

    if args.dataset_name == "metr_la":
        with open(args.sensor_ids_filepath) as f:
            sensor_ids = f.read().strip().split(",")
        distance_df = pd.read_csv(args.distances_filepath, dtype={"from": "str", "to": "str"})
        normalized_k = args.normalized_k
        _, sensor_id_to_ind, adj_mat = gen_distance_adj_mat(distance_df, sensor_ids, normalized_k)
        # Save to pickle file
        with open(args.output_pkl_filepath, "wb") as f:
            pickle.dump(adj_mat, f, protocol=2)
    elif args.dataset_name in ["pems03", "pems04", "pems07", "pems08"]:
        n_series = N_SERIES[args.dataset_name]
        binary_adj, dist_adj = gen_binary_adj_mat(args.distances_filepath, n_series, args.sensor_ids_filepath)
        # Save to pickle file
        with open(args.output_pkl_filepath, "wb") as f:
            pickle.dump(binary_adj, f, protocol=2)