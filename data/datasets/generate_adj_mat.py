"""
Script for generating adjacency matrix.
Author: ChunWei Shen
"""
from typing import Tuple

import csv
import pickle
import numpy as np
from metadata import N_SERIES

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
    
def gen_and_save_adj(dataset_name: str):
    """
    Args:
        dataset_name: dataset name
    """
    print(f"Generating {dataset_name} Adjacency Matrix ...")

    if dataset_name == "pems03":
        sensor_ids_filepath = f"./data/raw/{dataset_name}/{dataset_name}.txt"
    else:
        sensor_ids_filepath = None
    
    binary_adj, dist_adj = gen_binary_adj_mat(
        distance_df_filepath=f"./data/raw/{dataset_name}/{dataset_name}.csv", 
        n_series=N_SERIES[dataset_name], 
        sensor_ids_filepath=sensor_ids_filepath,
    )

    with open(f"./data/raw/{dataset_name}/{dataset_name}_adj.pkl", "wb") as f:
        pickle.dump(binary_adj, f, protocol=2)

    print("Finish!")

    return

if __name__ == "__main__":
    gen_and_save_adj(dataset_name="pems03")
    gen_and_save_adj(dataset_name="pems04")
    gen_and_save_adj(dataset_name="pems07")
    gen_and_save_adj(dataset_name="pems08")