"""
Data processor definitions.
Author: JiaWei Jiang

This file contains the definition of data processor cleaning and
processing raw data before entering modeling phase.
"""
import logging
import math
import os
import pickle
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import coo_matrix, csr_matrix
from torch import Tensor

from metadata import N_SERIES, MTSFBerks, TrafBerks
from paths import RAW_DATA_PATH
from utils.common import asym_norm, calculate_random_walk_matrix, calculate_scaled_laplacian, sym_norm
from utils.scaler import MaxScaler, MinMaxScaler, StandardScaler


class DataProcessor(object):
    """Data processor processing raw data, and providing access to
    processed data ready to be fed into modeling phase.

    Parameters:
       file_path: path of the raw data
           *Note: File reading supports .parquet extension in default
               setting, which can be modified to customized one.
       dp_cfg: hyperparameters of data processor
    """

    # https://stackoverflow.com/questions/59173744
    _df_holdout: Union[pd.DataFrame, np.ndarray]
    _df_test: Union[pd.DataFrame, np.ndarray]

    # The priori graph structure is optionally provided.
    _priori_adj_mat: Optional[List[Union[Tensor, coo_matrix, csr_matrix]]] = None

    def __init__(self, file_path: str, **dp_cfg: Any):
        # Setup data processor
        self._dp_cfg = dp_cfg
        self._df_holdout = None
        self._setup()

        # Load raw data
        if self.dataset_name in MTSFBerks:
            data_vals = np.loadtxt(file_path, delimiter=",")
            self._df = pd.DataFrame(data_vals)
        elif self.dataset_name in TrafBerks:
            if file_path.endswith("npz"):
                data_vals = np.load(file_path, allow_pickle=True)["data"][..., 0]
                self._df = pd.DataFrame(data_vals)
            else:
                self._df = pd.read_hdf(file_path)

    def run_before_splitting(self) -> None:
        """Clean and process data before data splitting (i.e., on raw
        static DataFrame).

        Return:
            None
        """
        logging.info("Run data cleaning and processing before data splitting...")

        # Holdout unseen test set
        if self.holdout_ratio != 0:
            self._holdout()

        # Initialize priori graph structure if provided
        if self.priori_gs["type"] is not None:
            self._init_priori_gs()

    def run_after_splitting(
        self,
        df_tr: Union[pd.DataFrame, np.ndarray],
        df_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Clean and process data after data splitting to avoid data
        leakage issue (e.g., scaling on the whole dataset).

        Parameters:
            df_tr: training data
            df_val: validation data

        Return:
            df_tr: processed training data
            df_val: processed validation data
            scaler: scaling object
        """
        logging.info("Run data cleaning and processing after data splitting...")
        scaler = None
        if self.scaling is not None:
            df_tr, df_val, scaler = self._scale(df_tr, df_val)

        return df_tr, df_val, scaler

    def get_df(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return raw or processed DataFrame for CV iteration."""
        return self._df

    def get_df_test(self) -> Union[pd.DataFrame, np.ndarray]:
        """Return unseen test set for final evaluation."""
        return self._df_test

    def get_priori_gs(self) -> Optional[List[Tensor]]:
        """Return priori graph structure."""
        return self._priori_adj_mat

    def _setup(self) -> None:
        """Retrieve all parameters specified to process data."""
        # Before data splitting
        self.dataset_name = self._dp_cfg["dataset_name"]
        self.holdout_ratio = self._dp_cfg["holdout_ratio"]

        # After data splitting
        self.scaling = self._dp_cfg["scaling"]
        self.priori_gs = self._dp_cfg["priori_gs"]

    def _holdout(self) -> None:
        """Holdout unseen test set before running CV iteration.

        `self._df_holdout` is the static raw DataFrame which can't be
        modified or transformed in-place. Because trafos (e.g., scaler)
        in different folds should transform the original holdout, not
        the one transformed by trafos in the previous folds.
        """
        holdout_size = math.floor(len(self._df) * self.holdout_ratio)
        cv_size = len(self._df) - holdout_size

        logging.info(f"Holdout unseen test with ratio {self.holdout_ratio}...")
        if isinstance(self._df, pd.DataFrame):
            self._df_holdout = self._df.iloc[-holdout_size:, :]
            self._df = self._df.iloc[:cv_size, :]
        else:
            self._df_holdout = self._df[-holdout_size:, :]
            self._df = self._df[:cv_size, :]

    def _init_priori_gs(self) -> None:
        """Initialize the priori graph structure.

        Ref: https://github.com/nnzhan/Graph-WaveNet/

        Return:
            None
        """
        n_series = N_SERIES[self.dataset_name]
        priori_gs_type = self.priori_gs["type"]
        if priori_gs_type == "identity":
            self._priori_adj_mat = [torch.eye(n_series)]
        else:
            adj_mat = self._load_adj_mat()
            assert n_series == adj_mat.shape[0], "#Series (i.e., #Nodes) should be aligned."

            if priori_gs_type == "sym_norm":
                self._priori_adj_mat = [sym_norm(adj_mat)]
            elif priori_gs_type == "transition":
                self._priori_adj_mat = [asym_norm(adj_mat)]
            elif priori_gs_type == "dbl_transition":
                self._priori_adj_mat = [asym_norm(adj_mat), asym_norm(adj_mat.T)]
            elif priori_gs_type == "laplacian":
                self._priori_adj_mat = [calculate_scaled_laplacian(adj_mat, lambda_max=None)]
            elif priori_gs_type == "random_walk":
                self._priori_adj_mat = [calculate_random_walk_matrix(adj_mat).T]
            elif priori_gs_type == "dual_random_walk":
                self._priori_adj_mat = [
                    calculate_random_walk_matrix(adj_mat).T,
                    calculate_random_walk_matrix(adj_mat.T).T,
                ]
            else:
                raise RuntimeError(f"Priori GS {priori_gs_type} isn't registered...")

        logging.info(f"Priori GS {priori_gs_type} has been set up!")

    def _load_adj_mat(self) -> np.ndarray:
        """Load hand-crafted adjacency matrix.

        Ref: https://github.com/nnzhan/Graph-WaveNet/
        """
        dataset = self.dataset_name
        adj_mat_file_path = os.path.join(RAW_DATA_PATH, dataset, f"{dataset}_adj.pkl")

        try:
            with open(adj_mat_file_path, "rb") as f:
                *_, adj_mat = pickle.load(f)
        except UnicodeDecodeError as e:
            with open(adj_mat_file_path, "rb") as f:
                *_, adj_mat = pickle.load(f, encoding="latin1")
        except Exception as e:
            logging.error("Fail to load the hand-crafted adjacency matrix...")
            logging.error("Err:", e)
            raise

        return adj_mat

    def _scale(
        self,
        df_tr: Union[pd.DataFrame, np.ndarray],
        df_val: Union[pd.DataFrame, np.ndarray],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Scale the data.

        Index of input DataFrame is retained to keep information about
        time steps.

        More scaling methods can be added.

        Parameters:
            df_tr: training data
            df_val: validation data

        Return:
            df_tr: scaled training data
            df_val: scaled validation data
            scaler: scaling object
        """
        scaler: Union[StandardScaler, MinMaxScaler, MaxScaler]
        if self.scaling == "standard":
            scaler = StandardScaler()
        elif self.scaling == "minmax":
            scaler = MinMaxScaler()
        elif self.scaling == "max":
            # See LSTNet
            scaler = MaxScaler()

        # Retain index information if any
        if isinstance(self._df, pd.DataFrame):
            df_tr_vals = df_tr.values
            df_val_vals = df_val.values
            df_tr_idx = df_tr.index
            df_val_idx = df_val.index
        else:
            df_tr_vals = df_tr
            df_val_vals = df_val

        # Scale data
        logging.info(f"Scale data using {self.scaling} scaler...")
        df_tr_vals = scaler.fit_transform(df_tr_vals)
        df_val_vals = scaler.transform(df_val_vals)
        df_tr = pd.DataFrame(df_tr_vals)
        df_val = pd.DataFrame(df_val_vals)
        if self._df_holdout is not None:
            # Scale holdout test set
            df_test_vals = scaler.transform(self._df_holdout)
            self._df_test = pd.DataFrame(df_test_vals)

        # Reassign index
        if isinstance(self._df, pd.DataFrame):
            df_tr.index = df_tr_idx
            df_val.index = df_val_idx
            self._df_test.index = self._df_holdout.index

        logging.info("Done.")

        return df_tr, df_val, scaler
