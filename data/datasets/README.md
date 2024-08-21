1. METR-LA
* [Traffic data link](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
    * `metr-la.h5`
* [Graph data link](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)
    * `graph_sensor_ids.txt`
    * `distances_la_2012.csv`

```
python -m data.datasets.generate_adj_mat --dataset_name metr_la --sensor_ids_filepath ./data/raw/metr_la/graph_sensor_ids.txt
--distances_filepath ./data/raw/metr_la/distances_la_2012.csv --output_pkl_filepath ./data/raw/metr_la/metr_la_adj.pkl
```

2. PEMS-BAY
* [Traffic data link](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
    * `pems-bay.h5`
* [Graph data link](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)
    * `adj_mx_bay.pkl`

3. PEMS03
* [Data link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS03)
    * `PEMS03.npz`
    * `PEMS03.csv`
    * `PEMS03.txt`

```
python -m data.datasets.generate_adj_mat --dataset_name pems03 --sensor_ids_filepath ./data/raw/pems03/PEMS03.txt 
--distances_filepath ./data/raw/pems03/PEMS03.csv --output_pkl_filepath ./data/raw/pems03/pems03_adj.pkl
```

4. PEMS04
* [Data link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS04)
    * `PEMS04.npz`
    * `PEMS04.csv`
    * `PEMS04.txt`

```
python -m data.datasets.generate_adj_mat --dataset_name pems04 --sensor_ids_filepath ./data/raw/pems04/PEMS04.txt 
--distances_filepath ./data/raw/pems04/PEMS04.csv --output_pkl_filepath ./data/raw/pems04/pems04_adj.pkl
```

5. PEMS07
* [Data link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS07)
    * `PEMS07.npz`
    * `PEMS07.csv`
    * `PEMS07.txt`

```
python -m data.datasets.generate_adj_mat --dataset_name pems07 --sensor_ids_filepath ./data/raw/pems07/PEMS07.txt 
--distances_filepath ./data/raw/pems07/PEMS07.csv --output_pkl_filepath ./data/raw/pems07/pems07_adj.pkl
```

6. PEMS08
* [Data link](https://github.com/guoshnBJTU/ASTGNN/tree/main/data/PEMS08)
    * `PEMS08.npz`
    * `PEMS08.csv`
    * `PEMS08.txt`

```
python -m data.datasets.generate_adj_mat --dataset_name pems08 --sensor_ids_filepath ./data/raw/pems08/PEMS08.txt 
--distances_filepath ./data/raw/pems08/PEMS08.csv --output_pkl_filepath ./data/raw/pems08/pems08_adj.pkl
```