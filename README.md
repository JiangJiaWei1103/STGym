# STGym: A Modular Benchmark for Spatio-Temporal Networks

STGym is a framework designed for the development, exploration, and evaluation of spatio-temporal networks. The modular design enhances understanding of model composition while enabling seamless adoption and extension of existing methods. It ensures reproducible and scalable experiments through a standardized training and evaluation pipeline, promoting fair comparisons across models.

## Key Features

<details>
  <summary><b>Modular Design</b></summary>
  Effortlessly explore various model compositions while facilitating the seamless adoption and extension of existing methods.
</details>

<details>
  <summary><b>Standardized Pipelines</b></summary>
  Guarantee reproducibility, scalability, and fair comparisons across models and datasets.
</details>

<details>
  <summary><b>Comprehensive Benchmarking</b></summary>
  Includes 16 models evaluated across six widely used traffic forecasting datasets.
</details>

<details>
  <summary><b>Flexible Configuration</b></summary>
  Uses Hydra for dynamic configuration, enabling easy overrides from the command line and speeding up experimentation without needing multiple config files.
</details>

<details>
  <summary><b>Automatic Tracking & Logging</b></summary>
  Seamlessly integrates with Weights & Biases for efficient tracking, logging, and recording of experiment results.
</details>

## Built-in Datasets and Baselines

### Datasets
* METR-LA, PEMS-BAY, PEMS03, PEMS04, PEMS07, PEMS08
* Electricity, Solar Energy, Traffic, Exchange Rate

### Baselines
* LSTM, TCN
* DCRNN, STGCN, GWNet, STSGCN, AGCRN, GMAN, MTGNN, DGCRN, GTS, STNorm, STID, SCINet, STAEformer, MegaCRN
* LST-Skip, TPA-LSTM, Linear, NLinear, DLinear

## How to Run
### 1. Installing Dependencies
#### Python
Python 3.7 or higher is required.
#### PyTorch
Install PyTorch according to your Python version.
#### Other Dependencies
```
pip install -r requirements.txt
```
#### Example Setups
Example 1: Python 3.8 + PyTorch 1.13.1 + CUDA 11.6
```
conda create -n STGym python=3.8
conda activate STGym
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
Example 2: Python 3.11 + PyTorch 2.4.0 + CUDA 12.4
```
conda create -n STGym python=3.11
conda activate STGym
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### 2. Downloading Datasets
Download the `raw.zip` file from [Google Drive](https://drive.google.com/file/d/1-C8E9bJNbqAqjJF97LUpFRWQRG8n5g8p/view?usp=share_link). Unzip the files to the `./data/` directory.
```
cd /path/to/STGym
unzip /path/to/raw.zip -d data/
```

### 3. Train & Evaluate Model
#### Train Your Own Model
##### 1. **Define Your Model**
Implement your model and place it in the `./modeling/sotas` directory.
##### 2. **Define Model Configuration**
Create a configuration file `.yaml` to set the model parameters and place it in the `./config/model` directory. With hydra-based configuration system, you can override various configuration setup, such as scaler, optimizer, loss, and other hyperparameters, directly from the command line. Default settings can be found in the `./config/` directory.
##### 3. **Train & Evaluate Your Own Model**
```
python -m tools.main model=<MODEL_NAME> data=<DATASET_NAME>
```
Replace `<MODEL_NAME>` with your model name and `<DATASET_NAME>` with any supported dataset.
#### Reproducing Built-in Models
```
python -m tools.run_sotas model=<MODEL_NAME> data=<DATASET_NAME>
```
Replace `<MODEL_NAME>` and `<DATASET_NAME>` with any supported model and dataset.