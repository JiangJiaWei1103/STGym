"""
Project-specific metadata for global access.
Author: JiaWei Jiang
"""
MODEL_NAME = "HARDPurG"

MTSFBerks = ["electricity", "solar_energy", "traffic", "exchange_rate"]
TrafBerks = ["metr_la", "pems_bay", "pems03", "pems04", "pems07", "pems08"]
N_SERIES = {
    "electricity": 321,
    "traffic": 862,
    "solar_energy": 137,
    "exchange_rate": 8,
    "metr_la": 207,
    "pems_bay": 325,
    "pems03": 358,
    "pems04": 307,
    "pems07": 883,
    "pems08": 170,
}

N_DAYS_IN_WEEK = 7
