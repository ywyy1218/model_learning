# wjx
# 2023/11/17 10:07
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

def load_data(file_path):
    data = np.load(file_path).reshape(-1, 1)
    return data

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback]
        target = dataset[i + 1:i + lookback + 1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)



