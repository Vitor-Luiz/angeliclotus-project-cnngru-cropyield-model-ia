# Imports

import os
import numpy as np
import pandas as pd
import xarray as xr
import intake
import fsspec
import cftime
from xarray.coding.cftimeindex import CFTimeIndex
from collections import Counter
import gc

# Model Imports
import tensorflow as tf
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, ParameterGrid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, Concatenate, GRU, RepeatVector
)

# Functions
def normalization(ds):
    #ds = (ds - np.mean(ds)) / np.std(ds)
    ds = (ds - np.min(ds)) / (np.max(ds) - np.min(ds))
    return ds

def denormalize(normalized_data, original_data):
    """
    Denormalizes data that was normalized using the min-max scaling method.

    Parameters:
        normalized_data: numpy array or similar
            The normalized data (values typically between 0 and 1).
        original_data: numpy array or similar
            The original dataset used to compute min and max values during normalization.

    Returns:
        denormalized_data: numpy array or similar
            Data rescaled to its original range.
    """
    # Calculate the minimum and maximum values from the original dataset
    min_val = np.min(original_data)
    max_val = np.max(original_data)

    # Apply the inverse of the min-max normalization formula
    denormalized_data = (normalized_data * (max_val - min_val)) + min_val

    return denormalized_data