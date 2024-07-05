import numpy as np
import pandas as pd
from patsy import dmatrix

def generate_spline_matrices(Y, X):
    nan_indexes_Y = np.isnan(Y)
    nan_indexes = nan_indexes_Y
    nan_indexes_X = np.isnan(X.to_numpy()[:, 0])
    for i in range(X.shape[1]):
        X_ensemble = X.to_numpy()[:, i]
        nan_indexes_X = np.isnan(X_ensemble)
        nan_indexes = (nan_indexes) | (nan_indexes_Y) | nan_indexes_X

    X_clean = np.delete(X.to_numpy(), nan_indexes, axis=0)
    Y_clean = np.delete(Y, nan_indexes)

    if X_clean.shape[0] != len(Y_clean):
        N = min(X_clean.shape[0], len(Y_clean))
        X_clean = X_clean[:N,:]
        Y_clean = Y_clean[:N]
        

    X_spline_combined = None

    for i in range(X_clean.shape[1]):
        X_ensemble = X_clean[:, i]
        knots = np.linspace(np.min(X_ensemble), np.max(X_ensemble), 5)[1:-1]
        X_spline = dmatrix("bs(X_ensemble, degree=3, knots=knots, include_intercept=False)",
                           {"X_ensemble": X_ensemble, "knots": knots}, return_type='dataframe')

        if X_spline_combined is None:
            X_spline_combined = X_spline
        else:
            X_spline_combined = np.concatenate((X_spline_combined, X_spline), axis=1)

    return X_spline, X_spline_combined, X_clean, Y_clean



import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def load_variables():
    # Get a list of all saved variables
    saved_variables = os.listdir('loaded_variables')

    # Load each variable
    for var_name in saved_variables:
        # Get the variable name without the file extension
        # print(f'Loading {var_name}...')
        var_name_without_ext = os.path.splitext(var_name)[0]
        
        if var_name.endswith('.pkl'):
            # Load DataFrame or Series
            globals()[var_name_without_ext] = pd.read_pickle(f'loaded_variables/{var_name}')
        elif var_name.endswith('.npy'):
            # Load numpy array
            globals()[var_name_without_ext] = np.load(f'loaded_variables/{var_name}')

    # print('All variables loaded!')

import numpy as np
def set_n_smallest_to_zero(arr, n):
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return [0] * len(arr)
    
    # Find the nth smallest element
    nth_smallest = sorted(arr)[n-1]
    print(nth_smallest)
    
    # Set elements smaller than or equal to nth_smallest to zero
    modified_arr = [0 if x <= nth_smallest else x for x in arr]
    modified_arr = np.array(modified_arr)
    return modified_arr

import numpy as np

def set_n_closest_to_zero(arr, n):
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return [0] * len(arr)
    
    # Find the absolute values of the elements
    abs_arr = np.abs(arr)
    
    # Find the indices of the n elements closest to zero
    closest_indices = np.argpartition(abs_arr, n)[:n]
    
    # Set the elements closest to zero to zero
    modified_arr = arr.copy()
    modified_arr[closest_indices] = 0
    
    return modified_arr

import torch

def set_n_smallest_to_zero_torch(arr, n):
    if n <= 0:
        return arr
    
    if n >= len(arr):
        return torch.zeros_like(arr)
    
    # Find the nth smallest element
    nth_smallest = torch.kthvalue(arr, n).values
    
    # Set elements smaller than or equal to nth_smallest to zero
    modified_arr = torch.where(arr <= nth_smallest, torch.tensor(0.0), arr)
    
    return modified_arr

import numpy as np

def quantile_score(p, z, q):
    """
    Calculate the Quantile Score (QS) for a given probability and set of observations and quantiles.

    Parameters:
    p (float): The probability level.
    z (np.array): The observed values.
    q (np.array): The predicted quantiles.

    Returns:
    float: The Quantile Score (QS).

    From "Flexible and consistent quantile estimation for
            intensity–duration–frequency curves"
            by
            Felix S. Fauer, Jana Ulrich, Oscar E. Jurado, and Henning W. Rust, 2021
    We implemented this directly into the network...
    """
    u = z - q
    rho = np.where(u > 0, p * u, (p - 1) * u)
    return np.sum(rho) # we could consider if this should be mean, abs. sum or just sum as currently. 


import tensorflow as tf

def create_data(multimodal: bool):
    x = tf.random.uniform([1000], 0.3, 10)
    y =  tf.random.gamma([1], alpha=0.1 + x / 20.0) + tf.math.log(x) # draw [shape] samples from each of the gamma distributions given, since we provide a 1000 every time...
    x = tf.random.gamma([10], alpha=0.2 + x / 50)
    
    if multimodal:
        x_extra = tf.random.uniform([500], 5, 10)
        y_extra = tf.random.normal([500], 6.0, 0.3)
        x = tf.concat([x, x_extra], axis=0)
        y = tf.concat([y, y_extra], axis=0)
    
    return  tf.expand_dims( tf.transpose(x), -1), tf.expand_dims(tf.transpose(y), -1)





 