####
# vN/src/lib/helper.py 
# 
# This file contains 'general' helper functions which are useful across multiple scripts 
####
import pandas as pd
import numpy as np
from datetime import datetime

def merge_duplicates(df, col_duplicate, col_merge_value, mode_operation="sum"):
    """
    Checks for duplicate entries and according to the given mode performs an operation on the specified column's value.

    :df:                dataframe used
    :col_duplicate:     name of column to be checked for duplicates
    :col_merge_value:   name of column which value to perform an operation on when it concerns a duplicate
    :mode_operation:    name which determines which operation is performed, default is 'sum' which takes the cummulative value
    :return:            dataframe which contains the unique entries and the operation's resulting column value
    """ 
    if mode_operation == "sum":
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()
    # Default sum the <col_merge_value> values of the duplicates
    else:
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()

def generate_hyperparameter_configurations(regularization, latent_factors, confidence_weighting=None):
    """
    Given the hyperparameter spaces generates all possible combinations for grid search

    :regularization:        contains lambda the regularization factor
    :confidence_weighting:  contains alpha the confidence weight factor
    :latent_factors:        contains the number of latent factors
    :returns:               dictionary in the format {<id>: {<hyperparameter_1 name>: <value>, ..., {<hyperparameter_N name>: <value>}}
    """
    configurations = {}

    # Initialize with all possible hyperparameter combinations
    i = 0

    if confidence_weighting is not None:
        for latent_factor in latent_factors:
            for reg in regularization:
                for alpha in confidence_weighting:
                    configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                    i+=1
    else:
        for latent_factor in latent_factors:
            for reg in regularization:
                configurations[i] = {"latent_factor": latent_factor, "reg": reg}
                i+=1
                            
    return configurations

def get_index_best_model(model_train_results, mode="max"):
    """
    Returns the index in <model_train_results> corresponding with the highest performance <p_test>.

    :model_train_results:   results of training the models in format:
                            [[<p_test>, {"seed": <seed>, "model": <model_best>, "hyperparameters": <hyperparams_optimal>, "precision_test": <p_test>}]]
    """
    performance_models = [item[0] for item in model_train_results]
    if mode == "max":
        return performance_models.index(max(performance_models))
    else:
        return performance_models.index(min(performance_models))

def get_dictionary_subsets(dictionary, n):
    keys = np.array(list(dictionary.keys()))
    # Random split selection
    np.random.shuffle(keys)

    # Generate n splits of the dictionary
    subsets_keys = np.split(keys, n) 
    return {i: [dictionary[key] for key in subset_keys] for i, subset_keys in enumerate(subsets_keys)}

def get_current_datetime():
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    return now.strftime("%d/%m/%Y %H:%M:%S")

def normalize_matrix(matrix):
    def x_norm(x, x_min, x_max):
        # RuntimeWarning: invalid value encountered in divide
        # Handles division by zero since row is zero
        if((x_max - x_min) == 0):
            return 0
        return (x - x_min) / (x_max - x_min)

    normalize_x = np.vectorize(x_norm)
    
    # Given a row of values normalize each value
    def normalize(row):
        x_min, x_max = np.amin(row), np.amax(row)
        return normalize_x(row, x_min, x_max)

    # Normalize matrix
    return np.apply_along_axis(normalize, 1, matrix)

