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

    Input:
        df                  - dataframe to be used
        col_duplicate       - name of column to be checked for duplicates
        col_merge_value     - name of column which value to perform an operation on when it concerns a duplicate
        mode_operation      - name which determines which operation is performed, default is 'sum' which takes the cummulative value
    Outputs:
        data frame          - dataframe which contains the unique entries and the operation's resulting column value
    """ 
    if mode_operation == "sum":
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()
    # Default sum the <col_merge_value> values of the duplicates
    else:
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()

def generate_hyperparameter_configurations(regularization, latent_factors, confidence_weighting=None):
    """
    Given the hyperparameter spaces generates all possible combinations for grid search

    Inputs:
        regularization          - contains lambda the regularization factor
        confidence_weighting    - contains alpha the confidence weight factor
        latent_factors          - contains the number of latent factors
    Outputs:
        dictionary              - in the format {<id>: {<hyperparameter_1 name>: <value>, ..., {<hyperparameter_N name>: <value>}}
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
    Returns the index in <model_train_results> corresponding with the highest performance

    Inputs:
        model_train_results         - results of training the models in format:
                                        [[<performance>, {"seed": <seed>, "model": <model_best>, "hyperparameters": <hyperparams_optimal>, "performance": <performance>}]]
    Outputs:
        integer                     - index
    """
    performance_models = [item[0] for item in model_train_results]
    # When maximization problem
    if mode == "max":
        return performance_models.index(max(performance_models))
    # When minimization problem
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

