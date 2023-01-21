####
# vN/src/lib/helper.py 
# 
# This file contains 'general' helper functions which are useful across multiple scripts 
####
import pandas as pd
import numpy as np

def merge_duplicates(df, col_duplicate, col_merge_value, mode="sum"):
    """
    Checks for duplicate entries and according to the given mode performs an operation on the specified column's value.

    :df:                dataframe used
    :col_duplicate:     name of column to be checked for duplicates
    :col_merge_value:   name of column which value to perform an operation on when it concerns a duplicate
    :mode:              name which determines which operation is performed, default is 'sum' which takes the cummulative value
    :return:            dataframe which contains the unique entries and the operation's resulting column value
    """ 
    if mode == "sum":
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()
    # Default sum the <col_merge_value> values of the duplicates
    else:
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()

def generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors=None):
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

    # UGLY but works
    if latent_factors is not None:
        for latent_factor in latent_factors:
            for reg in regularization:
                for alpha in confidence_weighting:
                    configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                    i+=1
    else:
        for reg in regularization:
            for alpha in confidence_weighting:
                configurations[i] = {"reg": reg, "alpha": alpha}
                i+=1
                           
    return configurations