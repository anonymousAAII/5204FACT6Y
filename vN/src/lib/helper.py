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
                configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                i+=1
                           
    return configurations

def generate_user_item_matrix(user_items_dict, users, items, coordinates_mode="interaction"):
    """
    Generates the user-item observation matrix R (Johnson 2014)

    :user_items_dict:   dictionary in the form {<user_id>: {<artist_1_id>: <weight>, ...., <artist_N_id>: <weight>}}
    :users:             list of user ids
    :items:             list of item ids
    :coordinates_mode:  modus of which coordinates to save; 'all' to save all coordinates of R, 'interaction' only those who contain an user-item interaction/observation
    :returns:           user-item observation matrix R, dictionary with coordinates in R of the form {<index>: {"r": <row index>, "c": <column index>}} 
    """
    # User-item observation matrix (Johnson 2014)    
    R = np.zeros((len(users), len(items)))

    # (x, y) coordinates in R
    R_coordinates = {}
    i = 0

    for row, user in enumerate(users):
        for col, item in enumerate(items):
            # Save ALL coordinates of R
            if coordinates_mode == "all":
                R_coordinates[i] = {"r": row, "c": col}
                i+=1

            # When no existing user-item interaction
            if not (item in user_items_dict[user]):
                continue

            # print(user_items_dict[user][item])
            R[row][col] = user_items_dict[user][item]

            # Only save coordinates of R where there has been an interaction, i.e. target != empty
            if coordinates_mode == "interaction":
                R_coordinates[i] = {"r": row, "c": col}
                i+=1

    return R, R_coordinates

def generate_masked_matrix(R, R_coordinates, mask_indices, mask_mode):
    """
    Generates a masked version of matrix R

    :R:             user-item matrix R which should be masked
    :R_coordinates: dictionary containing coordinates {"r": <row index>, "c": <column index>} of all or only filled cells in R 
    :mask_indices:  indices in <R_coordinates> of coordinates to mask
    :mask_mode:     modus of how to mask i.e. 'zero' = set values in R to zero, or 'value' = set values of R in zero matrix
    :returns:       masked version matrix of R
    """
    # Copy filled array and set mask to zero
    if mask_mode == "zero":
        R_masked = np.copy(R)
    # Create zero array and set mask to value
    else:
        R_masked = np.zeros(R.shape)

    # Loop through indices of mask
    for i in mask_indices:
        # Get mask's coordinates in R
        coordinates = R_coordinates[i]

        # According to <mask> mode set r_ui to zero       
        if mask_mode == "zero":
            R_masked[coordinates["r"]][coordinates["c"]] = 0
            continue

        # Set r_ui to value
        R_masked[coordinates["r"]][coordinates["c"]] = R[coordinates["r"]][coordinates["c"]]

    return R_masked