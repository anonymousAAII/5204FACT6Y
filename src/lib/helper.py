####
# vN/src/lib/helper.py 
# 
# This file contains 'general' helper functions which are useful across multiple scripts 
####
import pandas as pd
import numpy as np
from datetime import datetime
import os
from os import path
import sys

# 1st party imports
import constant
from lib import io

def init_directories():
    """
    Construct required framework directory structure to be able to construct the pipeline
    """
    # Folder to save intermediate variables of the pipeline
    var_folder1 = constant.VARIABLES_FOLDER
    if not path.exists(var_folder1):
            os.mkdir(var_folder1)

    # Per data set
    for data_set in constant.DATA_SETS.values():
        # Create folder for specific data set
        var_folder2 = var_folder1 + data_set["var_folder"]
        if not path.exists(var_folder2):
            os.mkdir(var_folder2)

        # Create folder for ground truth pipeline stage
        gt = "ground_truth/"
        if not path.exists(var_folder2 + gt):
            os.mkdir(var_folder2 + gt)

        for gt_model in constant.ALGORITHM_GROUND_TRUTH:
            var_folder3 = var_folder2 + gt + gt_model + "/"
            if not path.exists(var_folder3):
                os.mkdir(var_folder3)

        # Create folder for recommender pipeline stage
        rec = "recommender/"
        if not path.exists(var_folder2 + rec):
            os.mkdir(var_folder2 + rec)

        for rec_model in constant.ALGORITHM_RECOMMENDER:
            var_folder4 = var_folder2 + rec + rec_model + "/"
            if not path.exists(var_folder4):
                os.mkdir(var_folder4)

        # Create folder to store results
        if not path.exists(constant.RESULTS_FOLDER):
            os.mkdir(constant.RESULTS_FOLDER)

        # Create folder to save time performances (wall clock time)
        if not path.exists(constant.TIMING_FOLDER):
            os.mkdir(constant.TIMING_FOLDER)

        for data_set in constant.DATA_SETS.values():
            if not path.exists(constant.TIMING_FOLDER + data_set["log_file"]):
                io.initialize_empty_file(constant.TIMING_FOLDER + data_set["log_file"])

        # Create folder to save recommender system models
        if not path.exists(constant.MODELS_FOLDER):
            os.mkdir(constant.MODELS_FOLDER)

        for data_set in constant.DATA_SETS.values():
            if not path.exists(constant.MODELS_FOLDER + data_set["var_folder"]):
                os.mkdir(constant.MODELS_FOLDER + data_set["var_folder"])

def get_var_path(data_set, model_folder, model_settings):
    """
        Construct the relative path of where to store and retrieve variables
    """
    return constant.VARIABLES_FOLDER + data_set["var_folder"] + model_folder + model_settings["ALGORITHM"] + "/"

def get_log_path(data_set):
    """
        Construct the relative pathe of where to log info
    """
    return constant.TIMING_FOLDER + data_set["log_file"]

def get_current_datetime():
    """
        Returns system's datetime
    """
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    return now.strftime("%d/%m/%Y_%H:%M:%S")

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

def generate_hyperparameter_configurations(regularization, latent_factors, confidence_weighting):
    """
    Given the hyperparameter spaces generates all possible combinations for grid search

    Inputs:
        regularization          - contains lambda the regularization factor
        confidence_weighting    - contains alpha the confidence weigh factor
        latent_factors          - contains the number of latent factors
    Outputs:
        dictionary              - in the format {<id>: {<hyperparameter_1 name>: <value>, ..., {<hyperparameter_N name>: <value>}}
    """
    configurations = {}

    # Initialize with all possible hyperparameter combinations
    i = 0

    for latent_factor in latent_factors:
        for reg in regularization:
            for alpha in confidence_weighting:
                configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                i+=1
                            
    return configurations

def get_dictionary_subsets(dictionary, n, random=True):
    """
    Given a dictionary makes a split into subset dictionaries

    Inputs:
        dictionary      - dictionary to split
        n               - number of chunks to split the dictionary into
        random          - whether to perform a random split
    Outputs:
        list            - containing the resulting subset dictionaries
    """

    keys = np.array(list(dictionary.keys()))

    if random:
        # Random split selection
        np.random.shuffle(keys)

    # Generate n splits of the dictionary
    subsets_keys = np.split(keys, n) 
    return {i: [dictionary[key] for key in subset_keys] for i, subset_keys in enumerate(subsets_keys)}

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

def save_recommender(recommender):
    """"
    Saves recommender object locally
    """
    path = constant.MODELS_FOLDER + recommender.data_set["var_folder"]

    params = recommender.params
    file_name = "{}_{}_lf:{}_reg:{}_alph:{}".format(recommender.data_set["label"],
                                                recommender.model_type, 
                                                params["latent_factor"],
                                                params["reg"],
                                                params["alpha"])

    print("Saving recommender....{}".format(path + file_name))
    io.save(path + file_name, (file_name, recommender))
    return file_name

def get_recommender(data_set, file_name, my_globals):
    """
    Retrieves recommender object locally
    """
    file_path = constant.MODELS_FOLDER + data_set["var_folder"] + file_name
    
    if not path.exists(file_path):
        sys.exit('Recommender {} does not exist!'.format(file_path))
    
    io.load(file_path, my_globals)
    return my_globals[file_name]


    
    