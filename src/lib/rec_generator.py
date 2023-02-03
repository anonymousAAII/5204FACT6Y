####
# vN/src/lib/recommender.py
#
# This file simulates the recommender system containing create, select and exec/compute etc. functionality for each required recommender system component
####
import numpy as np
import pandas as pd
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k, precision_at_k
import scipy
from scipy import sparse
from tqdm import tqdm
import multiprocessing as mp
import time
import sys
from funk_svd import SVD
from sklearn.metrics import ndcg_score, dcg_score

# 1st party imports
from lib import istarmap
from lib import helper
import constant

def train_SVD_model(hyperparameter_configurations, batch, train, validation, test, metric, min_rating, max_rating, normalize=True):
    """
    Trains a recommender model using Funky SVD for a batch of hyper parameter combinations returning the model and its performance

    Inputs:
        hyperparameter_configurations       - list/batch containing different hyperparameter combinations
        batch                               - 'id' of batch being processed
        train                               - train set data frame with the columns <u_id>, <i_id> and <rating>
        validation                          - validation set data frame with again the user, item and rating data
        test                                - test set
        performance_metric                  - which performance metric to use to measure the model's performance
        min_rating                          - the minimum value a rating can have, i.e. can occur in the user-item matrix  
        max_rating                          - the maximum value a rating can have
        normalize                           - whether the normalize the rating values to a different scale range 
    Outputs:

    """
    print("Processing batch {}...SVD".format((batch + 1)))
    
    # To retrieve original data state
    train_tmp = train.copy(deep=True)
    validation_tmp = validation.copy(deep=True)
    test_tmp = test.copy(deep=True)
    
    num_entries, _ = test.shape
    num_items = len(test["i_id"].unique())
    num_users = num_entries / num_items
    shape = (int(num_users), int(num_items))
    
    K = constant.PERFORMANCE_METRICS["svd"][metric]["K"]
    results = {}

    # Train Funky SVD model for different hyperparameter configurations
    for i, params in enumerate(hyperparameter_configurations):
        # Give confidence weighing manually
        train["rating"] = train_tmp["rating"] * params["alpha"]
        validation["rating"] = validation_tmp["rating"] * params["alpha"]
        test["rating"] = test_tmp["rating"] * params["alpha"]

        svd = SVD(lr=0.001, reg=params["reg"], n_epochs=100, n_factors=params["latent_factor"], early_stopping=True, shuffle=False, min_rating=min_rating, max_rating=max_rating)

        # Fit model
        svd.fit(X=train, X_val=validation)

        # Try model on test set
        pred = svd.predict(test)

        y_true = np.reshape(np.array(test["rating"]), shape)
        y_score = np.reshape(np.array(pred), shape)

        # Measure performance on test set
        if metric == "ndcg":
            perf = ndcg_score(np.asarray(y_true), np.asarray(y_score), k=K)
        else:
            perf = dcg_score(np.asarray(y_true), np.asarray(y_score), k=K)

        results[i] = {"latent_factor": params["latent_factor"], "result": {"perf": perf, "model": svd, "params": params}} 
        
        print("Batch {}: {}@{} {}".format((batch + 1), metric, K, perf))

    return results

def train_ALS_model(hyperparameter_configurations, batch, train, validation, metric):
    """
    Trains a recommender model using ALS for a batch of hyper parameter combinations returning the model and its performance

    Inputs:
        hyperparameter_configurations       - list/batch containing different hyperparameter combinations
        batch                               - 'id' of batch being processed
        train                               - train set in sparse coo_matrix format 
        validation                          - validation set in sparse coo_matrix format
        performance_metric                  - which performance metric to use to measure the model's performance
    Outputs:

    """
    print("Processing batch {}...ALS".format((batch + 1)))
    
    K = constant.PERFORMANCE_METRICS["als"][metric]["K"]
    results = {}

    # Train ALS model for different hyperparameter configurations
    for i, params in enumerate(hyperparameter_configurations):
        # Create model
        model = AlternatingLeastSquares(factors=params["latent_factor"],
                                        regularization=params["reg"],
                                        alpha=params["alpha"])

        # Train model
        model.fit(train, show_progress=False)

        # Validate model
        if metric == "ndcg":
            perf = ndcg_at_k(model, train, validation, K=K, show_progress=False)
        else:
            perf = precision_at_k(model, train, validation, K=K, show_progress=False)
        
        print("Batch {}: {}@{} {}".format((batch + 1), metric, K, perf))

        results[i] = {"latent_factor": params["latent_factor"], "result": {"perf": perf, "model": model, "params": params}} 

    return results

def create_recommender_model(ground_truth, hyperparameter_configurations, algorithm, metric, split={"train": 0.7, "validation": 0.1}, multiprocessing=True):
    """
    Create a recommender system model according to the specified algorithm using the given "ground truth" i.e. true relevance scores

    Inputs:
        ground_truth                        - matrix containing estimated user-item relevance scores serving as the ground truth preferences
        hyperparameter_configurions         - dictionary containing the hyperparameter spaces i.e. all possible hyperparameter combinations for grid search
        algorithm                           - which algorithm to use to construct the recommender model either Funky SVD or ALS
        split                               - specifies the fractions in which the train/validation/test split should be applied
        multiprocessing                     - whether to apply multiprocessing for computational efficiency
    Outputs:
        dictionary                          - all models found during grid search indexed by <latent_factor> with corresponding validation performance
                                                {<latent_factor>: 
                                                    {<performance>: 
                                                        {"i_params": <index hyperparams config>, 
                                                        "params": <hyperparams config>, 
                                                        "performance": <performance>, 
                                                        "model": <model>}
                                                    }
                                                }
                                                for each model
    """
    R_coo = sparse.coo_matrix(ground_truth)
    R_min, R_max = R_coo.min(), R_coo.max()
    print("{}, {} = r_min, r_max".format(R_min, R_max))

    # Create keys from <latent_factors>
    df_params = pd.DataFrame.from_dict(hyperparameter_configurations)
    keys = np.array(df_params.loc["latent_factor"].drop_duplicates())

    # Data struct that will contain the found models in the form:
    # {<latent_factor>: {<performance>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "performance": <performance>, "model": <model>}}}
    # for each latent factor
    models = {key: {} for key in keys}

    # Funky SVD 
    if algorithm == "svd":
        row, col = ground_truth.shape
        values = ground_truth.flatten()
        u_id = np.repeat(np.arange(0, row), col)
        i_id = np.tile(np.arange(0, col), row)

        # Create whole user-item relevance scores data frame
        df = pd.DataFrame({"u_id": u_id, "i_id": i_id, "rating": values})
 
        # Create data splits
        train = df.sample(frac=split["train"], random_state=1)
        validation_size = split["validation"] / (1 - split["train"])
        validation = df.drop(train.index.tolist()).sample(frac=validation_size, random_state=1)
        test = df.drop(train.index.tolist()).drop(validation.index.tolist())

        # Multiprocessing: assign a batch of models to a processor
        if multiprocessing:
            print("Generating recommender preference estimation models...MULTIPROCESSING")
            # Generate random batches of hyperparameter configurations for multiprocessing
            configurations_batches = helper.get_dictionary_subsets(hyperparameter_configurations, mp.cpu_count())

            # Create and configure the process pool
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(batch, i, train, validation, test, metric, R_min, R_max) for i, batch in configurations_batches.items()]
                # Execute tasks and process results in order
                for batch_result in pool.starmap(train_SVD_model, items):
                    for result in batch_result.values():
                        models[result["latent_factor"]][result["result"]["perf"]] = result["result"] 
            # process pool is closed automatically              
        # Sequential
        else:
            print("Generating recommender preference estimation models...SEQUENTIAL")
            print("Processing batch {}...".format(1))
            
            results = train_SVD_model(list(hyperparameter_configurations.values()), train, validation, test, metric, R_min, R_max)
            
            for result in results.values():
                models[result["latent_factor"]][result["result"]["perf"]] = result["result"] 
    # Default use ALS which solves the matrix completion algorithm by seeing it as an optimazation problem using SVD
    else:
        # Create data split 
        train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=split["train"])
        validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=split["validation"]/(1 - split["train"]))
        
        # Multiprocessing: assign a batch of models to a processor
        if multiprocessing:
            print("Generating recommender preference estimation models...MULTIPROCESSING")
            # Generate random batches of hyperparameter configurations for multiprocessing
            configurations_batches = helper.get_dictionary_subsets(hyperparameter_configurations, mp.cpu_count())

            # Create and configure the process pool
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(batch, i, train, validation, metric) for i, batch in configurations_batches.items()]
                # Execute tasks and process results in order
                for batch_result in tqdm(pool.istarmap(train_ALS_model, items), total=len(items)):
                    for result in batch_result.values():
                        models[result["latent_factor"]][result["result"]["perf"]] = result["result"] 
        # Sequential
        else:
            print("Generating recommender preference estimation models...SEQUENTIAL")
            print("Processing batch {}...".format(1))

            results = train_ALS_model(list(hyperparameter_configurations.values()), 1, train, validation, metric)
            for result in results.values():
                models[result["latent_factor"]][result["result"]["perf"]] = result["result"] 
                
    return models

def get_best_recommender_models(recommender_models, selection_mode="latent"):
    """
    Given a dictionary of models finds the best models either the overal best model or the best model per <latent_factor>.
    Default is selecting the best model for each <latent_factor>

    Inputs:
        recommendation_system_est_model         - dictionary containing the models on which a best performance selection should be performed
        select_mode                             - modus of selecting the models either "all" or "latent"
    Outputs:
        dictionary                              - dictionary containing the best models to simulate a recommender system's preference estimations
    """
    best_recommender_models = {}

    # Selects the best model within each possible <latent_factor> -> one select/pick per unique <latent_factor>
    for latent_factor in recommender_models.keys():
        # Find best model for <latent_factor>
        key_max = np.amax(np.array(list(recommender_models[latent_factor].keys())))
        best_recommender_models[latent_factor] = recommender_models[latent_factor][key_max]

    # When <select_mode> is "latent" return dictionary with best model per <latent_factor> 
    if selection_mode == "latent":
        return best_recommender_models

    # Find overall best model
    perf_base = 0
    best_recommender_model = {}

    for latent_factor, data in best_recommender_models.items():
        if data["perf"] > perf_base:
            perf_base = data["perf"]
            best_recommender_model = {latent_factor: data}

    # Only return one single best model over all hyperparameter combinations (no picking by <latent_factor>)
    return best_recommender_model
    
