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
from implicit.evaluation import ndcg_at_k, precision_at_k, AUC_at_k
import scipy
from scipy import sparse
from tqdm import tqdm
import multiprocessing as mp
import time
import sys
from funk_svd.dataset import fetch_ml_ratings
from funk_svd import SVD
from sklearn.metrics import ndcg_score, dcg_score, mean_absolute_error, mean_squared_error

# 1st party imports
from lib import istarmap
from lib import helper
import constant

def train_SVD_model(hyperparameter_configurations, batch, train, validation, test, performance_metric, min_rating, max_rating, normalize=True):
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
    results = {}
    
    print("Processing batch {}...SVD".format((batch + 1)))
    K = constant.PERFORMANCE_METRIC_VARS["recommender system"]["svd"][performance_metric]["K"]
    
    # # Normalize between range [1, 5] for Funky SVD to work
    # if normalize == True:
    #     min_value = min(min(train["rating"].min(), validation["rating"].min()), test["rating"].min()) 
    #     max_value = max(max(train["rating"].max(), validation["rating"].max()), test["rating"].max())

    #     train["rating"] = min_rating + (((train["rating"]  - min_value) * (max_rating - min_rating)) / (max_value - min_value))
    #     validation["rating"] = min_rating + (((validation["rating"]  - min_value) * (max_rating - min_rating)) / (max_value - min_value))
    #     test["rating"] = min_rating + (((test["rating"]  - min_value) * (max_rating - min_rating)) / (max_value - min_value))
    
    num_entries, _ = test.shape
    num_items = len(test["i_id"].unique())

    # Train Funky SVD model for different hyperparameter configurations
    for i, params in enumerate(hyperparameter_configurations):
        svd = SVD(lr=0.001, reg=params["reg"], n_epochs=100, n_factors=params["latent_factor"], early_stopping=True, shuffle=False, min_rating=min_rating, max_rating=max_rating)

        # Fit model
        svd.fit(X=train, X_val=validation)

        # Measure performance on test set
        pred = svd.predict(test)

        num_users = num_entries / num_items
        shape = (int(num_users), int(num_items))

        y_true = np.reshape(np.array(test["rating"]), shape)
        y_score = np.reshape(np.array(pred), shape)

        # print(y_true)
        # print(y_score)
        
        if performance_metric == "ndcg":
            ndcg = ndcg_score(np.asarray(y_true), np.asarray(y_score), k=K)
        else:
            ndcg = dcg_score(np.asarray(y_true), np.asarray(y_score), k=K)

        results[i] = {"latent_factor": params["latent_factor"], "result": {"ndcg": ndcg, "model": svd, "params": params}} 
        
        print("Batch {}: {}@{} {}".format((batch + 1), performance_metric, K, ndcg))

    return results

def train_ALS_model(hyperparameter_configurations, batch, train, validation, performance_metric):
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
    results = {}
    
    print("Processing batch {}...ALS".format((batch + 1)))
    K = constant.PERFORMANCE_METRIC_VARS["recommender system"]["als"][performance_metric]["K"]

    # Train ALS model for different hyperparameter configurations
    for i, params in enumerate(hyperparameter_configurations):
        # Create model
        model = AlternatingLeastSquares(factors=params["latent_factor"],
                                        regularization=params["reg"],
                                        alpha=params["alpha"])

        # Train model
        model.fit(train, show_progress=False)

        # Validate model
        if performance_metric == "ndcg":
            ndcg = ndcg_at_k(model, train, validation, K=K, show_progress=False)
        elif performance_metric == "precision":
            ndcg = precision_at_k(model, train, validation, K=K, show_progress=False)
        else:
            ndcg = AUC_at_k(model, train, validation, K=K, show_progress=False)
        
        print("Batch {}: {}@{} {}".format((batch + 1), performance_metric, K, ndcg))

        results[i] = {"latent_factor": params["latent_factor"], "result": {"ndcg": ndcg, "model": model, "params": params}} 

    return results

def create_recommendation_est_system(ground_truth, hyperparameter_configurations, algorithm, split={"train": 0.7, "validation": 0.1}, multiprocessing=True):
    """
    Create a recommender system model according to the specified algorithm using the given "ground truth" i.e. true relevance scores

    Inputs:
        ground_truth                        - matrix containing estimated relevance scores serving as the ground truth preference
        hyperparameter_configurions         - dictionary containing the hyperparameter spaces i.e. all possible hyperparameter combinations for grid search
        algorithm                           - which algorithm to use to construct the recommender model either SVD or ALS
        split                               - specifies the fractions in which the train/validation/test split should be applied
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
    print(R_min, R_max)

    # Create keys from <latent_factors>
    df_params = pd.DataFrame.from_dict(hyperparameter_configurations)
    keys = np.array(df_params.loc["latent_factor"].drop_duplicates())

    # Data struct containing models in the form:
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
        train = df.sample(frac=split["train"])
        validation_size = split["validation"] / (1 - split["train"])
        validation = df.drop(train.index.tolist()).sample(frac=validation_size)
        test = df.drop(train.index.tolist()).drop(validation.index.tolist())

        # print(train.shape)
        # print(validation.shape)
        # print(test.shape)
        
        # # Drop such that we compute latent factors only using the values we know
        # train = train.drop(train[train.rating < 0].index)
        # validation = validation.drop(validation[validation.rating < 0].index)
        # test = test.drop(test[test.rating < 0].index)
        
        # # Set to zero such that we compute latent factors only using the values we know
        # train["rating"].mask(train["rating"] < 0 , 0, inplace=True)
        # validation["rating"].mask(validation["rating"] < 0 , 0, inplace=True)
        # test["rating"].mask(test["rating"] < 0 , 0, inplace=True)

        # Multiprocessing: assign a batch of models to a processor
        if multiprocessing:
            print("Generating recommender preference estimation models...MULTIPROCESSING")
            # Generate random batches of hyperparameter configurations for multiprocessing
            configurations_batches = helper.get_dictionary_subsets(hyperparameter_configurations, mp.cpu_count())

            # Create and configure the process pool
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(batch, i, train, validation, test, constant.PERFORMANCE_METRIC_REC, 0, R_max) for i, batch in configurations_batches.items()]
                # Execute tasks and process results in order
                for result in pool.starmap(train_SVD_model, items):
                    print(f'GOT RESULT: {result}', flush=True)
                    for train_result in result.values():
                        models[train_result["latent_factor"]][train_result["result"]["ndcg"]] = train_result["result"] 
            # process pool is closed automatically              
        # Sequential
        else:
            print("Generating recommender preference estimation models...SEQUENTIAL")
            print("Processing batch {}...".format(1))
            
            result = train_SVD_model(list(hyperparameter_configurations.values()), train, validation, test, constant.PERFORMANCE_METRIC_REC, 0, R_max)
            print(result)
            for train_result in result.values():
                models[train_result["latent_factor"]][train_result["result"]["ndcg"]] = train_result["result"] 
    # Default use ALS which solves the matrix completion algorithm by seeing it as an optimazation problem using SVD
    else:
        # Create data split 
        train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=split["train"])
        validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=split["validation"]/(1 - split["train"]))
        
        # Multiprocessing: assign a batch of models to a processor
        if multiprocessing:
            print("Generating recommender preference estimation models...MULTIPROCESSING")
            # Generate random batches of hyperparameter configurations for multiprocessing
            configurations_batches = helper.get_dictionary_subsets(hyperparameter_configurations, 4)

            # Create and configure the process pool
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(batch, i, train, validation, constant.PERFORMANCE_METRIC_REC) for i, batch in configurations_batches.items()]
                # Execute tasks and process results in order
                for result in tqdm(pool.istarmap(train_ALS_model, items), total=len(items)):
                    for train_result in result.values():
                        models[train_result["latent_factor"]][train_result["result"]["ndcg"]] = train_result["result"] 
        # Sequential
        else:
            print("Generating recommender preference estimation models...SEQUENTIAL")
            print("Processing batch {}...".format(1))

            result = train_ALS_model(list(hyperparameter_configurations.values()), 1, train, validation, constant.PERFORMANCE_METRIC_REC)
            for train_result in result.values():
                models[train_result["latent_factor"]][train_result["result"]["ndcg"]] = train_result["result"] 
                
    return models

def select_best_recommendation_est_system(recommendation_system_est_model, select_mode="latent"):
    """
    Given a dictionary of models finds the best models either the overal best model or the best model per <latent_factor>.
    Default is selecting the best model for each <latent_factor>

    Inputs:
        recommendation_system_est_model         - dictionary containing the models on which a best performance selection should be performed
        select_mode                             - modus of selecting the models either "all" or "latent"
    Outputs:
        dictionary                              - dictionary containing the best models to simulate a recommender system's preference estimations
    """
    best_recommendation_est_systems = {}
    # Selects the best model within each possible <latent_factor> -> one select/pick per unique <latent_factor>
    for latent_factor in recommendation_system_est_model.keys():
        # Find best model for <latent_factor>
        key_max = np.amax(np.array(list(recommendation_system_est_model[latent_factor].keys())))
        best_recommendation_est_systems[latent_factor] = recommendation_system_est_model[latent_factor][key_max]

    # When <select_mode> is "latent" return dictionary with best model per <latent_factor> 
    if select_mode == "latent":
        return best_recommendation_est_systems

    # Find overall best model
    ndcg_base = 0
    best_recommendation_est_system = {}
    for latent_factor, data in best_recommendation_est_systems.items():
        if data["ndcg"] > ndcg_base:
            ndcg_base = data["ndcg"]
            best_recommendation_est_system = {latent_factor: data}

    # Only return one single best model over all hyperparameter combinations (no picking by <latent_factor>)
    return best_recommendation_est_system

def recommendation_estimation(recommendation_est_system_model, ground_truth, algorithm):
    """
    
    Inputs:
        recommendation_est_system_model         - model object that simulates a recommender system's preferences estimation
        ground_truth                            - matrix containing the user-item true relevance scores
        algorithm                               - algorithm of the recommender model e.g. ALS or Funky SVD
    Outputs:
        matrix                                  - matrix containing the recommender model's predictions, i.e. estimated preference scores
    """
    print("Estimating preferences...")
    
    # Funky SVD
    if algorithm == "svd":
        row, col = ground_truth.shape
        u_id = np.repeat(np.arange(0, row), col)
        i_id = np.tile(np.arange(0, col), row)
        
        # Construct all user-item pairs for the recommender system model
        df = pd.DataFrame({"u_id": u_id, "i_id": i_id})
        
        # Calculate estimated preference scores
        pred = recommendation_est_system_model.predict(df)

        return np.array(pred).reshape(ground_truth.shape)
    # ALS
    else:
        return recommendation_est_system_model.user_factors @ recommendation_est_system_model.item_factors.T

def create_recommendation_policies(preference_estimates, temperature=1/5):
    """
    Generates the recommendation policies given the estimated preference scores. 
    The recommendation policies we consider are softmax distributions over the predicted scores with fixed inverse temperature. 
    These policies recommend a single item, drawn from the softmax distribution.

    Inputs:    
        preference_estimates        - estimated preference scores
        temperature                 - controls the softness of the probability distributions
    Outputs:
        dictionary                  - containing the picked policy (i.e. recommended item) and the recommendation policy both in user-item matrix format
    """
    print("Generating recommendation policies...")

    # Apply temperature parameter   
    divider = np.full(preference_estimates.shape[0], temperature)
    preference_estimates = np.divide(preference_estimates.T, divider).T
    
    # Compute the softmax transformation along the second axis (i.e., the rows)
    policies = scipy.special.softmax(preference_estimates, axis=1)

    # According to a given policy i.e. a probability distribution 
    # select an item by drawing from this distribution -> is the recommendation
    def select_policy(distribution, indices):
        i_drawn_policy = np.random.choice(indices, 1, p=distribution)
        recommendation = np.zeros(len(indices))
        recommendation[i_drawn_policy] = 1
        return recommendation
    
    # Since stationary policies pick a policy for an user only once
    indices = np.arange(len(policies[0]))
    recommendation = np.apply_along_axis(select_policy, 1, policies, indices)
    # print(recommendation)
    # print(policies)
    return {"recommendations": recommendation, "policies": policies}

def create_rewards(ground_truth):
    """
    Generates the rewards by a Bernoulli distribution per item and the expectation of the Bernoulli distribution

    Inputs:
        ground_truth        - ground truth 
    Outputs:
        tuple               - tuple containing the binary rewards and expecation of the binary rewards in user-item matrix format
    """
    print("Generating binary rewards...")
        
    expectation = np.copy(ground_truth)

    rewards = np.zeros(ground_truth.shape)

    # Based on the approximated interest, the rewards are drawn from the binomial distribution. 
    # The higher the interest the more likely it is to order the service and vice-versa.
    def draw_from_bernoulli(x):
        return np.random.binomial(1, 0.5)
    
    apply_bernoulli = np.vectorize(draw_from_bernoulli)   
    rewards = apply_bernoulli(rewards)
    return rewards, expectation
