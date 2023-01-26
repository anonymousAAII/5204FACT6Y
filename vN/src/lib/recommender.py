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
from implicit.evaluation import precision_at_k, AUC_at_k 
import scipy
from scipy import sparse
from tqdm import tqdm
import multiprocessing as mp
import time

# 1st party imports
from lib import istarmap
from lib import helper

def train_model(hyperparameter_configurations, batch, train, validation):
    results = {}
    
    print("Processing batch {}...".format(batch))

    # Train low-rank matrix completion algorithm (Bell and Sejnowski 1995)
    for i, params in enumerate(hyperparameter_configurations):
        # Create model
        model = AlternatingLeastSquares(factors=params["latent_factor"],
                                        regularization=params["reg"],
                                        alpha=params["alpha"])

        # Train model
        model.fit(train, show_progress=False)

        # Validate model
        precision = AUC_at_k(model, train, validation, K=100, show_progress=False)

        results[i] = {"latent_factor": params["latent_factor"], "result": {"p_val": precision, "model": model, "params": params}} 

    return results

# Create a recommendation system's preference estimation model using the given "ground truth"
def create_recommendation_est_system(ground_truth, hyperparameter_configurations, split={"train": 0.7, "validation": 0.1}, multiprocessing=True):
    """
    Generates models that simulate a recommender system's estimation of preferences using low-rank matrix completion (Bell and Sejnowski 1995)

    :ground_truth:                  matrix containing estimated relevance scores serving as the ground truth preference
    :hyperparameter_configurions:   dictionary containing the hyperparameter spaces i.e. all possible hyperparameter combinations for grid search
    :split:                         specifies the ratio in which the train/validation/test split should be applied
    :returns:                       all models found during grid search indexed by <latent_factor> with corresponding validation performance in terms of precision
                                    {<latent_factor>: 
                                        {<precision>: 
                                            {"i_params": <index hyperparams config>, 
                                            "params": <hyperparams config>, 
                                            "p_val": <validation precision>, 
                                            "model": <model>}
                                        }
                                    }
                                    for each model
    """
    R_coo = sparse.coo_matrix(ground_truth)
    # print(R)

    # Create data split 
    train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=split["train"])
    validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=split["validation"]/(1 - split["train"]))
    
    # Create keys from <latent_factors>
    df = pd.DataFrame.from_dict(hyperparameter_configurations)
    keys = np.array(df.loc["latent_factor"].drop_duplicates())

    # Data struct containing models in the form:
    # {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}
    # for each latent factor
    models = {key: {} for key in keys}

    start = time.time()


    # Multiprocessing: assign a batch of models to a processor
    if multiprocessing:
        print("Generating recommender preference estimation models...MULTIPROCESSING")
        # Generate random batches of hyperparameter configurations for multiprocessing
        configurations_batches = helper.get_dictionary_subsets(hyperparameter_configurations, 4)

        with mp.Pool(mp.cpu_count()) as pool:
            iterable = [(batch, i, train, validation) for i, batch in configurations_batches.items()]
            for batch_results in tqdm(pool.istarmap(train_model, iterable), total=len(iterable)):
                for result in batch_results.values():
                    models[result["latent_factor"]][result["result"]["p_val"]] = result["result"] 
        pool.close()
    # Sequential
    else:
        print("Generating recommender preference estimation models...SEQUENTIAL")
        print("Processing batch {}...".format(1))

        # NOT DRY (Don't Repeat Yourself)! Could have called the train() function, but function calls might have too much overhead in Python
        for i in tqdm(range(len(hyperparameter_configurations))):
            params = hyperparameter_configurations[i]

            # Create model
            model = AlternatingLeastSquares(factors=params["latent_factor"],
                                            regularization=params["reg"],
                                            alpha=params["alpha"])

            # Train model
            model.fit(train, show_progress=False)

            # Validate model
            precision = AUC_at_k(model, train, validation, K=100, show_progress=False)

            models[params["latent_factor"]][precision] =  {"p_val": precision, "model": model, "params": params}

        # for result in train_model(list(hyperparameter_configurations.values()), 1, train, validation):
        #     models[result["latent_factor"]][result["result"]["p_val"]] = result["result"] 
        
    print(models)
    end = time.time()
    print(end - start)

    return models

def select_best_recommendation_est_system(recommendation_system_est_model, select_mode="latent"):
    """
    Given a dictionary of models finds the best models either the overal best model or the best model per <latent_factor>.
    Default is selecting the best model for each <latent_factor>

    :recommendation_system_est_model:   dictionary containing the models on which a best performance selection should be performed
    :select_mode:                       modus of selecting the models either "all" or "latent"
    :returns:                           dictionary containing the best models to simulate a recommendation system's preference estimations
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
    p_base = 0
    best_recommendation_est_system = {}
    for latent_factor, data in best_recommendation_est_systems.items():
        p = data["p_val"]
        if p > p_base:
            p_base = p
            best_recommendation_est_system = {latent_factor: data}

    # Only return one single best model over all hyperparameter combinations (no picking by <latent_factor>)
    return best_recommendation_est_system

def recommendation_estimation(recommendation_est_system_model):
    """
    :recommendation_est_system_model:   model object that simulates a recommender system's preference estimations
    :returns:                           estimated preference scores
    """
    print("Estimating preferences...")
    # Calculate estimated preference scores
    return recommendation_est_system_model.user_factors @ recommendation_est_system_model.item_factors.T

def create_recommendation_policies(preference_estimates, temperature=1/5):
    """
    Generates the recommendation policies given the estimated preference scores. 
    The recommendation policies we consider are softmax distributions over the predicted scores with fixed inverse temperature. 
    These policies recommend a single item, drawn from the softmax distribution.

    :preference_estimates:  estimated preference scores
    :temperature:           controls the softness of the probability distributions
    :returns:               the picked policy (i.e. recommended item) -> recommendation, the probability of recommending an item to a user -> user policies
    """
    print("Generating recommendation policies...")

    # Apply temperature parameter   
    divider = np.full(preference_estimates.shape[0], temperature)
    preference_estimates = np.divide(preference_estimates.T, divider).T
    
    # Compute the softmax transformation along the second axis (i.e., the rows)
    policies = scipy.special.softmax(preference_estimates, axis=1)

    # According to a given probability distribution 
    # select a policy by drawing from this distribution -> is the recommendation
    def select_policy(distribution, indices):
        i_drawn_policy = np.random.choice(indices, 1, p=distribution)
        recommendation = np.zeros(len(indices))
        recommendation[i_drawn_policy] = 1
        return recommendation
    
    # Since stationary policies pick a policy for an user only once
    indices = np.arange(len(policies[0]))
    recommendation = np.apply_along_axis(select_policy, 1, policies, indices)
    return recommendation, policies

def create_rewards(ground_truth, normalize=True):
    """
    Generates the rewards by a Bernoulli distribution per item and the expectation of the Bernoulli distribution

    :ground_truth:  ground truth 
    :normalize:     whether to normalize the <ground_truth>
    :returns:       binary rewards, expecation of the binary rewards
    """
    print("Generating binary rewards...")
    
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

    # Create expectation of the Bernoulli distribution
    if normalize:
        # Normalize ground truth such that it can represent the expectation of the Bernoulli distribution
        expectation = np.apply_along_axis(normalize, 1, ground_truth)
    else:
        expectation = np.copy(ground_truth)

    rewards = np.zeros(ground_truth.shape)

    # Based on the approximated interest, the rewards are drawn from the binomial distribution. 
    # The higher the interest the more likely it is to order the service and vice-versa.
    def draw_from_bernoulli(x):
        return np.random.binomial(1, 0.5)
    
    apply_bernoulli = np.vectorize(draw_from_bernoulli)   
    rewards = apply_bernoulli(rewards)
    return rewards, expectation
