from scipy import sparse
import scipy
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, AUC_at_k 
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from os import path

# 1st party imports
from lib import io
from lib import helper
import constant

# Create a recommendation system's preference estimation model using the given "ground truth"
def create_recommendation_est_system(ground_truth, hyperparameter_configurations, split={"train": 0.7, "validation": 0.1}):
    """
    Generates models that simulate a recommender system's estimation of preferences using low-rank matrix completion (Bell and Sejnowski 1995)

    :ground_truth:                  matrix containing estimated relevance scores serving as the ground truth preference
    :hyperparameter_configurions:   dictionary containing the hyperparameter spaces i.e. all possible hyperparameter combinations for grid search
    :split:                         specifies the ratio in which the train/validation/test split should be applied
    :returns:                       all models found during grid search with corresponding validation performance in terms of precision
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

    print("Generating recommender preference estimation models...")

    # Train low-rank matrix completion algorithm (Bell and Sejnowski 1995)
    for i in tqdm(range(len(hyperparameter_configurations))):
        params = hyperparameter_configurations[i]

        # Create model
        model = AlternatingLeastSquares(factors=params["latent_factor"],
                                        regularization=params["reg"],
                                        alpha=params["alpha"])

        # Train model
        model.fit(train, show_progress=False)

        # Validate model
        precision = AUC_at_k(model, train, validation, K=1000, show_progress=False, num_threads=4)
        models[params["latent_factor"]][precision] = {"i_params": i, "params": params, "p_val": precision, "model": model} 

    return models

def select_best_recommendation_est_system(recommendation_system_est_model, select_mode="latent"):
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
    print("Estimating preferences...")
    # Calculate estimated preference scores
    return recommendation_est_system_model.user_factors @ recommendation_est_system_model.item_factors.T

def create_recommendation_policies(preference_estimates, temperature=1):
    """
    Generates the recommendation policies given the estimated preference scores. 
    The recommendation policies we consider are softmax distributions over the predicted scores with fixed inverse temperature. 
    These policies recommend a single item, drawn from the softmax distribution.
    """
    # Temperature controls the softness of the probability distributions
    inverse_temperature = lambda x: x/temperature
    inverse_temperature(preference_estimates)

    # Compute the softmax transformation along the second axis (i.e., the rows)
    policies = scipy.special.softmax(preference_estimates, axis=1)
    return policies

def create_rewards(ground_truth):
    print("Generating binary rewards...")
    rewards = np.zeros(ground_truth.shape)

    # Based on the approximated interest, the rewards are drawn from the binomial distribution. 
    # The higher the interest the more likely it is to order the service and vice-versa.
    def draw_from_bernoulli(x):
        return np.random.binomial(1, 0.5)
    
    apply_bernoulli = np.vectorize(draw_from_bernoulli)   
    rewards = apply_bernoulli(rewards)
    return rewards
    
if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    my_globals = globals()

    # Generate ground truth of artist preferences
    filename = "user_artist.py"
    exec(compile(open(filename, "rb").read(), filename, "exec"))

    # Load ground truth
    print("Loading ground_truth_fm...")
    io.load("ground_truth_fm", my_globals)
    # print(ground_truth_fm)

    # Hyperparameters
    latent_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    regularization = [0.001, 0.01, 0.1, 1.0]
    confidence_weighting = [0.1, 1.0, 10.0, 100.0]
    
    configurations = helper.generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors)

    # Generate models to simulate a recommender system's preference estimation
    recommendation_system_est_models_fm_file = "recommendation_system_est_models_fm"

    # Only when not yet generated    
    if not path.exists(constant.VARIABLES_FOLDER + recommendation_system_est_models_fm_file):
        recommendation_system_est_models_fm = create_recommendation_est_system(ground_truth_fm, configurations, split={"train": 0.7, "validation": 0.1})

        # Save models that can estimate preferences
        io.save(recommendation_system_est_models_fm_file, (recommendation_system_est_models_fm_file, recommendation_system_est_models_fm))
    
    # Load from file
    print("Loading recommender preference estimation models...")
    io.load(recommendation_system_est_models_fm_file, my_globals)
    # print(recommendation_system_est_models_fm)

    # Get the recommender preference estimation models with best performances
    best_recommendation_est_system_fm = select_best_recommendation_est_system(recommendation_system_est_models_fm, select_mode="all")
    # print(best_recommendation_est_system_fm)

    preference_estimates_fm = {}

    # Use the model to simulate a recommender system’s estimation of preferences
    for latent_factor, data in best_recommendation_est_system_fm.items():
        preference_estimates_fm[latent_factor] = recommendation_estimation(data["model"])

    policies_fm = {}

    # Use the estimated preferences to generate policies
    for latent_factor, preference_estimates in preference_estimates_fm.items():
        policies_fm[latent_factor] = create_recommendation_policies(preference_estimates)

    # We generate binary rewards using a Bernoulli distribution with expectation given by our ground truth
    rewards = create_rewards(ground_truth_fm)