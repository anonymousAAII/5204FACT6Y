import os
import numpy as np
from os import path

# 1st party imports
from lib import io
from lib import helper
from lib import recommender
from lib import envy
import constant

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    my_globals = globals()

    # Generate ground truth of user-artist preferences
    filename = "user_artist.py"
    exec(compile(open(filename, "rb").read(), filename, "exec"))

    # Load ground truth
    print("Loading ground_truth_fm...")
    io.load("ground_truth_fm", my_globals)
    # print(ground_truth_fm)

    ##########################
    #   RECOMMENDER SYSTEM: from here on we generate 144 preference estimation models one for each hyperparameter combination
    ##########################

    # Hyperparameter spaces
    latent_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    regularization = [0.001, 0.01, 0.1, 1.0]
    confidence_weighting = [0.1, 1.0, 10.0, 100.0]
    
    configurations = helper.generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors)

    # Generate models to simulate a recommender system's preference estimation
    recommendation_system_est_models_fm_file = "recommendation_system_est_models_fm"

    # Only when not yet generated
    if not path.exists(constant.VARIABLES_FOLDER + recommendation_system_est_models_fm_file):
        recommendation_system_est_models_fm = recommender.create_recommendation_est_system(ground_truth_fm, configurations, split={"train": 0.7, "validation": 0.1})

        # Save models that can estimate preferences as a recommender system would
        io.save(recommendation_system_est_models_fm_file, (recommendation_system_est_models_fm_file, recommendation_system_est_models_fm))
    
    # Load from file
    print("Loading recommender preference estimation models...")
    io.load(recommendation_system_est_models_fm_file, my_globals)
    # print(recommendation_system_est_models_fm)


    ##########################
    #   RECOMMENDER SYSTEM: from here on we continue with the best models we select which are either the overall best model
    #                       or the best model per latent_factor. Thus we have a list in the form:
    #                       {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}}
    ##########################

    # Get the recommender preference estimation models with best performances
    best_recommendation_est_system_fm = recommender.select_best_recommendation_est_system(recommendation_system_est_models_fm, select_mode="all")
    # print(best_recommendation_est_system_fm)

    preference_estimates_fm = {}

    # Use the model to simulate a recommender system’s estimation of preferences
    for latent_factor, data in best_recommendation_est_system_fm.items():
        preference_estimates_fm[latent_factor] = recommender.recommendation_estimation(data["model"])

    policies_fm = {}

    # Use the estimated preferences to generate policies
    for latent_factor, preference_estimates in preference_estimates_fm.items():
        policies, probability_policies = recommender.create_recommendation_policies(preference_estimates)
        policies_fm[latent_factor] = {"policies": policies, "probability_policies": probability_policies}

    ##########################
    #   REWARDS: the rewards are independent of the recommender system model, thus we only generate it once
    ##########################
    
    rewards_fm_file, expec_rewards_fm_file = "rewards_fm", "expec_rewards_fm"

    # We generate binary rewards using a Bernoulli distribution with expectation given by our ground truth
    if not path.exists(constant.VARIABLES_FOLDER + rewards_fm_file):
        rewards_fm, expec_rewards_fm = recommender.create_rewards(ground_truth_fm)
        io.save(rewards_fm_file, (rewards_fm_file, rewards_fm))
        io.save(expec_rewards_fm_file, (expec_rewards_fm_file, expec_rewards_fm))

    print("Loading rewards fm...")
    io.load(rewards_fm_file, my_globals)

    print("Loading expectations rewards fm...")
    io.load(expec_rewards_fm_file, my_globals)

    # TEMP TRYING FOR ONLY ONE MODEL
    latent_factor = list(policies_fm.keys())[0]
    model = best_recommendation_est_system_fm[latent_factor]
    policies = policies_fm[latent_factor]["policies"]
    probability_policies = policies_fm[latent_factor]["probability_policies"]
        
    envy.determine_envy_freeness(policies, probability_policies, rewards_fm, expec_rewards_fm)

    # # Try algorithm for one model
    # envy.OCEF(policies_fm[latent_factor], rewards_fm, 0, 3, 1, 1, 0)