import os
import numpy as np
from os import path
import matplotlib.pyplot as plt
import sys

# 1st party imports
from lib import io
from lib import helper
from lib import recommender
from lib import envy
from lib import plot
import constant

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    my_globals = globals()

    data_sets = {0: {"name": "fm",
                    "filename": "user_artist.py",
                    "experiment": "5.1/",
                    "vars": {
                        "ground_truth": None,
                        "recommendation_system_est_models": None,
                        "preference_estimates": None,
                        "policies": None
                    }}}

    # For all datas sets simulate recommendation system and run experiments
    for i, data_set in data_sets.items(): 
        if data_set["name"] == "fm":
            # Infix of folder to perform I/O-operations on for specifically this data set (reading, writing etc.)
            IO_INFIX = constant.VAR_SUB_FOLDER["fm"]
            # Variable name extension
            VAR_EXT = constant.VAR_EXT["fm"]
        else:
            IO_INFIX = constant.VAR_SUB_FOLDER["mv"]
            VAR_EXT = constant.VAR_EXT["movie"]

        # Full path where variables of I/O operations are stored
        IO_PATH = constant.VARIABLES_FOLDER + IO_INFIX

        filename = data_set["filename"]

        # Generate ground truth of user-item preferences
        exec(compile(open(filename, "rb").read(), filename, "exec"))

        ## LOAD: Load ground truth
        io.load(IO_INFIX + "ground_truth", my_globals)

        if data_set["name"] == "fm":
            print("Loading ground truth FM...")
            ground_truth = ground_truth_fm
        else:
            print("Loading ground truth MOVIE...")
            ground_truth = ground_truth_movie

        
        ##########################
        #   RECOMMENDER SYSTEM: from here on we generate 144 preference estimation models one for each hyperparameter combination
        ##########################

        # Hyperparameter spaces
        latent_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        regularization = [0.001, 0.01, 0.1, 1.0]
        confidence_weighting = [0.1, 1.0, 10.0, 100.0]
        
        configurations = helper.generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors)

        # Generate models to simulate a recommender system's preference estimation
        recommendation_system_est_models_file = "recommendation_system_est_models" 
        recommendation_system_est_models_file_path = IO_PATH + recommendation_system_est_models_file
        recommendation_system_est_models_var_name = recommendation_system_est_models_file + VAR_EXT 

        # Only when not yet generated
        if not path.exists(recommendation_system_est_models_file_path):
            recommendation_system_est_models = recommender.create_recommendation_est_system(ground_truth, configurations, split={"train": 0.7, "validation": 0.1})

            ## SAVE: Save models that can estimate preferences as a recommender system would
            io.save(IO_INFIX + recommendation_system_est_models_file, (recommendation_system_est_models_var_name, recommendation_system_est_models))
        
        ## LOAD: Load models that can estimate preferences
        io.load(IO_INFIX + recommendation_system_est_models_file, my_globals)
        
        if data_set["name"] == "fm":
            print("Loading recommender preference estimation models FM...")
            recommendation_system_est_models = recommendation_system_est_models_fm
        else:
            print("Loading recommender preference estimation models MOVIE...")
            recommendation_system_est_models = recommendation_system_est_models_movie

        ##########################
        #   RECOMMENDER SYSTEM: from here on we continue with the best models we select which are either the overall best model
        #                       or the best model per latent_factor. Thus we have a list in the form:
        #                       {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}}
        ##########################

        # Get the recommender preference estimation models with best performances
        best_recommendation_est_system = recommender.select_best_recommendation_est_system(recommendation_system_est_models, select_mode="latent")

        preference_estimates = {}

        # Use the model to simulate a recommender system’s estimation of preferences
        for latent_factor, data in best_recommendation_est_system.items():
            preference_estimates[latent_factor] = recommender.recommendation_estimation(data["model"])

        policies = {}

        # Use the estimated preferences to generate policies
        for latent_factor, data in preference_estimates.items():
            policies_, probability_policies = recommender.create_recommendation_policies(data)
            policies[latent_factor] = {"policies": policies_, "probability_policies": probability_policies}

        ##########################
        #   REWARDS: the rewards are independent of the recommender system model, thus we only generate it once
        ##########################
                
        rewards_file = "rewards"
        rewards_file_path = IO_PATH + rewards_file
        rewards_var_name = rewards_file + VAR_EXT

        expec_rewards_file = "expec_rewards"
        expec_rewards_file_path = IO_PATH + expec_rewards_file
        expec_rewards_var_name = expec_rewards_file + VAR_EXT

        # We generate binary rewards using a Bernoulli distribution with expectation given by our ground truth
        if not path.exists(rewards_file_path):
            rewards, expec_rewards = recommender.create_rewards(ground_truth)
            io.save(IO_INFIX + rewards_file, (rewards_var_name, rewards))
            io.save(IO_INFIX + expec_rewards_file, (expec_rewards_var_name, expec_rewards))

        # LOAD
        print("Loading rewards...")
        io.load(IO_INFIX + rewards_file, my_globals)

        print("Loading expectations rewards...")
        io.load(IO_INFIX + expec_rewards_file, my_globals)

        # Experiment 5.1: sources of envy
        experiment = data_set["experiment"]
        envy_free_file = "envy_free"
        avg_envy_user_file = "avg_envy_user"
        prop_envious_users_file = "prop_envious_users"

        print(data_set["experiment"])

    #     experiment_file_path = constant.VARIABLES_FOLDER + constant.EXPERIMENTS_FOLDER + experiment + avg_envy_user_file

    #     # Only run experiment when no results yet
    #     if not path.exists(experiment_file_path):
    #         # Experiment results
    #         keys = policies.keys()
    #         envy_free = {key: {} for key in keys}
    #         avg_envy_user = {key: {} for key in keys}
    #         prop_envious_users = {key: {} for key in keys}
            
    #         # For latent factor's model perform experiment
    #         for latent_factor in keys:
    #             print("latent_factor", latent_factor)
    #             model = best_recommendation_est_system[latent_factor]
    #             policies = policies[latent_factor]["policies"]
    #             probability_policies = policies[latent_factor]["probability_policies"]
                    
    #             envy_results = envy.determine_envy_freeness(policies, probability_policies, rewards, expec_rewards)
    #             envy_free[latent_factor] = envy_results["envy_free"]
    #             avg_envy_user[latent_factor] = envy_results["avg_envy_user"]
    #             prop_envious_users[latent_factor] = envy_results["prop_envious_users"]

    #         io.save(constant.EXPERIMENTS_FOLDER + experiment + envy_free_file, (envy_free_file, envy_free))
    #         io.save(constant.EXPERIMENTS_FOLDER + experiment + avg_envy_user_file, (avg_envy_user_file, avg_envy_user))
    #         io.save(constant.EXPERIMENTS_FOLDER + experiment + prop_envious_users_file, (prop_envious_users_file, prop_envious_users))

    #     # Load results experiment
    #     print("Loading results of experiment", experiment,"...")
    #     io.load(constant.EXPERIMENTS_FOLDER + experiment + envy_free_file, my_globals)
    #     io.load(constant.EXPERIMENTS_FOLDER + experiment + avg_envy_user_file, my_globals)
    #     io.load(constant.EXPERIMENTS_FOLDER + experiment + prop_envious_users_file, my_globals)


    # plot.plot([avg_envy_user])
    
    # # Try algorithm for one model
    # envy.OCEF(policies_fm[latent_factor], rewards_fm, 0, 3, 1, 1, 0)