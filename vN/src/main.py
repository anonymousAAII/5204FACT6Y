import os
import numpy as np
from os import path
import matplotlib.pyplot as plt
import time

# 1st party imports
from lib import io
from lib import helper
from lib import recommender
from lib import envy
from lib import plot
import constant

if __name__ == "__main__":
    my_globals = globals()

    # Data sets to perform experiments on
    data_sets = {   0: {"name": "movie",
                        "filename": "user_movie.py",
                        "vars": {
                            "ground_truth": None,
                            "recommendation_system_est_models": None,
                            "preference_estimates": None,
                            "recommendation_policies": None,
                            "rewards": None,
                            "expec_rewards": None
                        },
                        "experiments_results": {}
                        },
                    1: {"name": "fm",
                        "filename": "user_artist.py",
                        "vars": {
                            "ground_truth": None,
                            "recommendation_system_est_models": None,
                            "preference_estimates": None,
                            "recommendation_policies": None,
                            "rewards": None,
                            "expec_rewards": None
                        },
                        "experiments_results": {}
                    }
                }

    # For all datas sets simulate recommendation system and run experiments
    for i, data_set in data_sets.items(): 
        print("Starting for data set", data_set["name"], "...")

        # Define 'GLOBALS'
        if data_set["name"] == "fm":
            # Infix of folder to perform I/O-operations on for specifically this data set (reading, writing etc.)
            IO_INFIX = constant.VAR_SUB_FOLDER["fm"]
            # Variable name extension
            VAR_EXT = constant.VAR_EXT["fm"]
        else:
            IO_INFIX = constant.VAR_SUB_FOLDER["movie"]
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
            ground_truth = ground_truth_mv

        data_set["vars"]["ground_truth"] = ground_truth

        ##########################
        #   RECOMMENDER SYSTEM: from here on we generate 144 preference estimation models, one for each hyperparameter combination
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
            recommendation_system_est_models = recommendation_system_est_models_mv

        data_set["vars"]["recommendation_system_est_models"] = recommendation_system_est_models

        ##########################
        #   RECOMMENDER SYSTEM: from here on we continue with the best models we select which are either solely the overall best model
        #                       or the best model per latent_factor. Thus we have a list in the form:
        #                       {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}}
        ##########################

        # Get the recommender preference estimation models with best performances
        best_recommendation_est_system = recommender.select_best_recommendation_est_system(recommendation_system_est_models, select_mode="latent")

        preference_estimates = {}

        # Use the model to simulate a recommender system’s estimation of preferences
        for latent_factor, data in best_recommendation_est_system.items():
            preference_estimates[latent_factor] = recommender.recommendation_estimation(data["model"])

        data_set["vars"]["preference_estimates"] = preference_estimates

        recommendation_policies = {}

        # Use the estimated preferences to generate policies
        for latent_factor, data in preference_estimates.items():
            recommendations, policies = recommender.create_recommendation_policies(data)
            recommendation_policies[latent_factor] = {"recommendations": recommendations, "policies": policies}

        data_set["vars"]["recommendation_policies"] = recommendation_policies

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


        # LOAD: Load binary rewards and expectation of rewards
        io.load(IO_INFIX + rewards_file, my_globals)
        io.load(IO_INFIX + expec_rewards_file, my_globals)

        if data_set["name"] == "fm":
            print("Loading rewards FM...")
            print("Loading expectations rewards...")
            rewards = rewards_fm
            expec_rewards = expec_rewards_fm
        else:
            print("Loading rewards MOVIE...")
            print("Loading expectations rewards...")
            rewards = rewards_mv
            expec_rewards = expec_rewards_mv

        data_set["vars"]["rewards"] = rewards
        data_set["vars"]["expec_rewards"] = expec_rewards

        # Experiment 5.1: sources of envy
        experiment_dir = "5.1/"
        experiment_dir_path = IO_PATH + experiment_dir

        envy_free_file = "envy_free"
        avg_envy_user_file = "avg_envy_user"
        prop_envious_users_file = "prop_envious_users"
        
        start = time.time()

        # Run experiment 5.1
        if not path.exists(experiment_dir_path):
            print("Creating directory for experiment...", experiment_dir)
            os.mkdir(experiment_dir_path)
        
        # Only perform experiment when not yet executed
        if len(os.listdir(experiment_dir_path)) == 0:    
            print("Running experiment...", experiment_dir)    
        
            keys = recommendation_policies.keys()
            envy_free = {key: {} for key in keys}
            avg_envy_user = {key: {} for key in keys}
            prop_envious_users = {key: {} for key in keys}
            
            # For latent factor's model perform experiment
            for latent_factor in keys:
                print("<latent_factor>", latent_factor)
                model = best_recommendation_est_system[latent_factor]
                
                recommendations = recommendation_policies[latent_factor]["recommendations"]
                policies = recommendation_policies[latent_factor]["policies"]
                    
                envy_results = envy.determine_envy_freeness(recommendations, policies, rewards, expec_rewards)
                envy_free[latent_factor] = envy_results["envy_free"]
                avg_envy_user[latent_factor] = envy_results["avg_envy_user"]
                prop_envious_users[latent_factor] = envy_results["prop_envious_users"]

            # Save results of experiment
            io.save(IO_INFIX + experiment_dir + envy_free_file, (envy_free_file, envy_free))
            io.save(IO_INFIX + experiment_dir + avg_envy_user_file, (avg_envy_user_file, avg_envy_user))
            io.save(IO_INFIX + experiment_dir + prop_envious_users_file, (prop_envious_users_file, prop_envious_users))

        end = time.time()
        print(end - start)

        # Load results experiment
        print("Loading results of experiment...", experiment_dir)
        io.load(IO_INFIX + experiment_dir + envy_free_file, my_globals)
        io.load(IO_INFIX + experiment_dir + avg_envy_user_file, my_globals)
        io.load(IO_INFIX + experiment_dir + prop_envious_users_file, my_globals)

        data_set["experiments_results"]["5.1"] = {
                                                    envy_free_file: envy_free,
                                                    avg_envy_user_file: avg_envy_user,
                                                    prop_envious_users_file: prop_envious_users
                                                }        

    # Average envy plotted together    
    plot.plot_experiment_5_1A([data_sets[0]["experiments_results"]["5.1"]["avg_envy_user"], data_sets[1]["experiments_results"]["5.1"]["avg_envy_user"]],
                            "average envy", "number of factors", "MovieLens", "Last.fm", "average_envy")

    # print(data_sets[1]["experiments_results"]["5.1"]["prop_envious_users"])

    # Proportion of envious users plotted together
    plot.plot_experiment_5_1A([data_sets[0]["experiments_results"]["5.1"]["prop_envious_users"], data_sets[1]["experiments_results"]["5.1"]["prop_envious_users"]],
                            "prop of envious users (epsilon = 0.05)", "number of factors", "MovieLens", "Last.fm", "prop_envious_users")

    # Average envy plotted seperately
    plot.plot_experiment_single([data_sets[0]["experiments_results"]["5.1"]["avg_envy_user"]],
                            "average envy", "number of factors", "MovieLens", "average_envy_mv")
    plot.plot_experiment_single([data_sets[1]["experiments_results"]["5.1"]["avg_envy_user"]],
                            "average envy", "number of factors", "Last.fm", "average_envy_fm", 1)


    # # Try algorithm for one model
    # envy.OCEF(policies_fm[latent_factor], rewards_fm, 0, 3, 1, 1, 0)