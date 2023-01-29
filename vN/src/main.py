import os
import numpy as np
from os import path
import matplotlib.pyplot as plt
import time
import pyinputplus as pyip
import multiprocessing as mp

# 1st party imports
from lib import io
from lib import helper
from lib import recommender
from lib import envy
from lib import plot
import constant

def preferences_to_policies(latent_factor, model, ground_truth, algorithm):
    preference_estimates = recommender.recommendation_estimation(model, ground_truth, algorithm)
    recommendation_policies = recommender.create_recommendation_policies(preference_estimates)
    return {"latent_factor": latent_factor, "recommendation_policies": {"preferences": preference_estimates, "recommendations": recommendation_policies["recommendations"], "policies": recommendation_policies["policies"]}}

if __name__ == "__main__":
    # To save the experiment results
    experiment_results = {k: {} for k in constant.EXPERIMENT_RUN_OPTIONS.keys() if k not in {"all"}}

    print("**Please specify which experiments to run**")
    experiment_choice = pyip.inputMenu(list(constant.EXPERIMENT_RUN_OPTIONS.keys()))

    # Data sets to perform experiments on
    data_sets = {"all": {"name": "all"},
                "movie": {"name": "mv",
                        "filename": "user_movie.py",
                        "vars": {
                            "ground_truth": None,
                            "recommendation_system_est_models": None,
                            "recommendation_policies": None,
                            "rewards": None,
                            "expec_rewards": None
                        }
                    },
                "fm": {"name": "fm",
                        "filename": "user_artist.py",
                        "vars": {
                            "ground_truth": None,
                            "recommendation_system_est_models": None,
                            "recommendation_policies": None,
                            "rewards": None,
                            "expec_rewards": None
                        },
                    },
                }

    print("**Please specify for which (recommender system) data set**")
    data_set_choice = pyip.inputMenu(list(data_sets.keys()))
    
    print("**Which algorithm would you like the recommender system to use to estimate user preferences?**")
    ALGORITHM_CHOICE = pyip.inputMenu(constant.ALGORITHM)

    print("**With which <performance metric>@K would you like the recommender system to be trained, validated and selected?***")
    constant.PERFORMANCE_METRIC = pyip.inputMenu(list(constant.PERFORMANCE_METRIC_VARS.keys()))
    print("{}@{}".format(constant.PERFORMANCE_METRIC, constant.PERFORMANCE_METRIC_VARS[constant.PERFORMANCE_METRIC]["K"]))

    # To save figures/plots
    for data_set in data_sets.values():
        folder1 = constant.RESULTS_FOLDER + data_set["name"]
        if not path.exists(folder1):
            os.mkdir(folder1)

        for algorithm in constant.ALGORITHM:
            folder = folder1 + "/" + algorithm + "/"
            if not path.exists(folder):
                os.mkdir(folder)

    my_globals = globals()

    # For all datas sets simulate recommendation system and run experiments
    for key, data_set in data_sets.items(): 
        # Only execute code for requested data sets by user input
        if key == "all" or (data_set_choice != "all" and key != data_set_choice):
            continue

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

        # Possibility for different recommender model algorithms   
        if not path.exists(constant.VARIABLES_FOLDER + IO_INFIX + ALGORITHM_CHOICE + "/"):
            print("Creating variables directory for {} recommender model...".format(ALGORITHM_CHOICE))
            os.mkdir(constant.VARIABLES_FOLDER + IO_INFIX + ALGORITHM_CHOICE + "/")

        # Generate ground truth of user-item preferences
        exec(compile(open(data_set["filename"], "rb").read(), data_set["filename"], "exec"))
        
        ## LOAD: Load ground truth
        io.load(IO_INFIX + "ground_truth", my_globals)        

        if data_set["name"] == "fm":
            print("Loading ground truth FM...{}".format(IO_INFIX + "ground_truth"))
            ground_truth = ground_truth_fm
        else:
            print("Loading ground truth MOVIE...{}".format(IO_INFIX + "ground_truth"))
            ground_truth = ground_truth_mv

        data_set["vars"]["ground_truth"] = ground_truth

        IO_INFIX = IO_INFIX + ALGORITHM_CHOICE + "/"

        # Full path where variables of I/O operations are stored
        IO_PATH = constant.VARIABLES_FOLDER + IO_INFIX

        ##########################
        #   RECOMMENDER SYSTEM: from here on we generate 144 preference estimation models, one for each hyperparameter combination
        ##########################

        # Hyperparameter spaces
        latent_factors = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
        regularization = [0.001, 0.01, 0.1, 1.0]
        confidence_weighting = [0.1, 1.0, 10.0, 100.0]
        
        configurations = helper.generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors)

        # Generate models to simulate a recommender system's preference estimation
        recommendation_system_est_models_file = "recommendation_system_est_models" 
        recommendation_system_est_models_file_path = IO_PATH + recommendation_system_est_models_file
        recommendation_system_est_models_var_name = recommendation_system_est_models_file + VAR_EXT 

        # Only when not yet generated
        if not path.exists(recommendation_system_est_models_file_path):
            start = time.time()
            recommendation_system_est_models = recommender.create_recommendation_est_system(ground_truth, configurations, ALGORITHM_CHOICE, split={"train": 0.7, "validation": 0.1})

            ## SAVE: Save models that can estimate preferences as a recommender system would
            io.save(IO_INFIX + recommendation_system_est_models_file, (recommendation_system_est_models_var_name, recommendation_system_est_models))
            end = time.time() - start
            io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE[data_set["name"]], "Generating " + recommendation_system_est_models_file + "  " + str(end) + "\n")

        ## LOAD: Load models that can estimate preferences
        io.load(IO_INFIX + recommendation_system_est_models_file, my_globals)
        
        if data_set["name"] == "fm":
            print("Loading recommender preference estimation models FM...{}".format(IO_INFIX + recommendation_system_est_models_file, my_globals))
            recommendation_system_est_models = recommendation_system_est_models_fm
        else:
            print("Loading recommender preference estimation models MOVIE...{}".format(IO_INFIX + recommendation_system_est_models_file, my_globals))
            recommendation_system_est_models = recommendation_system_est_models_mv

        data_set["vars"]["recommendation_system_est_models"] = recommendation_system_est_models


        ##########################
        #   RECOMMENDER SYSTEM: from here on we continue with the best models we select which are either solely the overall best model
        #                       or the best model per latent_factor. Thus we have a list in the form:
        #                       {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}}
        ##########################

        # Get the recommender preference estimation models with best performances
        best_recommendation_est_system = recommender.select_best_recommendation_est_system(recommendation_system_est_models, select_mode="latent")
        
        recommendation_policies = {}

        recommendation_policies_file = "recommendation_policies"
        recommendation_policies_file_path = IO_PATH + recommendation_policies_file
        recommendation_policies_var_name = recommendation_policies_file + VAR_EXT

        if not path.exists(recommendation_policies_file_path):
            start = time.time()

            # Create and configure the process pool
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(latent_factor, data["model"], ground_truth, ALGORITHM_CHOICE) for latent_factor, data in best_recommendation_est_system.items()]
                # Execute tasks and process results in order
                for result in pool.starmap(preferences_to_policies, items):
                    recommendation_policies[result["latent_factor"]] = result["recommendation_policies"]
                    # print(f'Got result: {result}', flush=True)
            # Process pool is closed automatically

            end = time.time() - start            
            io.save(IO_INFIX + recommendation_policies_file, (recommendation_policies_var_name, recommendation_policies))
            io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE[data_set["name"]], "Construct recommendation_policies   " + str(end) + "\n")

        # LOAD: Load recommendation policies
        io.load(IO_INFIX + recommendation_policies_file, my_globals)

        if data_set["name"] == "fm":
            print("Loading recommendation policies FM...{}".format(IO_INFIX + recommendation_policies_file, my_globals))
            recommendation_policies = recommendation_policies_fm
        else:
            print("Loading recommendation policies rewards...{}".format(IO_INFIX + recommendation_policies_file, my_globals))
            recommendation_policies = recommendation_policies_mv

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
        if not path.exists(rewards_file_path) or not path.exists(expec_rewards_file_path):
            start = time.time()
            rewards, expec_rewards = recommender.create_rewards(ground_truth)
            io.save(IO_INFIX + rewards_file, (rewards_var_name, rewards))
            io.save(IO_INFIX + expec_rewards_file, (expec_rewards_var_name, expec_rewards))
            end = time.time() - start
            io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE[data_set["name"]], "Generate rewards (expectation)   " + str(end) + "\n")

        # LOAD: Load binary rewards and expectation of rewards
        io.load(IO_INFIX + rewards_file, my_globals)
        io.load(IO_INFIX + expec_rewards_file, my_globals)

        if data_set["name"] == "fm":
            print("Loading rewards FM...{}".format(IO_INFIX + rewards_file, my_globals))
            print("Loading expectations rewards...{}".format(IO_INFIX + expec_rewards_file, my_globals))
            rewards = rewards_fm
            expec_rewards = expec_rewards_fm
        else:
            print("Loading rewards MOVIE...{}".format(IO_INFIX + rewards_file, my_globals))
            print("Loading expectations rewards...{}".format(IO_INFIX + expec_rewards_file, my_globals))
            rewards = rewards_mv
            expec_rewards = expec_rewards_mv

        data_set["vars"]["rewards"] = rewards
        data_set["vars"]["expec_rewards"] = expec_rewards


        ##########################
        # EXPERIMENT 5.1: sources of envy
        ##########################

        if experiment_choice == "5.1" or experiment_choice == "all":
            label = experiment_choice if experiment_choice != "all" else "5.1"            
            experiment_dir = constant.EXPERIMENT_RUN_OPTIONS[label]["experiment_dir"]
            experiment_dir_path = IO_PATH + experiment_dir

            envy_free_file = "envy_free"
            avg_envy_user_file = "avg_envy_user"
            prop_envious_users_file = "prop_envious_users"
            
            # Run experiment 5.1
            if not path.exists(experiment_dir_path):
                print("Creating directory for experiment...", experiment_dir)
                os.mkdir(experiment_dir_path)
    
            # Only perform experiment when not yet executed
            if len(os.listdir(experiment_dir_path)) == 0:    
                start = time.time()
                print("**Running experiment...", experiment_dir)    
            
                keys = recommendation_policies.keys()
                envy_free = {key: {} for key in keys}
                avg_envy_user = {key: {} for key in keys}
                prop_envious_users = {key: {} for key in keys}
                
                # For latent factor's model perform experiment
                for latent_factor in keys:
                    print("<latent_factor>", latent_factor)
                    model = best_recommendation_est_system[latent_factor]
                    
                    preferences = recommendation_policies[latent_factor]["preferences"]
                    recommendations = recommendation_policies[latent_factor]["recommendations"]
                    policies = recommendation_policies[latent_factor]["policies"]
                        
                    envy_results = envy.determine_envy_freeness(recommendations, policies, rewards, expec_rewards, preferences)
                    envy_free[latent_factor] = envy_results["envy_free"]
                    avg_envy_user[latent_factor] = envy_results["avg_envy_user"]
                    prop_envious_users[latent_factor] = envy_results["prop_envious_users"]

                # Save results of experiment
                io.save(IO_INFIX + experiment_dir + envy_free_file, (envy_free_file, envy_free))
                io.save(IO_INFIX + experiment_dir + avg_envy_user_file, (avg_envy_user_file, avg_envy_user))
                io.save(IO_INFIX + experiment_dir + prop_envious_users_file, (prop_envious_users_file, prop_envious_users))
                
                end = time.time() - start
                io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE[data_set["name"]], "Experiment {}   ".format(label) + str(end) + "\n")

            # Load results experiment
            print("Loading results of experiment...{}".format(IO_INFIX + experiment_dir))
            io.load(IO_INFIX + experiment_dir + envy_free_file, my_globals)
            io.load(IO_INFIX + experiment_dir + avg_envy_user_file, my_globals)
            io.load(IO_INFIX + experiment_dir + prop_envious_users_file, my_globals)

            experiment_results[label][data_set["name"]] = {"data": 
                                                                {
                                                                    envy_free_file: envy_free,
                                                                    avg_envy_user_file: avg_envy_user,
                                                                    prop_envious_users_file: prop_envious_users
                                                                },
                                                            "label": constant.DATA_LABELS[data_set["name"]],
                                                            "linestyle": constant.DATA_LINESTYLES[data_set["name"]],
                                                            "color": constant.DATA_COLORS[data_set["name"]]
                                                            }        

    
        io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE[data_set["name"]], "-------------------------------------------------------\n")

    
    ##########################
    # RESULTS: Plot and save results of experiments
    ##########################
    if experiment_choice == "all":
        plot_all = True
    
    # Experiment 5.1
    if plot_all or experiment_choice == "5.1":
        label = experiment_choice if experiment_choice != "all" else "5.1"

        # Average envy plotted together
        lines = []  
        labels = []
        linestyles = [] 
        colors = []

        for name, data in experiment_results[label].items():
            lines.append(data["data"]["avg_envy_user"])
            labels.append(data["label"])  
            linestyles.append(data["linestyle"])
            colors.append(data["color"])          


        plot.plot_experiment_line(lines, "average envy", "number of factors", labels, linestyles, colors, "average_envy", x_upper_bound=128)

        # Proportion of envious users plotted together
        lines = []  
        labels = []
        linestyles = [] 
        colors = []

        for name, data in experiment_results[label].items():
            lines.append(data["data"]["prop_envious_users"])
            labels.append(data["label"])  
            linestyles.append(data["linestyle"])
            colors.append(data["color"])              

        plot.plot_experiment_line(lines, "prop of envious users (epsilon = 0.05)", "number of factors", labels, linestyles, colors, "prop_envy_users", x_upper_bound=128)

    # # Try algorithm for one model
    # envy.OCEF(policies_fm[latent_factor], rewards_fm, 0, 3, 1, 1, 0)