import os
from os import path
import numpy as np
import matplotlib.pyplot as plt
import time
import pyinputplus as pyip
import multiprocessing as mp
from sklearn.preprocessing import MinMaxScaler
import sys

# 1st party imports
from classes.recommender import Recommender
from classes.audit import Audit
from classes.experiment import Experiment
from lib import io
from lib import helper
from lib import rec_generator
from lib import plot
import constant

def create_recommender(data_set, normalized, model, model_type, params, perf, metric, ground_truth, temperature=1/5):
    """
    Creates one single Recommender object (i.e. a recommender system) and saves it locally in the framework

    Inputs:
        data_set                            - data set used to generate the recommender model 
                                              (is element of the <DATA_SETS> in src/constant.py)
        normalized                          - whether the ground truth preference scores on which the recommender model
                                              was build was min-max normalized   
        model                               - recommender model e.g. ALS or Funky SVD object 
        model_type                          - which algorithm the recommender system is build on
        params                              - the hyperparameter configuration of the model 
                                              {"latent factor: <latent_factor>: "reg": <regularization>, "alpha": <confidence weighing>}
        perf                                - performance metric used to validate and select model e.g. NDCG@K
        ground_truth                        - the true user-item preference scores on which the model was build 
                                              (trained, validated, tested) and formed its user-item recommendation policies
        temperature                         - hyperparameter of the recommender's system policy funcion which is a softmax
                                              distribution over all users' estimated preference scores                                      
    Outputs:
        string                              - file path of where the Recommender object is saved
    """
    recommender = Recommender(data_set=data_set,
                            normalized=normalized,
                            model=model, 
                            model_type=model_type, 
                            params=params, 
                            perf=perf, 
                            metric = metric,
                            ground_truth=ground_truth,
                            temperature=temperature)

    return helper.save_recommender(recommender)

if __name__ == "__main__":
    helper.init_directories()

    # Available data sets
    data_sets = constant.DATA_SETS
    
    print("**Continue with a DUMMY DEMO <d>? Or manual <m>?**")
    constant.DEBUG = True if pyip.inputMenu(["d", "m"]) == "d" else False
    print("\n")
    
    # For now leave out the "all" option to run experiments
    all_options = list(constant.EXPERIMENT_RUN_OPTIONS.keys())
    all_options.pop(0)
    
    # When debug mode is on skip this section and take dummy input
    if not constant.DEBUG:

        # Select experiments
        print("**EXPERIMENTS: Please specify which experiments to run**")
        experiment_choice = pyip.inputMenu(list(all_options))
        print("\n")
        
        experiments_chosen = all_options if experiment_choice == "all" else [experiment_choice]

        # Select data set(s)
        print("**DATA SETS: Please specify which (recommender system) data set**")
        options = list(data_sets.keys())
        options.insert(0, "all")
        data_set_choice = pyip.inputMenu(options)
        print("\n")
 
        print("=================================\n")

        data_sets_chosen = {v["label"]: k for k, v in data_sets.items()} if data_set_choice == "all" else {data_sets[data_set_choice]["label"]: data_set_choice}
        model_choice = {k: {"ground_truth": {}, "recommender": {}} for k in data_sets_chosen}

        for k, name in data_sets_chosen.items():  
            name = name.upper()

            # Specify ground truth model
            print("**{} - GROUND TRUTH MODEL: Which algorithm would you like to use to generate the ground truth?**".format(name))
            model_choice[k]["ground_truth"]["ALGORITHM"] = pyip.inputMenu(constant.ALGORITHM_GROUND_TRUTH)
            algorithm = model_choice[k]["ground_truth"]["ALGORITHM"] 
            print("\n")

            print("**{} - GROUND TRUTH MODEL: With which <performance metric>@K?***".format(name))
            options = list(constant.PERFORMANCE_METRICS[algorithm].keys())
            model_choice[k]["ground_truth"]["METRIC"] = pyip.inputMenu(options)
            metric = model_choice[k]["ground_truth"]["METRIC"] 
            print("{}@{}\n".format(metric, constant.PERFORMANCE_METRICS[algorithm][metric]["K"]))

            print("-------------------------------\n")

            # Specify recommender system model
            print("**{} - RECOMMENDER MODEL: Which algorithm would you like to use to estimate user preferences?**".format(name))
            model_choice[k]["recommender"]["ALGORITHM"] = pyip.inputMenu(constant.ALGORITHM_RECOMMENDER)
            algorithm = model_choice[k]["recommender"]["ALGORITHM"]
            print("\n")

            print("**{} - RECOMMENDER MODEL: With which <performance metric>@K?***".format(name))
            options = list(constant.PERFORMANCE_METRICS[algorithm].keys())
            model_choice[k]["recommender"]["METRIC"] = pyip.inputMenu(options)
            metric = model_choice[k]["recommender"]["METRIC"] 
            print("{}@{}\n".format(metric, constant.PERFORMANCE_METRICS[algorithm][metric]["K"]))
            
            print("**{} - RECOMMENDER MODEL: would you like for the input relevance scores to be NORMALIZED (Min-Max)?***".format(name))
            model_choice[k]["recommender"]["normalize"] = True if pyip.inputMenu(["y", "n"]) == "y" else "n"
            print("\n")

            print("=================================\n")
    else:
        # Run dummy user input mode
        experiment_choice = constant.DUMMY_EXPERIMENT_CHOICE
        experiments_chosen = all_options if experiment_choice == "all" else [experiment_choice]
        data_set_choice = constant.DUMMY_DATA_SET_CHOICE
        data_sets_chosen = {v["label"]: k for k, v in data_sets.items()} if data_set_choice == "all" else {data_sets[data_set_choice]["label"]: data_set_choice}
        model_choice = {k: model_choice for k, model_choice in constant.DUMMY_MODEL_CHOICE.items()}

    constant.MODELS_CHOSEN = model_choice
    my_globals = globals()

    # For each data set track whether a pipeline module is (re)build from scratch 
    new_build = {label:  {"gt": False, "recommender": False} for label, name in data_sets_chosen.items()}

    # Construct the pipeline -> run for all data sets requested by user
    for label, name in data_sets_chosen.items(): 
        set_name = name.upper()
        print("Starting for data set...{}".format(set_name))
    
        data_set = data_sets[name]
        gt_settings = model_choice[label]["ground_truth"]
        rec_settings = model_choice[label]["recommender"]

        # Define paths for variables
        var_path_rec = helper.get_var_path(data_set, constant.FOLDER_NAMES["rec_model"], rec_settings)
        var_path_gt = helper.get_var_path(data_set, constant.FOLDER_NAMES["gt_model"], gt_settings)

        # For results
        results_path = constant.RESULTS_FOLDER

        # For time logging
        log_path = helper.get_log_path(data_set)
        io.initialize_empty_file(log_path)
        
        if not path.exists(var_path_gt + constant.FILE_NAMES["gt"]):
            new_build[label]["gt"] = True


        ##########################
        #   GROUND TRUTH: considered as the true user-item preference scores
        ##########################

        # Generate ground truth matrix of user-item relevance scores
        exec(compile(open(data_set["filename"], "rb").read(), data_set["filename"], "exec"))
        
        ## LOAD: Load ground truth
        print("Loading ground truth {}...{}".format(set_name, var_path_gt + constant.FILE_NAMES["gt"]))
        io.load(var_path_gt + constant.FILE_NAMES["gt"], my_globals)

        # Whether to normalize the relevance scores to range [0, 1]
        # Because e.g. the (N)DCG metric doesn't work well with negative (feedback) scores   
        min_max_scaling = rec_settings["normalize"]
        # mask_negative_before_scaling = True

        # Normalize to range [0, 1]
        if min_max_scaling: 
            print("Min-max scaling ground truth relevance scores...")
            # Indices masks of negative and positive values
            pos_mask = ground_truth >= 0
            neg_mask = ground_truth < 0

            # Negative feedback we consider the lower bound and thus set to zero
            ground_truth = pos_mask * ground_truth

            # Scale relevance scores
            scaler = MinMaxScaler()
            model = scaler.fit(ground_truth)
            ground_truth = model.transform(ground_truth)
        

        ##########################
        #   RECOMMENDER SYSTEM: from here on we generate 144 models that simulate the preference estimation part of a recommender system
        ##########################

        # Hyperparameter spaces
        latent_factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        regularization = [0.001, 0.01, 0.1, 1.0]
        confidence_weighting = [0.1, 1.0, 10.0, 100.0] 
                
        configurations = helper.generate_hyperparameter_configurations(regularization, latent_factors, confidence_weighting)

        recommender_models_file = constant.FILE_NAMES["rec_model"] 
        recommender_models_file_path = var_path_rec + recommender_models_file

        # Generate models to simulate the recommender system's preference estimation
        if not path.exists(recommender_models_file_path):
            new_build[label]["recommender"] = True
            start = time.time()
            recommender_models = rec_generator.create_recommender_model(ground_truth, configurations, 
                                                                    rec_settings["ALGORITHM"], rec_settings["METRIC"], 
                                                                    split={"train": 0.7, "validation": 0.1})
            end = time.time() - start
            io.write_to_file(log_path, "Generating {} {}\n".format(recommender_models_file, str(end)))

            ## SAVE: Save recommender models that can estimate preferences
            io.save(recommender_models_file_path, (recommender_models_file, recommender_models))


        ## LOAD: Load recommender models that can estimate preferences
        print("Loading recommender preference estimation models {}...{}".format(set_name, recommender_models_file_path))
        io.load(recommender_models_file_path, my_globals)

    
        ##########################
        #   RECOMMENDER SYSTEM: from here on we continue with the best models we select which are either solely the overall best model
        #                       or the best model per latent_factor. Thus we have a list in the form:
        #                       {<latent_factor>: {<precision>: {"i_params": <index hyperparams config>, "params": <hyperparams config>, "p_val": <validation precision>, "model": <model>}}}}
        ##########################

        # Get the recommender models with best performances
        best_recommender_models = rec_generator.get_best_recommender_models(recommender_model, selection_mode="latent")

        # Only populate with recommender models once
        if len(os.listdir(constant.MODELS_FOLDER + data_set["var_folder"])) == 0:
            start = time.time()
            # Create Recommender system objects in parallel
            with mp.Pool(mp.cpu_count()) as pool:
                # Prepare arguments
                items = [(data_set, 
                        rec_settings["normalize"],
                        model["model"], 
                        rec_settings["ALGORITHM"], 
                        model["params"], 
                        model["perf"], 
                        rec_settings["METRIC"],
                        ground_truth) for model in best_recommender_models.values()]
                
                # Execute tasks and process results in order
                for result in pool.starmap(create_recommender, items):
                    print(f'Saved recommender: {result}', flush=True)
            # Process pool is closed automatically
            end = time.time() - start
            
            io.write_to_file(log_path, "Initializing {} recommenders {}\n".format(str(len(best_recommender_models)), str(end)))

    ##########################
    # AUDITING EXPERIMENTS
    ##########################
    # Running option "all" experiments not yet supported
        
    # EXPERIMENT 5.1: sources of envy
    if experiment_choice == "5.1":
        for label, name in data_sets_chosen.items():
            data_set = data_sets[name]
        
            recommenders = os.listdir(constant.MODELS_FOLDER + data_set["var_folder"])    
            recommenders = [helper.get_recommender(data_set, file_name, my_globals) for file_name in recommenders]
        
            experiments = Experiment(helper.get_log_path(data_set), experiments_chosen, recommenders)
            experiments.exp_5_1()
            data_set["experiment"] = experiments


        ##########################
        # RESULTS: Plot and save results of experiments
        ##########################
        # Average envy plotted together
        data = plot.gerenate_plot_data(data_sets_chosen, data_sets, "5.1", "avg_envy_users", "5.1_average_envy")
        plot.plot_experiment_line(data["lines"], "average envy", "number of factors", data["labels"], data["linestyles"], data["colors"], data["file_name"], x_upper_bound=128)

        # Proportion of envious users plotted together
        # experiment.audits[0].params["basic"]["epsilon"]
        data = plot.gerenate_plot_data(data_sets_chosen, data_sets, "5.1", "prop_envious_users", "5.1_prop_envy_users")
        plot.plot_experiment_line(data["lines"], "prop of envious users (epsilon = {})".format(0.05), "number of factors", data["labels"], data["linestyles"], data["colors"], data["file_name"], x_upper_bound=128)



    # End of logging
    for label, name in data_sets_chosen.items():
        data_set = data_sets[name]
        log_path = helper.get_log_path(data_set)

        if new_build[label]["gt"]:
            io.write_to_file(log_path, "ground_truth_model")
            io.write_to_file(log_path, model_choice[label]["ground_truth"], mode="json")
        
        if new_build[label]["recommender"]:
            io.write_to_file(log_path, "recommender_model")
            io.write_to_file(log_path, model_choice[label]["recommender"], mode="json")
        
        io.write_to_file(log_path, "-------------------END---------------------")