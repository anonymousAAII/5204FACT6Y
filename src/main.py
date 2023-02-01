import os
import numpy as np
import numpy.ma as ma
from os import path
import matplotlib.pyplot as plt
import time
import pyinputplus as pyip
import multiprocessing as mp
from sklearn.metrics import ndcg_score, dcg_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from glob import glob

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

    data_sets = constant.DATA_SETS
    
    print("**Continue with a DUMMY DEMO <d>? Or manual <m>?**")
    constant.DEBUG = True if pyip.inputMenu(["d", "m"]) == "d" else False
    print("\n")
    
    # When debug mode is on skip this section and take dummy input
    if not constant.DEBUG:
        # For now leave out the "all" option to run experiments
        all_options = list(constant.EXPERIMENT_RUN_OPTIONS.keys())
        all_options.pop(0)

        # To select experiments
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
        # Available model options per data set through CLI
        model_choice = {k: {"ground_truth": {}, "recommender": {}} for k in data_sets_chosen}

        for k, name in data_sets_chosen.items():  
            name = name.upper()
            # For ground truth model
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

            # For recommender system
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
        experiment_choice = constant.DUMMY_EXPERIMENT_CHOICE
        data_set_choice = constant.DUMMY_DATA_SET_CHOICE
        data_sets_chosen = {v["label"]: k for k, v in data_sets.items()} if data_set_choice == "all" else {data_sets[data_set_choice]["label"]: data_set_choice}
        # Available model options per data set through CLI
        model_choice = {k: constant.DUMMY_MODEL_CHOICE for k in data_sets_chosen}

    constant.MODELS_CHOSEN = model_choice
    my_globals = globals()

    # For all datas sets requested by user generate recommender systems
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
        io.write_to_file(log_path, model_choice, mode="json")
        
        # Generate ground truth matrix of user-item relevance scores
        exec(compile(open(data_set["filename"], "rb").read(), data_set["filename"], "exec"))
        
        ## LOAD: Load ground truth
        print("Loading ground truth {}...{}".format(set_name, var_path_gt + constant.FILE_NAMES["gt"]))
        io.load(var_path_gt + constant.FILE_NAMES["gt"], my_globals)

        # Whether to normalize the relevance scores to range [0, 1]
        # Because e.g. the (N)DCG metric doesn't work well with negative scores   
        min_max_scaling = rec_settings["normalize"]

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
        
        data_set["vars"]["ground_truth"] = ground_truth

        ##########################
        #   RECOMMENDER SYSTEM: from here on we generate 144 preference estimation models, one for each hyperparameter combination
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

    # EXPERIMENT 5.1: sources of envy
    for label, name in data_sets_chosen.items():
        data_set = data_sets[name]
        recommenders = os.listdir(constant.MODELS_FOLDER + data_set["var_folder"])    
        recommenders = [helper.get_recommender(data_set, file_name, my_globals) for file_name in recommenders]
       
        experiments = Experiment(helper.get_log_path(data_set), experiment_choice, recommenders)
        experiments.exp_5_1()
        data_set["experiment"] = experiments


    ##########################
    # RESULTS: Plot and save results of experiments
    ##########################
    # Average envy plotted together
    lines = []  
    labels = []
    linestyles = [] 
    colors = []
    
    ds = []

    for label, name in data_sets_chosen.items(): 
        data_set = data_sets[name]
        experiment = data_set["experiment"]
        results = experiment.experiment_results["5.1"] 

        plot_style = data_set["plot_style"]
        lines.append(results["avg_envy_users"])
        labels.append(plot_style["label"])  
        linestyles.append(plot_style["linestyle"])
        colors.append(plot_style["color"])    
        ds.append(label + experiment.recommenders[0].model_type)   

    "_".join(ds)
    plot.plot_experiment_line(lines, "average envy", "number of factors", labels, linestyles, colors, "5.1_average_envy_{}".format(ds), x_upper_bound=128)

    # Proportion of envious users plotted together
    lines = []  
    labels = []
    linestyles = [] 
    colors = []

    ds = []
    epsilon = None

    for label, name in data_sets_chosen.items(): 
        data_set = data_sets[name]
        experiment = data_set["experiment"]
        results = experiment.experiment_results["5.1"] 
        epsilon = experiment.audits[0].params["basic"]["epsilon"]
    
        plot_style = data_set["plot_style"]
        lines.append(results["prop_envious_users"])
        labels.append(plot_style["label"])  
        linestyles.append(plot_style["linestyle"])
        colors.append(plot_style["color"])    
        ds.append(label + experiment.recommenders[0].model_type)   

    "_".join(ds)
    plot.plot_experiment_line(lines, "prop of envious users (epsilon = {})".format(epsilon), "number of factors", labels, linestyles, colors, "prop_envy_users_{}".format(ds), x_upper_bound=128)

    # # # Try algorithm for one model
    # # envy.OCEF(policies_fm[latent_factor], rewards_fm, 0, 3, 1, 1, 0)