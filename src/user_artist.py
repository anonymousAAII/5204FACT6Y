####
# vN/src/user_artist.py
#
# Processes Last.fm data set to generate the ground truth which forms the base of further code
#
# Data set thanks to:
# - Last.fm website, http://www.lastfm.com
#
# Iván Cantador, Peter Brusilovsky, and Tsvi Kuflik. \
# “2nd Workshop on Information Heterogeneity and Fusion in Recommender Systems (HetRec 2011)”. 
# In: Proceedings of the 5th ACM conference on Recommender systems. 
# RecSys 2011. Chicago, IL, USA: ACM, 2011.
###
import pandas as pd
import numpy as np
from datetime import datetime
import scipy
from scipy import sparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from os import path
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from implicit.evaluation import ndcg_at_k, precision_at_k
from pandas.api.types import CategoricalDtype
import multiprocessing as mp
from tqdm import tqdm
import time
import sys

# 1st party imports
from lib import helper
from lib import io
import constant

def train_model(R_coo, configurations, seed, metric, model_type):
    """
    For a random seed trains a model on a sparse matrix for all given hyperparameter combinations.

    :R_coo:             sparse matrix
    :configurations:    hyperparameters' space
    :seed:              seed (counter)
    """
    # Create 70%/10%/20% train/validation/test data split of the user-item top 2500 listening counts
    train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=0.7)
    validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=1/3)

    # Logistic Matrix Factorization    
    if model_type == "lmf":
        # For retrieving original state of the data sets
        train_tmp = train.copy()
        validation_tmp = validation.copy()
        test_tmp = test.copy()    

    # To safe performance per hyperparameter combination as {<precision>: <hyperparameter_id>}
    performance_per_configuration = {}

    K = constant.PERFORMANCE_METRICS[model_type][metric]["K"]
    perf_base = 0
    model_best = None

    print("Start training for {}th seed".format(seed + 1))
    
    # Hyperparameter tuning through grid search
    for i in tqdm(range(len(configurations))):
        hyperparameters = configurations[i]

        # Initialize model
        if model_type == "lmf":
            print("Logistic Matrix Factorization (LMF)...")
            # Notice you can't specify the confidence weighing parameter alpha here so it is applied to the data sets below
            model = LogisticMatrixFactorization(factors=hyperparameters["latent_factor"], 
                                                regularization=hyperparameters["reg"])
            train = hyperparameters["alpha"] * train_tmp
            validation = hyperparameters["alpha"] * validation_tmp
            test = hyperparameters["alpha"] * test_tmp
        elif model_type == "als":
            print("Alternating Least Squares (ALS)...")
            model = AlternatingLeastSquares(factors=hyperparameters["latent_factor"], 
                                            regularization=hyperparameters["reg"],
                                            alpha=hyperparameters["alpha"])
        else:
            sys.exit('Model type unspecified')

        # Train matrix factorization algorithm on the train set
        model.fit(train, show_progress=False)        

        # Benchmark model performance using validation set
        if metric == "ndcg":
            perf = ndcg_at_k(model, train, validation, K=K, show_progress=False)
        elif metric == "precision":
            perf = precision_at_k(model, train, validation, K=K, show_progress=False)
        else:
            sys.exit('Performance metric type unspecified')
        
        print("Seed {}: {}@{} {}".format((seed + 1), metric, K, perf))

        # When current model outperforms previous one update tracking states
        if perf > perf_base:
            perf_base = perf
            model_best = model

        performance_per_configuration[perf] = i

    print("Training ended for {}th seed".format(seed + 1))
    
    hyperparams_optimal = configurations[performance_per_configuration[perf_base]]

    # Evaluate TRUE performance of best model on test set for model selection later on
    if metric == "ndcg":
        perf_test = ndcg_at_k(model_best, train, test, K=K, show_progress=False)  
    elif metric == "precision":
        perf_test = precision_at_k(model_best, train, test, K=K, show_progress=False)  
    else:
        sys.exit('Performance metric type unspecified')

    return [perf_test, {"seed": seed, "model": model_best, "hyperparameters": hyperparams_optimal, "perf_test": perf_test}]

if __name__ == "__main__":
    NAME = "fm"
    DATA_SET = constant.DATA_SETS[NAME]
    SET_NAME = DATA_SET["label"].upper()
    MODEL_SETTINGS = constant.MODELS_CHOSEN[NAME]["ground_truth"]
    ALGORITHM = MODEL_SETTINGS["ALGORITHM"]
    METRIC = MODEL_SETTINGS["METRIC"]

    # Define paths
    VAR_PATH = helper.get_var_path(DATA_SET, constant.FOLDER_NAMES["gt_model"], MODEL_SETTINGS)
    LOG_PATH = helper.get_log_path(DATA_SET)
    
    # Global accesible variables
    my_globals = globals()
    
    # Read in data files as data frames
    for var_name, file_name in DATA_SET["data"].items():
        my_globals[var_name] = pd.read_csv(DATA_SET["data_src"] + file_name, sep="\t",  encoding="latin-1")

    # Get cumulative streams per artist
    artist_streams = helper.merge_duplicates(user_artists, "artistID", "weight")
    artist_streams_ranked = artist_streams.sort_values(by=["weight"], ascending=False)
    # Get the top 2500 streamed items (i.e cumulative user-item counts) 
    items = np.array(artist_streams_ranked["artistID"])[0:2500]

    # Filter users that interacted with the top 2500 items 
    # '100%' data set, i.e. contains all relevant users and items
    user_item = user_artists[user_artists["artistID"].isin(items)] 

    # Log transform of raw count input data to downscale large values
    def log_transform(r):
        return np.log(r) 

    # Pre-process the raw counts with log-transformation
    user_item["weight"] = user_item["weight"].map(log_transform)
    
    # Get users
    users = user_item["userID"].unique()
    # Get items
    items = user_item["artistID"].unique()
    shape = (len(users), len(items))

    # Create indices for users and items
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
    user_index = user_item["userID"].astype(user_cat).cat.codes
    item_index = user_item["artistID"].astype(item_cat).cat.codes

    # Conversion via COO matrix to the whole user-item observation/interaction matrix R
    R_coo = sparse.coo_matrix((user_item["weight"], (user_index, item_index)), shape=shape)
    R = R_coo.tocsr()

    # For computation efficiency only generate a model initially (i.e. when not yet exists)
    model_file = constant.FILE_NAMES["gt_model"]
    model_file_path = VAR_PATH + model_file

    if not path.exists(model_file_path):
        start = time.time()
        print("Start generating ground truth model {}...".format(SET_NAME))

        # Train, validate and test model of 3 different data splits
        num_random_seeds = 3

        # To save TRUE (i.e. test) performance of optimal model per seed (i.e. data set split)
        performance_per_seed = {}
        
        # Hyperparameters' (search) space
        latent_factors = [16, 32, 64, 128]
        regularization = [0.01, 0.1, 1.0, 10.0]
        confidence_weighting = [0.1, 1.0, 10.0, 100.0]
        
        # Get model's hyperparameters to be tuned 
        configurations = helper.generate_hyperparameter_configurations(regularization, latent_factors, confidence_weighting)

        # Cross-validation
        print("Training for", num_random_seeds, "models...MULTIPROCESSING")

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(train_model, [(R_coo, configurations, seed, METRIC, ALGORITHM) for seed in range(num_random_seeds)])
        pool.close()

        # Model selection by test set performance    
        ground_truth_model = results[helper.get_index_best_model(results)][1]

        # SAVE: Save model that can generate the ground truth
        io.save(model_file_path, (model_file, ground_truth_model))
        
        end = time.time() - start
        io.write_to_file(LOG_PATH, "Generating {} {}\n".format(model_file, str(end)))

    # Only generate ground truth when not yet generated
    ground_truth_file = constant.FILE_NAMES["gt"]
    ground_truth_file_path = VAR_PATH + ground_truth_file

    if not path.exists(ground_truth_file_path):   
        start = time.time()
        print("Loading ground truth model {}...".format(SET_NAME))

        ## LOAD: Load model settings that can generate the ground truth
        io.load(model_file_path, my_globals)
        model = ground_truth_model["model"]
        
        print("Generating 'ground truth' {}...".format(SET_NAME))
        ground_truth = model.user_factors @ model.item_factors.T

        # SAVE: Save ground truth preferences
        io.save(ground_truth_file_path, (ground_truth_file, ground_truth))   
        
        end = time.time() - start
        io.write_to_file(LOG_PATH, "Predicting {} {}\n".format(ground_truth_file, str(end)))
