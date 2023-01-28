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
from sklearn.model_selection import train_test_split
import scipy
from scipy import sparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from os import path
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.lmf import LogisticMatrixFactorization
from implicit.evaluation import ndcg_at_k
from pandas.api.types import CategoricalDtype
import multiprocessing as mp
from tqdm import tqdm
import time

# 1st party imports
from lib import helper
from lib import io
import constant

def train_model(R_coo, configurations, seed, built_in_LMF):
    """
    For a random seed trains a model on a sparse matrix for all given hyperparameter combinations.

    :R_coo:             sparse matrix
    :configurations:    hyperparameters' space
    :seed:              seed (counter)
    """
    # Create 70%/10%/20% train/validation/test data split of the user-item top 2500 listening counts
    train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=0.7)
    validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=1/3)

    # To safe performance per hyperparameter combination as {<precision>: <hyperparameter_id>}
    performance_per_configuration = {}

    ndcg_base = 0
    model_best = None

    print("Start training for {}th seed".format(seed + 1))
    
    # Hyperparameter tuning through grid search
    for i in tqdm(range(len(configurations))):
        hyperparameters = configurations[i]

        # Initialize model
        if built_in_LMF:
            # Notice you can't specify the confidence weighing parameter alpha here so it is applied below
            model = LogisticMatrixFactorization(factors=hyperparameters["latent_factor"], 
                                                regularization=hyperparameters["reg"])
        else:
            model = AlternatingLeastSquares(factors=hyperparameters["latent_factor"], 
                                            regularization=hyperparameters["reg"],
                                            alpha=hyperparameters["alpha"])

        # Train matrix factorization algorithm on the train set
        model.fit(train.multiply(hyperparameters["alpha"] if built_in_LMF else 1), show_progress=False)        

        # Benchmark model performance using validation set
        ndcg = ndcg_at_k(model, train.multiply(hyperparameters["alpha"] if built_in_LMF else 1), validation.multiply(hyperparameters["alpha"] if built_in_LMF else 1), K=constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"], show_progress=False)
        
        print("Seed: {} NDCG@{}".format((seed + 1), constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"]), ndcg)

        # When current model outperforms previous one update tracking states
        if ndcg > ndcg_base:
            ndcg_base = ndcg
            model_best = model

        performance_per_configuration[ndcg] = i

    print("Training ended for {}th seed".format(seed + 1))
    
    hyperparams_optimal = configurations[performance_per_configuration[ndcg_base]]

    # Evaluate TRUE performance of best model on test set for model selection later on
    ndcg_test = ndcg_at_k(model_best, train.multiply(hyperparams_optimal["alpha"] if built_in_LMF else 1), test.multiply(hyperparams_optimal["alpha"] if built_in_LMF else 1), K=constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"], show_progress=False)  

    return [ndcg_test, {"seed": seed, "model": model_best, "hyperparameters": hyperparams_optimal, "ndcg_test": ndcg_test}]

if __name__ == "__main__":
    io.initialize_empty_file(constant.TIMING_FOLDER + constant.TIMING_FILE["fm"])

    # Source folder of datasets
    DATA_SRC = "../data/hetrec2011-lastfm-2k/"
    # Mapping from data <variable name> to its <filename>
    data_map = {"user_artists": "user_artists.dat"}

    # Variable names of datasets to be used
    var_names = list(data_map.keys())

    # Infix of folder to perform I/O-operations on for specifically this data set (reading, writing etc.)
    IO_INFIX = constant.VAR_SUB_FOLDER["fm"]
    # Full path where variables of I/O operations are stored
    IO_PATH = constant.VARIABLES_FOLDER + IO_INFIX
 
    # Variable name extension
    VAR_EXT = constant.VAR_EXT["fm"]
    
    # Global accesible variables
    my_globals = globals()
    
    # Read in data files
    for var_name, file_name in data_map.items():
        # Read data in Pandas Dataframe (mainly for manual exploration and visualization)    
        my_globals[var_name] = pd.read_csv(DATA_SRC + file_name, sep="\t",  encoding="latin-1")

    # Get cumulative streams per artist
    artist_streams = helper.merge_duplicates(user_artists, "artistID", "weight")
    artist_streams_ranked = artist_streams.sort_values(by=["weight"], ascending=False)
    # Get the top 2500 streamed items (i.e cumulative user-item counts) 
    items = np.array(artist_streams_ranked["artistID"])[0:2500]

    # Filter users that interacted with the top 2500 items 
    # '100%' data set, i.e. contains all relevant users and items
    user_item = user_artists[user_artists["artistID"].isin(items)] 

    # Log transform of raw count input data (Johnson 2014)
    def log_transform(r):
        return np.log(r) 

    built_in_LMF = False

    # Approximate LMF by applying a log-transform beforehand   
    if built_in_LMF == False:
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
    model_file = "ground_truth_model"
    model_file_path = IO_PATH + model_file
    model_var_name = model_file + VAR_EXT

    if not path.exists(model_file_path):
        start = time.time()
        print("Start generating ground_truth model FM...")

        # Train, validate and test model of 3 different data splits
        num_random_seeds = 3

        # To save TRUE (i.e. test) performance of optimal model per seed (i.e. data set split)
        performance_per_seed = {}
        
        # Hyperparameters' (search) space
        latent_factors = [16, 32, 64, 128]
        regularization = [0.01, 0.1, 1.0, 10.0]
        confidence_weighting = [0.1, 1.0, 10.0, 100.0]
        
        # Get model's hyperparameters to be tuned 
        configurations = helper.generate_hyperparameter_configurations(regularization, confidence_weighting, latent_factors)

        # Cross-validation
        print("Training for", num_random_seeds, "models...MULTIPROCESSING")

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(train_model, [(R_coo, configurations, seed, True) for seed in range(num_random_seeds)])
        pool.close()

        # Model selection by test set performance    
        ground_truth_model = results[helper.get_index_best_model(results)][1]

        # SAVE: Save model that can generate the ground truth
        io.save(IO_INFIX + model_file, (model_var_name, ground_truth_model))
        
        end = time.time() - start
        io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE["fm"], "Generating " + model_file + "  " + str(end) + "\n")

    # Only generate ground truth when not yet generated
    ground_truth_file = "ground_truth"
    ground_truth_file_path = IO_PATH + ground_truth_file
    ground_truth_var_name = ground_truth_file + VAR_EXT

    if not path.exists(ground_truth_file_path):   
        start = time.time()
        print("Loading ground truth model FM...")

        ## LOAD: Load model settings that can generate the ground truth
        io.load(IO_INFIX + model_file, my_globals)
        model = ground_truth_model_fm["model"]
        
        print("Generating 'ground truth' FM...")
        ground_truth_fm = model.user_factors @ model.item_factors.T

        # SAVE: Save ground truth preferences
        io.save(IO_INFIX + ground_truth_file, (ground_truth_var_name, ground_truth_fm))   
        
        end = time.time() - start
        io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE["fm"], "Predicting " + ground_truth_file + "  " + str(end) + "\n")

