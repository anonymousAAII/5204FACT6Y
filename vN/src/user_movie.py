####
# vN/src/user_movie.py
#
# Processes MovieLens data set to generate the ground truth which forms the base of further code
#
# Data set thanks to:
#
# F. Maxwell Harper and Joseph A. Konstan. 2015. 
# The MovieLens Datasets: History and Context. 
# ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. 
# DOI=http://dx.doi.org/10.1145/2827872
###
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
from scipy import sparse
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from os import path
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ndcg_at_k
from pandas.api.types import CategoricalDtype
import multiprocessing as mp
import time

# 1st party imports
from lib import helper
from lib import io
import constant

def train_model(R_coo, configurations, seed):
    """
    For a random seed trains a model on a sparse matrix for all given hyperparameter combinations.

    :R_coo:             sparse matrix
    :configurations:    hyperparameters' space
    :seed:              seed (counter)
    """
    # Create 70%/10%/20% train/validation/test data split of the user-item top 2500 listening counts
    train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=0.7)
    validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=1/3, random_state=seed)

    # To safe performance per hyperparameter combination as {<precision>: <hyperparameter_id>}
    performance_per_configuration = {}

    ndcg_base = 0
    model_best = None

    print("Start training for {}th seed".format(seed + 1))
    
    # Hyperparameter tuning through grid search
    for i, hyperparameters in configurations.items():
        # Initialize model
        model = AlternatingLeastSquares(factors=hyperparameters["latent_factor"], 
                                        regularization=hyperparameters["reg"],
                                        alpha=hyperparameters["alpha"])
        
        # Train standard matrix factorization algorithm (Hu, Koren, and Volinsky (2008)) on the train set
        model.fit(train, show_progress=False)        

        # Benchmark model performance using validation set
        ndcg = ndcg_at_k(model, train, validation, K=constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"], show_progress=False)
        
        print("Seed {}: NDCG@{}".format((seed + 1), constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"]), ndcg)

        # When current model outperforms previous one update tracking states
        if ndcg > ndcg_base:
            ndcg_base = ndcg
            model_best = model

        performance_per_configuration[ndcg] = i

    print("Training ended for {}th seed".format(seed + 1))
    
    hyperparams_optimal = configurations[performance_per_configuration[ndcg_base]]

    # Evaluate TRUE performance of best model on test set for model selection later on
    ndcg_test = ndcg_at_k(model_best, train, test, K=constant.PERFORMANCE_METRIC_VARS["NDCG"]["K"], show_progress=False)  

    return [ndcg_test, {"seed": seed, "model": model_best, "hyperparameters": hyperparams_optimal, "ndcg_test": ndcg_test}]

if __name__ == "__main__":
    io.initialize_empty_file(constant.TIMING_FOLDER + constant.TIMING_FILE["movie"])

    """
    <DATA_SRC> and <data_map> that are commented out are used when one wants to run it for the 25M MovieLens data set
    also containing half star ratings 
    """
    # # Used for 25M MovieLens data set
    # DATA_SRC = "../data/ml-25m/"
    # data_map = {"user_movies": "ratings.csv"}

    # Used for 1M MovieLens data set
    # Source folder of datasets
    DATA_SRC = "../data/ml-1m/"
    # Mapping from data <variable name> to its <filename>    
    data_map = {"user_movies": "ratings.dat"}

    header = {"user_movies": ["userID", "movieID", "rating", "timestamp"]}

    # Variable names of datasets to be used
    var_names = list(data_map.keys())

    # Infix of folder to perform I/O-operations on for specifically this data set (reading, writing etc.)
    IO_INFIX = constant.VAR_SUB_FOLDER["movie"]
    # Full path where variables of I/O operations are stored
    IO_PATH = constant.VARIABLES_FOLDER + IO_INFIX
 
    # Variable name extension
    VAR_EXT = constant.VAR_EXT["movie"]
    
    # Global accesible variables
    my_globals = globals()
    
    # Read in data files
    for var_name, file_name in data_map.items():
        print("Reading in", file_name,"...")
        # # Used for 25M MovieLens data set
        # my_globals[var_name] = pd.read_csv(DATA_SRC + file_name)
        # my_globals[var_name].columns = header[var_name]
        
        # Used for 1M MovieLens data set
        my_globals[var_name] = pd.read_csv(DATA_SRC + file_name, sep="::", header=None, names=header[var_name], encoding="latin-1", engine="python")
    
    # Get the number of ratings per movie (movie rating count)
    user_movies["rating_count"] = np.full(len(user_movies), 1)    
    movies_rating_count = helper.merge_duplicates(user_movies, "movieID", "rating_count")
    movies_rating_count_ranked = movies_rating_count.sort_values(by=["rating_count"], ascending=False)

    # print(movies_rating_count_ranked)

    # Get the top 2500 number of times rated items (i.e number of cumulative rating count) 
    items = np.array(movies_rating_count_ranked["movieID"])[0:2500]
    
    # Filter users that interacted with the top 2500 items 
    # '100%' data set, i.e. contains all relevant users and items
    user_item = user_movies[user_movies["movieID"].isin(items)] 
    
    # Get users that gave the most rating
    users_rating_count = helper.merge_duplicates(user_movies, "userID", "rating_count")
    users_rating_count_ranked = users_rating_count.sort_values(by=["rating_count"], ascending=False)

    # Keep top 2000 users
    users = np.array(users_rating_count_ranked["userID"])[0:2000]
    user_item = user_item[user_item["userID"].isin(users)]

    # Since setting ratings < 3 are usually considered as negative (Wang et al. 2018), we set ratings < 3 to zero
    def transform(r):
        if r < 3:
            r = 0
        return r

    # Pre-process the ratings
    user_item["rating"] = user_item["rating"].map(transform)
    
    # Get users
    users = user_item["userID"].unique()
    # Get items
    items = user_item["movieID"].unique()

    shape = (len(users), len(items))

    # Create indices for users and items that have the most rating
    user_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    item_cat = CategoricalDtype(categories=sorted(items), ordered=True)
    user_index = user_item["userID"].astype(user_cat).cat.codes
    item_index = user_item["movieID"].astype(item_cat).cat.codes

    # print(user_index)
    # exit()

    # Conversion via COO matrix to the whole user-item observation/interaction matrix R
    R_coo = sparse.coo_matrix((user_item["rating"], (user_index, item_index)), shape=shape)
    R = R_coo.tocsr()

    # For computation efficiency only generate a model initially (i.e. when not yet exists)
    model_file = "ground_truth_model"
    model_file_path = IO_PATH + model_file
    model_var_name = model_file + VAR_EXT

    if not path.exists(model_file_path):
        start = time.time()
        print("Start generating ground_truth model MOVIE...")

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
        results = pool.starmap(train_model, [(R_coo, configurations, seed) for seed in range(num_random_seeds)])
        pool.close()

        # Model selection by test set performance    
        ground_truth_model = results[helper.get_index_best_model(results)][1]

        # SAVE: Save model that can generate the ground truth
        io.save(IO_INFIX + model_file, (model_var_name, ground_truth_model))

        end = time.time() - start
        io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE["movie"], "Generating " + model_file + "  " + str(end) + "\n")

    # Only generate ground truth when not yet generated
    ground_truth_file = "ground_truth"
    ground_truth_file_path = IO_PATH + ground_truth_file
    ground_truth_var_name = ground_truth_file + VAR_EXT

    if not path.exists(ground_truth_file_path):   
        start = time.time()
        print("Loading ground truth model MOVIE...")

        ## LOAD: Load model settings that can generate the ground truth
        io.load(IO_INFIX + model_file, my_globals)
        model = ground_truth_model_mv["model"]
        
        print("Generating 'ground truth' MOVIE...")
        ground_truth_mv = model.user_factors @ model.item_factors.T

        # SAVE: Save ground truth preferences
        io.save(IO_INFIX + ground_truth_file, (ground_truth_var_name, ground_truth_mv))   
        
        end = time.time() - start
        io.write_to_file(constant.TIMING_FOLDER + constant.TIMING_FILE["movie"], "Predicting " + ground_truth_file + "  " + str(end) + "\n")
