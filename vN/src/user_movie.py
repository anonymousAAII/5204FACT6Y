####
# vN/src/user_movie.py
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
from datetime import datetime
from sklearn.model_selection import train_test_split
import scipy
from scipy import sparse
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, AUC_at_k 
from os import path
from tqdm import tqdm
from pandas.api.types import CategoricalDtype

# 1st party imports
from lib import helper
from lib import io
import constant

if __name__ == "__main__":
    # Source folder of datasets
    DATA_SRC = "../data/ml-1m"
    # Mapping from data <variable name> to its <filename>
    data_map = {"movies": "movies.dat", 
                "user_movies": "ratings.dat"}

    # Variable names of datasets to be used
    var_names = list(data_map.keys())
    
    # Global accesible variables
    my_globals = globals()

    exit()
    
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
    model_file = "ground_truth_model_fm"

    if not path.exists(constant.VARIABLES_FOLDER + model_file):
        print("Start generating ground_truth_fm model...")
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
        for seed in tqdm(range(num_random_seeds)):

            # Create 70%/10%/20% train/validation/test data split of the user-item top 2500 listening counts
            train, validation_test = implicit.evaluation.train_test_split(R_coo, train_percentage=0.7)
            validation, test = implicit.evaluation.train_test_split(scipy.sparse.coo_matrix(validation_test), train_percentage=1/3)

            # To safe performance per hyperparameter combination as {<precision>: <hyperparameter_id>}
            performance_per_configuration = {}

            p_base = 0
            model_best = None

            # Hyperparameter tuning through grid search
            for i, hyperparameters in configurations.items():
                # Initialize model
                model = AlternatingLeastSquares(factors=hyperparameters["latent_factor"], 
                                                regularization=hyperparameters["reg"],
                                                alpha=hyperparameters["alpha"])
                
                # Train standard matrix factorization algorithm (Hu, Koren, and Volinsky (2008)) on the train set
                model.fit(train, show_progress=False)        

                # Benchmark model performance using validation set
                p = AUC_at_k(model, train, validation, K=1000, show_progress=False, num_threads=4)

                # When current model outperforms previous one update tracking states
                if p > p_base:
                    p_base = p
                    model_best = model

                # print("Seed:", seed, ", iteration:", i)
                # print("Hyperparameters:", hyperparameters)
                # print("Precision=", p, "\n")

                performance_per_configuration[p] = i

            p_val_max = np.amax(np.array(list(performance_per_configuration.keys())))
            print("Best validation performance =", p_val_max)
            hyperparams_optimal = configurations[performance_per_configuration[p_val_max]]
            print("Optimal hyperparameters for model of current seed", seed," =", hyperparams_optimal)

            # Evaluate TRUE performance of best model on test set for model selection later on
            p_test = AUC_at_k(model_best, train, test, K=1000, show_progress=False, num_threads=4)  
            print("Performance on test set =", p_test, "\n")
            performance_per_seed[p_test] = {"seed": seed, "model": model_best, "hyperparameters": hyperparams_optimal, "precision_test": p_test}

        print("Test performance per seed with corresponding hyperparameter configuration:")
        print(performance_per_seed)

        # Model selection by test set performance    
        key_best_end_model = np.amax(np.array(list(performance_per_seed.keys())))
        global ground_truth_model_fm 
        ground_truth_model_fm = performance_per_seed[key_best_end_model]
        # Save model that can generate the ground truth
        io.save(model_file, (model_file, ground_truth_model_fm))

    # Only find ground truth when not yet generated
    ground_truth_file = "ground_truth_fm"

    if not path.exists(constant.VARIABLES_FOLDER + ground_truth_file):   
        print("Loading ground_truth_fm model...")

        # Load model settings that can generate the ground truth
        io.load(model_file, my_globals)
        model = ground_truth_model_fm["model"]
        
        print("Generating 'ground truth' fm...")
        ground_truth_fm = model.user_factors @ model.item_factors.T

        # Save ground truth preferences
        io.save(ground_truth_file, (ground_truth_file, ground_truth_fm))   

