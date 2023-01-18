import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from scipy import sparse
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k

def merge_duplicates(df, col_duplicate, col_merge_value, mode="sum"):
    """
    merge_duplicate checks for duplicate entries 
    and according to the given mode performs an operation on the specified column value.

    :df: dataframe used
    :col_duplicate: name of column to be checked for duplicates
    :col_merge_value: name of column which value to perform an operation on when it concerns a duplicate
    :mode: name which determines which operation is performed, default is 'sum'
    :return: dataframe which contains the unique entries and their values after the operation is performed
    """ 
    if mode == "sum":
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()
    # Default sum the <col_merge_value> values of the duplicates
    else:
        return df.groupby(col_duplicate, as_index = False)[col_merge_value].sum()

def generate_user_item_matrix(user_items_dict, users, items):
    # User-item observation matrix (Johnson 2014)    
    R = np.zeros((len(users), len(items)))

    for row, user in enumerate(users):
        for col, item in enumerate(items):
            # When no existing user-item interaction
            if not (item in user_items_dict[user]):
                continue
            
            # print(user_items_dict[user][item])
            R[row][col] = user_items_dict[user][item]

    return R

if __name__ == "__main__":
    TOP_2500_STREAMED = "items" # Or user-items

    # Source folder of datasets
    DATA_SRC = "../data/hetrec2011-lastfm-2k/"
    # Mapping from data <variable name> to its <filename>
    data_map = {"artists": "artists.dat", 
                "tags": "tags.dat", 
                "user_artists": "user_artists.dat"}

    # Variable names of datasets to be used
    var_names = list(data_map.keys())
    
    # Global accesible variables
    myVars = vars()
    
    # Read in data files
    for var_name, file_name in data_map.items():
        # Read data in Pandas Dataframe (mainly for manual exploration and visualization)    
        myVars[var_name] = pd.read_csv(DATA_SRC + file_name, sep="\t",  encoding="latin-1")

    # Get the top 2500 streamed items (i.e cumulative user-item counts) 
    if TOP_2500_STREAMED == "items":
        # Get cumulative streams per artist
        artist_streams = merge_duplicates(user_artists, "artistID", "weight")
        rank_artist_streams = artist_streams.sort_values(by=["weight"], ascending=False)
        # Get top 2500 items most listend to
        items = np.array(rank_artist_streams["artistID"])[0:2500]

    # Filter users that interacted with the top 2500 items 
    user_item_100 = user_artists[user_artists["artistID"].isin(items)] # '100%' dataset

    # Log transform of raw count input data (Johnson 2014)
    def log_transform(r, alpha = 1):
        return alpha * np.log(r) # ln or log10?

    # Pre-process the raw counts with log-transformation
    user_item_100["weight"] = user_item_100["weight"].map(log_transform)

    # Get users
    users = np.array(user_item_100["userID"].unique())
    # Get items
    items = np.array(user_item_100["artistID"].unique())

    # Create dict struct for fast lookup O(1) of values
    artists_dict = {} # {<id>: {"name": <name>}}
    tags_dict = {} # {<tagID>: <tagValue>}
    user_artists_dict = {} # {<userID>: {<artistID>: <weight>}}

    for user in users:
        subset = user_item_100[user_item_100["userID"] == user]
        user_artists_dict[user] = dict(zip(np.array(subset["artistID"]), np.array(subset["weight"])))

    def generate_hyperparameter_configuration():
        # Hyper parameters
        latent_factors = [16, 32, 64, 128]
        regularization = [0.01, 0.1, 1.0, 10.0]
        weighting_parameter = [0.1, 1.0, 10.0, 100.0]

        configurations = {}
        
        # Initialize with all possible hyperparameter combinations
        i = 0
        for latent_factor in latent_factors:
            for reg in regularization:
                for alpha in weighting_parameter:
                    configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                    i+=1
                    
        return configurations

    # Create user-item observation matrix R (Johnson 2014)


    # Create 70%/10%/20% train/validation/test data split of the user-item listening counts three times using three different seeds
    num_random_seeds = 3 # TO DO SET TO 3

    # To safe TRUE (i.e. test) performance of model per seed (i.e. data set split)
    performance_per_seed = {}
    
    # Model's hyper parameters to be tuned using grid search
    configurations = generate_hyperparameter_configuration()

    for seed in range(num_random_seeds):
  
        # When random_state set to an None, train_test_split will return different results for each
        # Equivalent to setting a different random seed each time  
        train, validation_test = train_test_split(user_item_100, test_size=0.3)
        validation, test = train_test_split(validation_test, test_size=2/3)
        
        # Train data
        R_train = generate_user_item_matrix(user_artists_dict, np.array(train["userID"].unique()), np.array(train["artistID"].unique()))
        R_train_csr = sparse.csr_matrix(R_train) 
        
        # Validation data
        R_validation = generate_user_item_matrix(user_artists_dict, np.array(validation["userID"].unique()), np.array(validation["artistID"].unique()))
        R_validation_csr = sparse.csr_matrix(R_validation) 

        # Test data
        R_test = generate_user_item_matrix(user_artists_dict, np.array(test["userID"].unique()), np.array(test["artistID"].unique()))
        R_test_csr = sparse.csr_matrix(R_test) 

        # To safe performance per hyperparameter combination as {<precision>: <hyperparameter_id>}
        performance_per_configuration = {}

        # Hyperparameter tuning through grid search
        for i, hyperparameters in configurations.items():
            # Initialize model
            model = AlternatingLeastSquares(factors=hyperparameters["latent_factor"], 
                                            regularization=hyperparameters["reg"],
                                            alpha=hyperparameters["alpha"])
            
            # Train standard matrix factorization algorithm (Hu, Koren, and Volinsky (2008)) on the train set
            model.fit(R_train_csr)        

            # Benchmark model performance using validation set
            p = precision_at_k(model, R_train_csr, R_validation_csr, K=1000, num_threads=4)  
            print("Seed:", seed, ", iteration:", i)
            print("Hyperparameters:", hyperparameters)
            print("Precision=", p, "\n")

            performance_per_configuration[p] = i

        p_val_max = np.amax(np.array(list(performance_per_configuration.keys())))
        print("Best validation performance =", p_val_max)
        optimal_param_config = configurations[performance_per_configuration[p_val_max]]
        print("Optimal hyperparameters for model of current seed =", optimal_param_config)

        # Evaluate TRUE performance of best model on test set for model selection later on
        model = AlternatingLeastSquares(factors=optimal_param_config["latent_factor"], 
                                        regularization=optimal_param_config["reg"],
                                        alpha=optimal_param_config["alpha"])
        
        model.fit(R_train_csr)
        p_test = precision_at_k(model, R_train_csr, R_test_csr, K=1000, num_threads=4)  

        performance_per_seed[seed] = {"model": model, "p_test": p_test, "hyperparameters": optimal_param_config}

    print("Test performance per seed with corresponding hyperparameter configuration:")
    print(performance_per_seed)

    # # Use test set for model selection (which configuration generalizes the best)
    # for seed, eval_data in performance_per_seed.items():
    #     param = eval_data["hyperparameters"] 

    exit()   
    # Mapping from <user_id> to user index in sparse-matrix (R) (Johnson 2014) and vice versa
    users_index = dict(zip(users, np.arange(len(users))))
    index_users = dict(zip(np.arange(len(users)), users))
    # Mapping from <item_id> to item index in sparse-matrix (R) (Johnson 2014) and vice versa
    itemss_index = dict(zip(items, np.arange(len(items))))
    index_items = dict(zip(np.arange(len(items)), items))

