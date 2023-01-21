import argparse
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import pickle
from scipy import sparse

from Recommender import Recommender, grid_search
from implicit.evaluation import train_test_split, precision_at_k

def read_lastfm():
    """
    Reads the user_artists table from the Lastfm-2k dataset. Filters entries to
    only include entries with the top 2500 most listened artists.
    Outputs:
        df - A dataframe with columns 'userID', 'artistID' and 'weight'
    """
    df = pd.read_table("../data/hetrec2011-lastfm-2k/user_artists.dat", sep="\t",
                       header=0, names=['userID', 'itemID', 'confidence'])
    top_artists = df['itemID'].value_counts()[:2500].index
    df = df[df['itemID'].isin(top_artists)]
    return df

def read_movielens():
    """
    Reads the ratings table from the MovieLens-1M dataset. Filters entries to
    only include entries with the top 2000 most rating users and the top 2500
    most rated movies.
    Outputs:
        df - A dataframe with columns 'userID', 'movieID' and 'rating'
    """
    df = pd.read_table("../data/ml-1m/ratings.dat", sep="::",
                       names=['userID', 'itemID', 'confidence'],
                       usecols=[0,1,2], engine='python')
    top_users = df['userID'].value_counts()[:2000].index
    top_movies = df['itemID'].value_counts()[:2500].index
    df = df[df['userID'].isin(top_users)]
    df = df[df['itemID'].isin(top_movies)]
    return df

def df_to_csr(df, row_name, column_name, entry_name, IDs_to_indices=False):
    """
    Converts dataframe to sparse matrix in CSR format.
    Inputs:
        df - Dataframe to be converted
        row_name - Name of dataframe column containing the row values
        column_name - Name of dataframe column containing the column values
        value_name - Name of dataframe column containing the entry values
        IDs_to_indices - If True, set IDs of rows and columns to corresponding
                         index in dataframe
    Outputs:
        csr - CSR matrix
    """
    users = df["userID"].unique()
    artists = df["itemID"].unique()
    shape = (len(users), len(artists))
    
    # Replace IDs for users and artists with indices
    row_cat = CategoricalDtype(categories=sorted(users), ordered=True)
    column_cat = CategoricalDtype(categories=sorted(artists), ordered=True)
    row_index = df[row_name].astype(row_cat).cat.codes
    column_index = df[column_name].astype(column_cat).cat.codes
    
    if IDs_to_indices:
        df[row_name] = row_index
        df[column_name] = column_index

    # Create sparse matrix
    coo = sparse.coo_matrix((df[entry_name], (row_index, column_index)),
                            shape=shape)
    csr = coo.tocsr()
    return csr

def get_ground_truths(df):
    """
    Generates ground truth preferences given a dataframe.
    Inputs:
        df - Dataframe with columns 'userID', 'itemID' and 'confidence'
    Outputs:
        ground_truths - Ground truth preferences
    """
    # Get sparse matrix
    user_item_matrix = df_to_csr(df, "userID", "itemID", "confidence")
    
    factors = [16, 32, 64, 128]
    reg = [0.01, 0.1, 1., 10.]
    conf_weights = [0.1, 1., 10., 100.]
    seeds = [0, 42, 123]
    best_model = None
    best_score = 0
    for seed in seeds:
        # Make random split
        train_matrix, test_matrix = train_test_split(user_item_matrix, 0.8,
                                                     seed)
        train_matrix, val_matrix = train_test_split(user_item_matrix, 0.875,
                                                    seed)
        # Test hyperparameters
        _, rec = grid_search(train_matrix, val_matrix, factors, reg,
                             conf_weights, seed)
        test_score = precision_at_k(rec.model, train_matrix, val_matrix,
                                     show_progress=False)
        if test_score > best_score:
            best_model = rec
    
    # Take best hyperparameters
    # max_indices = np.unravel_index(hyperparams.argmax(), hyperparams.shape)
    # factors = factors[max_indices[0]]
    # reg = reg[max_indices[1]]
    # conf_weights = conf_weights[max_indices[2]]
    
    # Create ground truth preferences
    # rec = Recommender(factors=factors, regularization=reg, alpha=conf_weights,
    #                   compute_dense_matrix=True)
    # rec.fit_model(user_item_matrix)
    ground_truths = best_model.preferences
    return ground_truths

def get_lastfm_ground_truths():
    """
    Generates ground truth preferences for the Lastfm-2k dataset.
    Outputs:
        ground truths - Ground truth preferences
    """
    # Read data
    df = read_lastfm()
    df['confidence'] = df['confidence'].apply(np.log) # log transformation
    
    # Get ground truths
    ground_truths = get_ground_truths(df)
    return ground_truths

def get_movielens_ground_truths():
    """
    Generates ground truth preferences for the MovieLens-1M dataset.
    Outputs:
        ground truths - Ground truth preferences
    """
    # Read data
    df = read_movielens()
    df.loc[df['confidence'] < 3, 'confidence'] = 0 # set ratings < 3 to 0
    
    # Get ground truths
    ground_truths = get_ground_truths(df)
    return ground_truths

if __name__ == '__main__':
    """
    Creates ground truths for the Lastfm-2k and MovieLens-1M datasets and saves
    them in the files lastfm_ground_truths and movielens_ground_truths.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='both', type=str,
                        help='Datasets to read. Options: \'lastfm\, \'movielens\' or \'both\' (default)')
    args = parser.parse_args()
    if args.dataset != 'movielens':
        lastfm_gt = get_lastfm_ground_truths()
        with open('../results/lastfm_ground_truths', 'wb') as f:
            pickle.dump(lastfm_gt, f)
    if args.dataset != 'lastfm':
        movielens_gt = get_movielens_ground_truths()
        with open('../results/movielens_ground_truths', 'wb') as f:
            pickle.dump(movielens_gt, f)
