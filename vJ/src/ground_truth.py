import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import sparse
from Recommender import Recommender
from implicit.evaluation import train_test_split, precision_at_k

def read_lastfm():
    """
    Reads the user_artists table from the lastfm-2k dataset. Filters artists to
    include only the top 2500 artists.
    Outputs:
        df - A dataframe with columns 'userID', 'artistID' and 'weight'
        """
    # Read data and select 2500 most popular artists
    df = pd.read_table("../data/hetrec2011-lastfm-2k/user_artists.dat",
                                 sep="\t",)
    top_artists = df['artistID'].value_counts()[:2500].index
    df = df[df['artistID'].isin(top_artists)]
    return df

def grid_search(train_matrix, val_matrix, factors, regularization, confidence_weights):
    """
    Performs grid search over the hyperparameters for a recommender using the
    precision_at_k metric.
    Inputs:
        train_matrix - Sparse user-item matrix of training data
        val_matrix - Sparse user-item matrix of validation data
        factors - List of number of latent factors to test
        regularization - List regularization values to test
        confidence_weights - List of confidence weight values to test
    Outputs:
        Results - Matrix where entry (i,j,k) is the found precision for number
                  of factors i, regularization j and confidence weight k
    """
    results = np.zeros((len(factors), len(regularization), len(confidence_weights)))
    for fi, ri, ci in np.ndindex(results.shape):
        rec = Recommender(factors=factors[fi], regularization=regularization[ri],
                          alpha=confidence_weights[ci])
        rec.fit_model(train_matrix)
        results[fi, ri, ci] = precision_at_k(rec.model, train_matrix,
                                                 val_matrix)
    return results

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
    artists = df["artistID"].unique()
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

def get_lastfm_ground_truths():
    """
    Generates ground truth preferences for the lastfm-2k dataset.
    Outputs:
        ground truths - Ground truth preferences
    """
    # Read data
    df = read_lastfm()
    df['weight'] = df['weight'].apply(np.log) # log transformation
    
    # Get sparse matrix
    user_artist_matrix = df_to_csr(df, "userID", "artistID", "weight")
    
    # Make random split
    train_matrix, test_matrix = train_test_split(user_artist_matrix, 0.8, 42)
    train_matrix, val_matrix = train_test_split(user_artist_matrix, 0.875, 42)
    
    # Find best hyperparameters
    factors = [16, 32, 64, 128]
    reg = [0.01, 0.1, 1., 10.]
    conf_weights = [0.1, 1., 10., 100.]
    hyperparams = grid_search(train_matrix, val_matrix, factors, reg, conf_weights)
    max_indices = np.unravel_index(hyperparams.argmax(), hyperparams.shape)
    factors = factors[max_indices[0]]
    reg = reg[max_indices[1]]
    conf_weights = conf_weights[max_indices[2]]
    
    # Create ground truth preferences
    rec = Recommender(factors=factors, regularization=reg, alpha=conf_weights,
                      compute_dense_matrix=True)
    rec.fit_model(user_artist_matrix)
    ground_truths = rec.user_item_logits
    return ground_truths