####
# vN/src/constant.py
#
# This file simulates the existence of 'constants' which can be used globally accross files
####
# Debug modus
DEBUG = True
DUMMY_DATA_SET_CHOICE = "all"
DUMMY_MODEL_CHOICE = {"fm": 
                        {"ground_truth": {"ALGORITHM": "als", "METRIC": "ndcg"}, 
                        "recommender": {"ALGORITHM": "als", "METRIC": "ndcg", "normalize": True}},
                "mv": 
                        {"ground_truth": {"ALGORITHM": "als", "METRIC": "ndcg"}, 
                        "recommender": {"ALGORITHM": "als", "METRIC": "ndcg", "normalize": True}}
                }

DUMMY_EXPERIMENT_CHOICE = "5.1"


MODELS_CHOSEN = None

# Data sets on which the experiments can be performed
DATA_SETS = {"movie": {"label": "mv",
                "filename": "user_movie.py",
                # "data": {"user_movies": "ratings.csv"},
                # "data_src": "../data/ml-25m/",
                "data": {"user_movies": "ratings.dat"},
                "data_src": "../data/ml-1m/",
                # Extension of variable name
                "var_ext": "_mv",
                "var_folder": "mv/",
                # Plotting lay-out
                "plot_style": {
                        "label": "MovieLens",
                        "linestyle": "dotted",
                        "color": "tab:orange"
                        },
                "log_file": "mv.log",
                "experiment": None
                },
        "fm": {"label": "fm",
                "filename": "user_artist.py",
                "data": {"user_artists": "user_artists.dat"},
                "data_src": "../data/hetrec2011-lastfm-2k/",
                "var_ext": "_fm",
                # Variable path infix depending on the data set
                "var_folder": "fm/", 
                # Plotting lay-out
                "plot_style": {
                        "label": "Last.fm",
                        "linestyle": "-.",
                        "color": "tab:blue"
                        },
                "log_file": "fm.log",
                "experiment": None
        },
}

FOLDER_NAMES = {"gt_model": "ground_truth/",
                "rec_model": "recommender/"}

FILE_NAMES = {"gt_model": "ground_truth_model",
                "gt": "ground_truth",
                "rec_model": "recommender_model"}

# Specifies which experiments can be ran and the location their results are stored
EXPERIMENT_RUN_OPTIONS = {"all": None,
                        "5.1": {},
                        "IN PROGRESS": {}}

# Algorithm options for the ground truth model
ALGORITHM_GROUND_TRUTH = ["als", "lmf"]

# Algorithm options for the recommender system
ALGORITHM_RECOMMENDER = ["svd", "als"]

# Possible performance metrics to construct the ground truth and recommender model
PERFORMANCE_METRICS = { "als": {"ndcg": {"K": 40},
                                "precision": {"K": 40}},
                        "lmf": {"ndcg": {"K": 40},
                                "precision": {"K": 40}},
                        "svd": {"ndcg": {"K": 40},
                                "dcg": {"K": 40}}
                        }                        

# Folder to save intermediate variables of the pipeline
VARIABLES_FOLDER = "variables/"

# Name of folder where the results are stored
RESULTS_FOLDER = "results/"

# Folder to save time performances (wall clock time)
TIMING_FOLDER = RESULTS_FOLDER + "execution_times/"

# Folder to save recommender system models
MODELS_FOLDER = "models/"


