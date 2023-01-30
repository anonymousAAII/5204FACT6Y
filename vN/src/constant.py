####
# vN/src/constant.py
#
# This file simulates the existence of 'constants' which can be used globally accross files
####
# Specifies which experiments can be ran and the location their results are stored
EXPERIMENT_RUN_OPTIONS = {"all": {"id": "all"},
                        "5.1": {"id": "5.1",
                                "experiment_dir": "5.1/"}}

# Algorithm of recommender system
ALGORITHM = {"SVD": "svd", "ALS": "als"}

# Possible performance metrics to construct the ground truth and recommender model
PERFORMANCE_METRIC_VARS = {"ground truth": {
                                        "ndcg": {"K": 40},
                                        "precision": {"K": 40},
                                        "auc": {"K": 40}
                                        },
                        "recommender system": {
                                                "svd": {"ndcg": {"K": 40},
                                                        "dcg": {"K": 40}},
                                                "als": {"ndcg": {"K": 40},
                                                        "precision": {"K": 40},
                                                        "auc": {"K": 40}}
                                                }
                        }
                        
# Default performance metric for the ground truth model
PERFORMANCE_METRIC = None
# Performance metric for the recommender system
PERFORMANCE_METRIC_REC = None

# Folder to save time performances (wall clock time)
TIMING_FOLDER = "results/execution times/"
TIMING_FILE = {"fm": "fm.log", "mv": "mv.log"}

# Folder to save intermediate variables of the pipeline
VARIABLES_FOLDER = "variables/"
# Variable path infix depending on the data set
VAR_SUB_FOLDER = {"fm": "fm/", "movie": "mv/"}
# Name of experiments folder
EXPERIMENTS_FOLDER = "experiments/"
# Name of folder where the results are stored
RESULTS_FOLDER = "results/"
# Extension of variable name depending on the data set name
VAR_EXT = {"fm": "_fm", "movie": "_mv"}

# Plotting lay-out
DATA_LABELS = {"fm": "Last.fm", "mv": "MovieLens"}
DATA_LINESTYLES = {"fm": "-.", "mv": "dotted"}
DATA_COLORS = {"fm": "tab:blue", "mv": "orange"}