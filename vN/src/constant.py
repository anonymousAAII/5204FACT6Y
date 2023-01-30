####
# vN/src/constant.py
#
# This file simulates the existence of 'constants' which can be used globally accross files
####
EXPERIMENT_RUN_OPTIONS = {"all": {"id": "all"},
                        "5.1": {"id": "5.1",
                                "experiment_dir": "5.1/"}}

# Algorithm of recommender system's preference estimates
ALGORITHM = {"SVD": "svd", "ALS": "als"}

# Parameter values for the performance metrics
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
                        
PERFORMANCE_METRIC = None
PERFORMANCE_METRIC_SVD = None

# Folder to save time performances
TIMING_FOLDER = "results/execution times/"
TIMING_FILE = {"fm": "fm.log", "mv": "mv.log"}

VARIABLES_FOLDER = "variables/"
# Variable path infix depending on the data set
VAR_SUB_FOLDER = {"fm": "fm/", "movie": "mv/"}
# Name of experiments folder
EXPERIMENTS_FOLDER = "experiments/"
# Name of folder where the results are stored
RESULTS_FOLDER = "results/"
# Extension of variable name depending on the data set name
VAR_EXT = {"fm": "_fm", "movie": "_mv"}
# Label of data depending on the data set name
DATA_LABELS = {"fm": "Last.fm", "mv": "MovieLens"}
DATA_LINESTYLES = {"fm": "-.", "mv": "dotted"}
DATA_COLORS = {"fm": "tab:blue", "mv": "orange"}