####
# vN/src/constant.py
#
# This file simulates the existence of 'constants' which can be used globally accross files
####
EXPERIMENT_RUN_OPTIONS = {"all": {"id": "all"},
                        "5.1": {"id": "5.1",
                                "experiment_dir": "5.1/"}}

# Algorithm of recommender system's preference estimates
ALGORITHM = ["SVD", "ALS"]

# Folder to save time performances
TIMING_FOLDER = "results/execution times/"
TIMING_FILE = {"fm": "fm.log", "movie": "mv.log"}

VARIABLES_FOLDER = "variables/"
# Variable path infix depending on the data set
VAR_SUB_FOLDER = {"fm": "fm/", "movie": "mv/"}
# Name of experiments folder
EXPERIMENTS_FOLDER = "experiments/"
# Name of folder where the results are stored
RESULTS_FOLDER = "results/"
# Extension of variable name depending on the data set
VAR_EXT = {"fm": "_fm", "movie": "_mv"}

