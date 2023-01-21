####
# vN/src/lib/io.py
# 
# This file contains functions for Input/Output operations. 
####
import pickle

# 1st party imports
import constant

# Source: https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
def save(filename, *args):
    """
    Saves variables in a file such that afterwards they can be (re)loaded and utilized in other scripts

    :filename:      filename to store the variables
    :*args:         variables to store must be in format (<variable name>, <variable)
    """
    d = {}

    for item in args:
        # Copy over desired values
        d[item[0]] = item[1]
    with open(constant.VARIABLES_FOLDER + filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

# Source: https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
def load(filename, globals):
    """
    Loads variables from a file into globals() such that they can be utilized

    :filename:  filename to load variables from
    :globals:   globals() 
    """
    with open(constant.VARIABLES_FOLDER + filename, 'rb') as f:
        for name, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            globals[name] = v
