####
# vN/src/lib/io.py
# 
# This file contains functions for Input/Output operations. 
####
import pickle
from os import path

# 1st party imports
import constant
from lib import helper

# Source: https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
def save(filename, *args):
	"""
	Saves variables in a file such that afterwards they can be (re)loaded and utilized in other scripts

	:filename:		filename to store the variables
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

	:filename:		filename to load variables from
	:globals:		globals() 
	"""
	with open(constant.VARIABLES_FOLDER + filename, 'rb') as f:
		for name, v in pickle.load(f).items():
			# Set each global variable to the value from the file
			globals[name] = v

def initialize_empty_file(file_path):
	"""
	Initilizes an empty file given by its file path with the current date time

	:file_path:		path of file to initialize	 
	""" 
	with open(file_path, "a" if path.exists(file_path) else "w") as file: 
		file.write(helper.get_current_datetime() + "\n")
	print("File {} initialized successfully!".format(file_path))

def write_to_file(file_path, string):
	"""
	Writes the given string to a file given by its file path

	:file_path:		path of file to write to
	:string:		string to write
	"""
	with open(file_path, "a") as file:
		file.write(string) 


