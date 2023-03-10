####
# vN/src/lib/io.py
# 
# This file contains functions for Input/Output operations. 
####
import pickle
from os import path
import json

# 1st party imports
import constant
from lib import helper

# Source: https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
def save(filename, *args):
	"""
	Saves variables in a file such that afterwards they can be (re)loaded and utilized in other scripts

	Input:
		filename		- filename to store the variables
		*args:			- variables to store must be in format (<variable name>, <variable>)
	"""
	d = {}

	for item in args:
		# Copy over desired values
		d[item[0]] = item[1]

	with open(filename, 'wb') as f:
		# Put them in the file 			
		pickle.dump(d, f)

# Source: https://stackoverflow.com/questions/66271284/saving-and-reloading-variables-in-python-preserving-names
def load(filename, globals):
	"""
	Loads variables from a file into globals() such that they can be utilized

	Inputs:
		filename:		- filename to load variables from
		globals			- globals() object containing the global variables
	"""
	with open(filename, 'rb') as f:
		for name, v in pickle.load(f).items():
			# Set each global variable to the value from the file
			globals[name] = v

def initialize_empty_file(file_path):
	"""
	Initilizes an empty file given by its file path with the current date time

	Inputs:
		file_path		- path of file to initialize	 
	""" 
	with open(file_path, "a" if path.exists(file_path) else "w") as file: 
		file.write(helper.get_current_datetime() + "\n")
	print("File {} initialized successfully!".format(file_path))

def write_to_file(file_path, string, mode="txt"):
	"""
	Writes the given string to a file given by its file path

	Input:
		file_path		- path of file to write to
		string			- string to write
		mode			- what input type e.g. json needs to be written
	"""
	if mode == "json":
		with open(file_path, "a") as file:
			json.dump(string, file)
			file.write("\n") 
	else:
		with open(file_path, "a") as file:
			file.write(string + "\n") 

	


