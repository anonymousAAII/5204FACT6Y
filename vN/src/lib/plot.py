####
# vN/src/lib/plot.py
# 
# This file contains all the functionality to plot and save data of experiments
####
import matplotlib.pyplot as plt

# 1st party imports
import constant

def plot_experiment_5_1A(data, y_label, x_label, label_1, label_2, data_name, window):
    line_1 = data[0]
    line_2 = data[1]

    plt.figure(window)
    plt.plot(line_1.keys(), line_1.values(), label=label_1, linestyle='-.')
    plt.plot(line_2.keys(), line_2.values(), label=label_2, linestyle='dotted', color="orange")

    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel(x_label)    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + "experiment_5_1A_" + data_name + ".png")
    plt.show()

def plot_experiment_single(data, y_label, x_label, label_1, data_name, window=1):
    line_1 = data[0]

    plt.figure(window)
    plt.plot(line_1.keys(), line_1.values(), label=label_1, linestyle='-.')

    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel(x_label)    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + "experiment_5_1A_" + data_name + ".png")
    plt.show()

