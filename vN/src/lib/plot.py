####
# vN/src/lib/plot.py
# 
# This file contains all the functionality to plot and save data of experiments
####
import matplotlib.pyplot as plt

# 1st party imports
import constant

def plot_experiment_line(data, y_label, x_label, labels, linestyles, colors, file_name, x_upper_bound = None, sci_x=False, window=1):
    """
    Makes a line plot given a list of data containing the x, y data of each line in a dictionary in the format {x_1: y_1, ..., x_N: y_N}

    :data:              list containing the x and y data of each line
    :y_label:           the y label of the figure
    :x_label:           the x label of the figure
    :labels:            the label of each data line
    :linestyles:        the linestyle of each data line
    :colors:            the color of each data line
    :file_name:         the file name to save the figure too
    :x_upper_bound:     when not None plot the lines until this given value
    :sci_x:             when not False adds scientific notation the the x-axis tickmarks (e.g. a x 10^b)
    :window:            window identifier
    """
    plt.figure(window)

    # Add scientific notation to the x-axis tickmarks
    if sci_x:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    for i, line in enumerate(data):
        plt.plot(line.keys(), line.values(), label=labels[i], linestyle=linestyles[i], color=colors[i])

    plt.legend()

    plt.ylabel(y_label)
    plt.xlabel(x_label)    
    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + file_name + ".png")
    plt.show()