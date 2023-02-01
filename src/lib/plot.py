####
# vN/src/lib/plot.py
# 
# This file contains all the functionality to plot and save data of experiments
####
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

# 1st party imports
import constant

def gerenate_plot_data(data_sets_chosen, data_sets, experiment_key, result_key, file_name):
    """
    Generate plot data according to the given data set, experiment key and result key
    """
    lines = []  
    labels = []
    linestyles = [] 
    colors = []
    
    ds = []

    for label, name in data_sets_chosen.items(): 
        data_set = data_sets[name]
        experiment = data_set["experiment"]
        results = experiment.experiment_results[experiment_key] 

        plot_style = data_set["plot_style"]
        lines.append(results[result_key])
        labels.append(plot_style["label"])  
        linestyles.append(plot_style["linestyle"])
        colors.append(plot_style["color"])    
        ds.append(label + "_" + experiment.recommenders[0].model_type)   

    "_".join(ds)
    file_name = file_name + str(ds)
    return {"lines": lines, "labels": labels, "linestyles": linestyles, "colors": colors, "file_name": file_name}

def plot_experiment_line(data, y_label, x_label, labels, linestyles, colors, file_name, x_upper_bound = None, sci_x=False, window=1, linewidth=2.5, smoothing=175):
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
        x, y = np.array(list(line.keys())), np.array(list(line.values()))
        if x_upper_bound is not None:
            x = np.extract(x <= np.full(len(x), x_upper_bound), x)
            y = np.extract(x <= np.full(len(x), x_upper_bound), y)

        f = interpolate.interp1d(x, y,kind = 'linear')
        xnew = np.linspace(x.min(), x.max(), smoothing) 
        plt.plot(xnew, f(xnew), label=labels[i], linestyle=linestyles[i], color=colors[i], linewidth=linewidth)
    
    plt.legend()

    plt.ylabel(y_label)
    plt.xlabel(x_label)    
    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + file_name + ".png")
    plt.show()