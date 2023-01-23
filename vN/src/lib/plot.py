import matplotlib.pyplot as plt

# 1st party imports
import constant

def plot_experiment_5_1A(data):
    data_1 = data[0]

    fig, ax = plt.subplots()
    line1, = ax.plot(data_1.keys(), data_1.values(), label="Last fm", linestyle='-.')

    # Create a legend for the first line.
    first_legend = ax.legend(handles=[line1], loc='upper right')

    # Add the legend manually to the Axes.
    ax.add_artist(first_legend)
    plt.ylabel("average envy")
    plt.xlabel("number of factors")    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + 'experiment_5_1A.png')
    plt.show()


