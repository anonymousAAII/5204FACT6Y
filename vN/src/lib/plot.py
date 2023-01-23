import matplotlib.pyplot as plt

def plot(data):
    keys = data[0].keys()

    fig, ax = plt.subplots()

    for experiment in data:
        line1, = ax.plot(keys, experiment.values(), label="Last fm", linestyle='-.')

        # Create a legend for the first line.
        first_legend = ax.legend(handles=[line1], loc='upper right')

        # Add the legend manually to the Axes.
        ax.add_artist(first_legend)
    
    plt.show()
