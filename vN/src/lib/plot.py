import matplotlib.pyplot as plt

# 1st party imports
import constant

def plot_experiment_5_1A(data):
    fm = data[0]
    mv = data[1]

    fig, ax = plt.subplots()
    plt.plot(fm.keys(), fm.values(), label="Last fm", linestyle='-.')
    plt.plot(mv.keys(), mv.values(), label="MovieLens", linestyle='dotted', color="orange")

    plt.legend()
    plt.ylabel("average envy")
    plt.xlabel("number of factors")    
    plt.grid()
    plt.savefig(constant.RESULTS_FOLDER + 'experiment_5_1A.png')
    plt.show()
