import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import sparse
import time
from tqdm import tqdm

from Recommender import Recommender, grid_search
from implicit.evaluation import train_test_split

def get_model_envy(policies, preferences, epsilon=0.05):
    """
    Computes the average utility of users, the average envy of users and the
    proportion of epsilon-envious users in a system given recommendation
    probabilities and ground truth preferences.
    Inputs:
        policies - Matrix with probabilities of items (columns) being
                   recommended to users (rows)
        preferences - Matrix with ground truth preferences of user (rows) for
                      items (columns)
        epsilon - Envy threshold for being considered envious
    Outputs:
        avg_util - Average utility of users
        avg_envy - Average envy of users
        prop_envious - Proportion of users with envy > epsilon
    """
    assert policies.shape == preferences.shape, \
           f"Shapes {policies.shape} and {preferences.shape} do not match."
    user_utils = np.zeros(policies.shape[0])
    user_envies = np.zeros(policies.shape[0])
    util_matrix = preferences @ policies.T
    for target_user in range(policies.shape[0]):
        utils = util_matrix[target_user]
        target_util = utils[target_user]
        max_util = np.max(utils)
        user_utils[target_user] = target_util
        user_envies[target_user] = np.max(max_util - target_util, 0)
    total_util = np.sum(user_utils)
    avg_envy = np.mean(user_envies)
    prop_envious = np.count_nonzero(user_envies > epsilon) / user_envies.shape[0]
    return total_util, avg_envy, prop_envious

def do_experiment(ground_truths, factors, seed=None):
    # Scale ground_truths between 0 and 1
    #ground_truths -= np.min(ground_truths)
    #ground_truths /= np.max(ground_truths)
    
    # Create split
    ground_truths_sparse = sparse.csr_matrix(ground_truths)
    train_matrix, test_matrix = train_test_split(ground_truths_sparse, 0.8,
                                                 seed)
    train_matrix, val_matrix = train_test_split(ground_truths_sparse, 0.875,
                                                seed)
    
    # Hyperparameters
    reg = [0.001, 0.01, 0.1, 1.]
    conf_weights = [0.1, 1., 10., 100.]
    epsilon = 0.05
    total_util_list = list()
    avg_envy_list = list()
    prop_envious_list = list()
    for factor in tqdm(factors):
        # Find best hyperparameters
        _, rec = grid_search(train_matrix, val_matrix, [factor], reg,
                             conf_weights, seed)
        # max_indices = np.unravel_index(hyperparams.argmax(), hyperparams.shape)
        # best_reg = reg[max_indices[1]]
        # best_conf_weight = conf_weights[max_indices[2]]
        # # Train recommender model
        # rec = Recommender(factors=factor, regularization=best_reg,
        #                   alpha=best_conf_weight, temperature=0.2,
        #                   compute_dense_matrix=True, random_state=seed)
        # rec.fit_model(test_matrix)
        
        # Assess envy
        total_util, avg_envy, prop_envious = get_model_envy(rec.policies,
                                                          ground_truths,
                                                          epsilon)
        total_util_list.append(total_util)
        avg_envy_list.append(avg_envy)
        prop_envious_list.append(prop_envious)
    return total_util_list, avg_envy_list, prop_envious_list

def main(dataset_name, factors):
    print(f"Experiment with {dataset_name} data started")
    start = time.perf_counter()
    with open(f'../results/{dataset_name}_ground_truths', 'rb') as f:
        gt = pickle.load(f)
    total_util, avg_envy, prop_envious = do_experiment(gt, factors, 42)
    with open(f'../results/mispecification_{dataset_name}', 'wb') as f:
        pickle.dump((avg_envy, prop_envious), f)
    end = time.perf_counter()
    print(f"Experiment with {dataset_name} data finished")
    print(f"Experiment took {end-start} seconds")
    print(f"Total utility: {total_util}")
    print(f"Average envy: {avg_envy}")
    print(f"Proportion of 0.05-envious users: {prop_envious}\n")
    return avg_envy, prop_envious

if __name__ == '__main__':
    avg_envy = dict()
    prop_envious = dict()
    datasets = ['lastfm', 'movielens']
    factors = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for dataset in datasets:
        avg_envy[dataset], prop_envious[dataset] = main(dataset, factors)
    
    for dataset in datasets:
        plt.plot(factors, avg_envy[dataset], label=dataset)
    plt.xlabel("number of factors")
    plt.ylabel("average envy")
    plt.legend()
    plt.show()

    for dataset in datasets:
        plt.plot(factors, prop_envious[dataset], label=dataset)
    plt.xlabel("number of factors")
    plt.ylabel("prop of 0.05-envious users")
    plt.legend()
    plt.show()
