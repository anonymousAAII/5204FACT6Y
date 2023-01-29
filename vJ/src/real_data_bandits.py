import numpy as np
import pickle
from scipy import sparse
from tqdm import tqdm

from auditing import AUDIT
from implicit.evaluation import train_test_split
from Recommender import grid_search

def experiment(policies, preferences, N, delta, alpha, epsilon, gamma, lambda_):
    def reward_func(policy, user):
        item = np.random.choice(policy.shape[0], p=policy)
        if preferences[user][item] > np.random.rand():
            return 1
        else:
            return 0
    for _ in tqdm(range(N)):
        envy, duration, cost = AUDIT(policies, reward_func, delta, alpha,
                                     epsilon, gamma, lambda_)
        print(envy, duration, cost)

def main(ground_truths, seed=None):
    # Scale ground_truths between 0 and 1
    ground_truths -= np.min(ground_truths)
    ground_truths /= np.max(ground_truths)
    
    # Create split
    ground_truths_sparse = sparse.csr_matrix(ground_truths)
    train_matrix, test_matrix = train_test_split(ground_truths_sparse, 0.8,
                                                 seed)
    train_matrix, val_matrix = train_test_split(ground_truths_sparse, 0.875,
                                                seed)
    
    # Hyperparameters
    factor = 48
    reg = [0.001, 0.01, 0.1, 1.]
    conf_weights = [0.1, 1., 10., 100.]
    temperature = 0.2
    N = 10
    delta = 0.05
    alpha = 0.15
    epsilon = 0.05
    gamma = 0.1
    lambda_ = 0.1
    _, rec = grid_search(train_matrix, val_matrix, [factor], reg, conf_weights,
                         temperature, seed)
    experiment(rec.policies, ground_truths, N, delta, alpha, epsilon, gamma, lambda_)

if __name__ == '__main__':
    with open('../results/lastfm_ground_truths', 'rb') as f:
        gt = pickle.load(f)
    main(gt, seed=42)
