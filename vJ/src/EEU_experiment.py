###############################################################################
# This code was taken from
# https://github.com/paulmelki/Frank-Wolfe-Algorithm-Python/blob/master/frank_wolfe.py
# with adjustments made to suite it to our purposes.
###############################################################################

import numpy as np
import pickle
from tqdm import tqdm

from misspecification_experiment import get_model_envy

def loss(policies, preferences, b):
    """
    The function we are trying to minimize.
    """
    util_per_user = np.diag(policies @ preferences.T)
    total_util = np.sum(util_per_user)
    result = -total_util
    if b != 0:
        differences = util_per_user - total_util / util_per_user.shape[0]
        D = np.sum(np.square(differences))
        result += b * np.sqrt(D)
    return result

def gradient(policies, preferences, b):
    """
    The gradient of the loss function.
    """
    preferences
    grad = -preferences
    if b != 0:
        util_per_user = np.diag(policies @ preferences.T)
        M = util_per_user.shape[0]
        total_util = np.sum(util_per_user)
        differences = util_per_user - total_util / M
        D = np.sum(np.square(differences))
        D_grad = ((-2 * np.sum(differences) + 2 * differences) * preferences.T).T
        grad += b * D_grad / (2 * D)
    return grad

# %% Define function to find a point in the Oracle
def fwOracle(gradient):
    """
    Function that computes the Frank-Wolfe Oracle defined as:
        argmin(s) <gradient(f(x)), s> where s in the feasible
        set D and < > is the inner product.
    :param gradient: (p, 1) numpy vector
                Should be the gradient of f(x)
    :return: s: (p, 1) numpy vector
                FW Oracle as defined above
    """
    # Initialize the zero vector
    s = np.random.rand(gradient.shape[0], gradient.shape[1])

    # Check if all coordinates of x are 0
    # If they are, then the Oracle contains zero vector
    if (gradient != 0).sum() == 0:
        return s

    # Otherwise, follow the following steps
    else:
        # Compute the (element-wise) absolute value of x
        a = abs(gradient)
        # Find the first coordinate that has the maximum absolute value
        i = np.unravel_index(a.argmax(), a.shape)
        # Compute s
        s[i] = - np.sign(gradient[i])
        s /= np.sum(s, axis=1, keepdims=True)
        return s

#%% Define function for applying the Frank-Wolfe algorithm to solve LASSO problem
def frankWolfe(preferences, b=50, tol=0.0001, K=1000):
    """
    Function that applies the Frank-Wolfe algorithm on a LASSO problem, given
    the required x, A, y and K.
    :param A: (n, p) numpy matrix
                Design matrix of the LASSO problem
    :param y: (n, ) numpy vector
                Target vector of the LASSO problem
    :param b: float
                penalty parameter
    :param K: integer > 0
                Maximum number of iterations
    :param tol: float > 0
                    Tolerance rate of the error ||f(x_k) - f(x_(k-1))||
    :return: data: f(x): K-dimensional numpy vector
                argmin(D) of f
            diffx: (K-1)-dimensional numpy vector
                difference ||f(x_k) - f(x_(k-1))||
            k: integer > 0
                The number of iterations made
    """
    # Initialise:
    # x : sequence of K data points
    #       (each being a p-dimensional vector of features)
    # s : sequence of K "oracles"
    #       (each being a p-dimensional vector)
    # rho : step-size sequence having K elements
    # data : K-dimensional vector of resulting data points
    # data : (K-1)-dimensional vector of the difference f(x_k) - f(x_(k-1))
    # x[0] and s[0] to p-dimensional vectors of zeros (starting points)
    data = [None] * K
    x = np.random.rand(preferences.shape[0], preferences.shape[1])
    x /= np.sum(x, axis=1, keepdims=True)
    s = np.random.rand(preferences.shape[0], preferences.shape[1])
    s /= np.sum(s, axis=1, keepdims=True)

    # Apply the Frank-Wolfe Algorithm
    for k in tqdm(range(1, K)):
        rho = 2 / (2 + k)
        s = fwOracle(gradient(x, preferences, b))
        x = (1 - rho) * x + rho * s
        data[k] = loss(x, preferences, b)
        if k > 1:
            diffx = data[k] - data[k - 1]
            if tol >= abs(diffx): break

    # Return
    return x

if __name__ == '__main__':
    datasets = ['lastfm', 'movielens']
    np.random.seed(0)
    for dataset in datasets:
        print(f"\nComputing policies for {dataset}")
        with open(f'../results/{dataset}_ground_truths', 'rb') as f:
            gt = pickle.load(f)
        exp_dict = dict()
        EUU_policies = frankWolfe(gt, b=50)
        OPT_policies = frankWolfe(gt, b=0)
        EEU_results = get_model_envy(EUU_policies, gt)
        OPT_results = get_model_envy(OPT_policies, gt)
        exp_dict['EEU'] = EEU_results
        exp_dict['OPT'] = OPT_results
        with open(f'../results/EEU_{dataset}', 'wb') as f:
            pickle.dump(exp_dict, f)
        print(f"\nEEU:\nTotal utility: {EEU_results[0]}")
        print(f"Average envy: {EEU_results[1]}\n")
        print(f"Proportion of 0.05-envious users: {EEU_results[2]}\n")
        print(f"OPT:\nTotal utility: {OPT_results[0]}")
        print(f"Average envy: {OPT_results[1]}\n")
        print(f"Proportion of 0.05-envious users: {OPT_results[2]}\n")
