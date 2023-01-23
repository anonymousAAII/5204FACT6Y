import numpy as np

def f(policies, preferences, b):
    util_per_user = np.diag(policies @ preferences.T)
    total_util = np.sum(util_per_user)
    differences = util_per_user - total_util / util_per_user.shape[0]
    D = np.sum(np.square(differences))
    return total_util - b * np.sqrt(D)

def gradient(policies, preferences, b):
    util_per_user = np.diag(policies @ preferences.T)
    M = util_per_user.shape[0]
    total_util = np.sum(util_per_user)
    differences = util_per_user - total_util / M
    D = np.sum(np.square(differences))
    D_grad = ((-2 * np.sum(differences) + 2 * differences) * preferences.T).T
    grad = -preferences + b * D_grad / (2 * D)
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
def frankWolfeLASSO(preferences, b=50, tol=0.0001, K=100):
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
    x = [None] * K
    s = [None] * K
    rho = [None] * K
    data = [None] * K
    diffx = [None] * K
    x[0] = np.random.rand(preferences.shape[0], preferences.shape[1])
    x[0] /= np.sum(x[0], axis=1, keepdims=True)
    s[0] = np.random.rand(preferences.shape[0], preferences.shape[1])
    s[0] /= np.sum(s[0], axis=1, keepdims=True)

    # Apply the Frank-Wolfe Algorithm
    for k in range(1, K):
        if k % 10 == 0:
            print(k)
        rho[k] = 2 / (2 + k)
        s[k] = fwOracle(gradient(x[k - 1], preferences, b))
        x[k] = (1 - rho[k]) * x[k - 1] + rho[k] * s[k]
        data[k] = f(x[k], preferences, b)
        if k > 1:
            diffx[k - 1] = data[k] - data[k - 1]
            if tol >= abs(diffx[k - 1]): break

    # Return
    return x[k]
