import math
import numpy as np

def bound_size(N, r, K, theta, omega, sigma):
    """
    Computes the size of the confidence bound beta.
    Inputs:
        N - Number of times arm has been pulled
        r - Sum of rewards from pulling that branch
        K - Number of arms at the start of OCEF
        theta - See lemma 4
        omega - See lemma 4
        sigma - See lemma 4
    """
    left = np.sqrt((2 * sigma ** 2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / N)
    right = np.sqrt(np.log((2 * (K + 1)) / theta * np.log((1 + omega) * N)))
    beta = left * right
    return beta

def OCEF(target_policy, other_policies, reward_func, delta, alpha, epsilon):
    """
    Impliments the OCEF algorithm.
    Inputs:
        target_policy - Policy of the target user
        other_policies - List of policies of other users
        reward_func - Function that takes a policy and returns a reward
        delta - Confidence parameter
        alpha - Conservative exploration parameter
        epsilon - Envy parameter
    Outputs:
        True if target user is envious, False if not epsilon-envious
    """
    t = 0
    K = len(other_policies)
    omega = 1
    sigma = 0.5
    theta = np.log(1 + omega) * math.pow((omega * delta) / (2 * (2 * omega)),
                                         1 / (1 + omega))
    while other_policies:
        t += 1
        policy = np.random.choice(other_policies)
    return False 
