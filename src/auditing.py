import math
import numpy as np
from tqdm import tqdm

def bound_size(N, K, theta, omega, sigma):
    """
    Computes the size of the confidence bound beta.
    Inputs:
        N - Number of times arm has been pulled, must be greater than zero
        K - Number of arms at the start of OCEF
        theta - See lemma 4
        omega - See lemma 4
        sigma - See lemma 4
    """
    left = np.sqrt((2 * sigma ** 2 * (1 + np.sqrt(omega)) ** 2 * (1 + omega)) / N)
    right = np.sqrt(np.log((2 * (K + 1)) / theta * np.log((1 + omega) * N)))
    beta = left * right
    return beta

def conservative_estimate(index_selection, Ns, mus, betas, alpha, delta, sigma,
                          A, r, t):
    """
    Computes the conservative estimate.
    Inputs (for all arrays, first arm is assumed to be the baseline):
        index_selection - Index of arm selected for exploration
        Ns - Array of number of times each arm was pulled
        mus - Array of estimated utilities for each arm
        betas - Array of bound sizes for each arm
        alpha - Conservative exploration parameter
        delta - Confidence parameter
        sigma - See lemma 4
        A - Number of times an arm other than the baseline was selected
        r - Sum of rewards from the times an arm other than the baseline was
            selected
        t - Timestep
    """
    if A != 0:
        phi_left = sigma * np.sqrt(2 * A * np.log((6 * A ** 2) / delta))
        phi_right = 2 / 3 * np.log((6 * A ** 2) / delta)
        phi = phi_left + phi_right
    else:
        phi = 0
    beta_N = 0
    beta_N = betas[1:] @ Ns[1:]
    Phi = min(beta_N, phi)
    l_low = mus[index_selection] - betas[index_selection]
    target_upper = mus[0] + betas[0]
    ksi = r - Phi + l_low + (Ns[0] - (1 - alpha) * t) * target_upper
    return ksi
    

def OCEF(policies, reward_func, delta, alpha, epsilon, sigma=0.5, omega=0.01):
    """
    Implements the OCEF algorithm to audit whether a target user is envious of
    other users.
    Inputs:
        policies - Numpy array of policies, first entry is assumed to be the
                   baseline
        reward_func - Function that takes a policy and returns a reward
        delta - Confidence parameter
        alpha - Conservative exploration parameter
        epsilon - Envy parameter
    Outputs:
        envy - True if target user is envious, False if not epsilon-envious
        t - The number of timesteps
        cost - Cost of the audit
    """
    t = 0 # timestep
    K = len(policies) - 1 # number of other policies
    A = 0 # number of times arm other than baseline was selected
    r = 0 # sum of rewards from times arm other than baseline was selected
    theta = np.log(1 + omega) * math.pow((omega * delta) / (2 * (2 + omega)),
                                         1 / (1 + omega)) # see lemma 4
    # Bound size at start, for N = 0, beta must be bigger than for N = 1
    beta_start = bound_size(1, K, theta, omega, sigma) + 1e-08
    beta_min = beta_start # Smallest beta of arm other than baseline
    # For each policy, store:
        # N: number of times pulled
        # r: sum of reward over times pulled
        # mu: average reward
        # beta: bound size
    # target_dict = {'policy': target_policy, 'N': 0, 'r': 0, 'mu': 0,
    #                'beta': beta_start}
    Ns = np.zeros(K + 1)
    rs = np.zeros(K + 1)
    mus = np.zeros(K + 1)
    betas = np.full(K + 1, beta_start)
    # S = [] # List of dictionaries of other policies
    # for policy in other_policies:
    #     S.append({'policy': policy, 'N': 0, 'r': 0, 'mu': 0,
    #               'beta': beta_start})
    while policies.shape[0] > 1:
        t += 1
        policy_index = np.random.choice(policies.shape[0] - 1) + 1
        #policy_dict = S[policy_index] # Choose policy to explore
        if conservative_estimate(policy_index, Ns, mus, betas, alpha, delta,
                                 sigma, A, r, t) < 0 or betas[0] > beta_min:
            # Pull baseline, do not explore
            policy_index = 0
        reward = reward_func(policies[policy_index])
        # Update policy's variables
        Ns[policy_index] += 1
        rs[policy_index] += reward
        mus[policy_index] = rs[policy_index] / Ns[policy_index]
        betas[policy_index] = bound_size(Ns[policy_index], K, theta, omega, sigma)
        target_lower = mus[0] - betas[0] + epsilon
        target_upper = mus[0] + betas[0]
        if policy_index != 0: # If we pulled an arm other than the baseline
            # Update variables
            A += 1
            r += reward
            beta_min = min(beta_min, betas[policy_index])
            if mus[policy_index] - betas[policy_index] > target_upper:
                # Lower bound greater than target upper bound: envy
                # Compute cost of audit
                envy = True
                cost = t * mus[0] - rs[0] - r
                return envy, t, cost
            if mus[policy_index] + betas[policy_index] <= target_lower:
                # Upper bound smaller than target lower bound: remove policy from list
                # del S[policy_index]
                policies = np.delete(policies, policy_index, axis=0)
                Ns = np.delete(Ns, policy_index)
                rs = np.delete(rs, policy_index)
                mus = np.delete(mus, policy_index)
                betas = np.delete(betas, policy_index)
        else: # If we pulled the baseline
            # temp_list = []
            # for other_dict in S:
            #     if other_dict['mu'] - other_dict['beta'] > target_upper:
            #         # Lower bound greater than target upper bound: envy
            #         # Compute cost of audit
            #         envy = True
            #         cost = t * target_dict['mu'] - target_dict['r'] - r
            #         return envy, t, cost
            #     # Check for each policy if upper bound is still greater than target lower bound
            #     if target_lower < other_dict['mu'] + other_dict['beta']:
            #         temp_list.append(other_dict)
            # S = temp_list
            if np.any(mus[1:] - betas[1:] > target_upper):
                # Lower bound greater than target upper bound: envy
                # Compute cost of audit
                envy = True
                cost = t * mus[0] - rs[0] - r
                return envy, t, cost
            # Keep policies where upper bound is still greater than target lower bound
            indices_to_keep = np.nonzero(mus[1:] + betas[1:] > target_lower)[0] + 1
            indices_to_keep = np.concatenate((np.array([0]), indices_to_keep))
            policies = policies[indices_to_keep]
            Ns = Ns[indices_to_keep]
            rs = rs[indices_to_keep]
            mus = mus[indices_to_keep]
            betas = betas[indices_to_keep]
    # List of other policies empty, not epsilon-envious
    # Compute cost of audit
    envy = False
    cost = t * mus[0] - rs[0] - r
    return envy, t, cost

def AUDIT(policies, reward_func, delta, alpha, epsilon, gamma, lambda_):
    """
    Implements the AUDIT algorithm to audit whether a system is has envy or is
    envy-free with probabilistic relaxations. Warning: is currently very slow.
    Inputs:
        policies - Policies where the entry represents the probability of an
                   item (column) being recommended to a user (row)
        reward_func - Function that takes a policy and a user index and returns
                      a reward for that user based on an item picked by the
                      policy.
        delta - Confidence parameter
        alpha - Conservative exploration parameter
        epsilon - Envy parameter
        gamma - Envy parameter
        lambda_ - Envy parameter
    """
    M = math.ceil(np.log(3 / delta) / lambda_)
    S_indices = np.random.choice(len(policies), size=M, replace=False)
    K = math.ceil(np.log(3 * M / delta) / np.log(1 / (1 - gamma)))
    avg_duration = 0
    avg_cost = 0
    for S_index in S_indices:
        arms = np.concatenate((policies[:S_index], policies[S_index + 1:]))
        arms = arms[np.random.choice(len(arms), size=K, replace=False)]
        arms = np.insert(arms, 0, policies[S_index], axis=0)
        f_reward = lambda x: reward_func(x, S_index)
        envy, duration, cost = OCEF(arms, f_reward, delta, alpha, epsilon)
        if envy:
            return envy, duration, cost
        avg_duration += duration
        avg_cost += cost
    envy = False
    avg_duration /= K
    avg_cost /= K
    return envy, avg_duration, avg_cost
