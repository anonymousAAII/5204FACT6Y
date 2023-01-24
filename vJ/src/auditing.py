import math
import numpy as np

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

def conservative_estimate(target_dict, policy_dicts, selected_dict, alpha,
                          delta, sigma, A, r, t):
    """
    Computes the conservative estimate.
    Inputs:
        target_dict - Dictionary for target user
        policy_dicts - List of dictionaries for other users
        selected_dict - Dictionary for selected user
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
    for policy_dict in policy_dicts:
        beta_N += policy_dict['beta'] * policy_dict['N']
    Phi = min(beta_N, phi)
    l_low = selected_dict['mu'] - selected_dict['beta']
    target_high = target_dict['mu'] + target_dict['beta']
    ksi = r - Phi + l_low + (target_dict['N'] - (1 - alpha) * t) * target_high
    return ksi
    

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
        t - The number of timesteps
    """
    t = 0 # timestep
    K = len(other_policies) # number of other policies
    omega = 0.5 # see lemma 4
    sigma = 0.5 # see lemma 4
    A = 0 # number of times arm other than baseline was selected
    r = 0 # sum of rewards from times arm other than baseline was selected
    theta = np.log(1 + omega) * math.pow((omega * delta) / (2 * (2 * omega)),
                                         1 / (1 + omega)) # see lemma 4
    # Bound size at start, for N = 0, beta must be bigger than for N = 1
    beta_start = bound_size(1, K, theta, omega, sigma) + 1e-08
    beta_min = beta_start # Smallest beta of arm other than baseline
    # For each policy, store:
        # N: number of times pulled
        # r: sum of reward over times pulled
        # mu: average reward
        # beta: bound size
    target_dict = {'policy': target_policy, 'N': 0, 'r': 0, 'mu': 0,
                   'beta': beta_start}
    policy_dicts = []
    for policy in other_policies:
        policy_dicts.append({'policy': policy, 'N': 0, 'r': 0, 'mu': 0,
                             'beta': beta_start})
    while policy_dicts:
        t += 1
        explore = True
        policy_dict = np.random.choice(policy_dicts)
        if conservative_estimate(target_dict, policy_dicts, policy_dict, alpha, delta,
                                 sigma, A, r, t) < 0 or target_dict['beta'] > beta_min:
            # Pull baseline, do not explore
            policy_dict = target_dict
            explore = False
        reward = reward_func(policy_dict['policy'])
        # Update policy's variables
        policy_dict['N'] += 1
        policy_dict['r'] += reward
        policy_dict['mu'] = policy_dict['r'] / policy_dict['N']
        policy_dict['beta'] = bound_size(policy_dict['N'], K, theta, omega, sigma)
        if explore: # If we pulled an arm other than the baseline
            # Update variables
            A += 1
            r += reward
            beta_min = min(beta_min, policy_dict['beta'])
            if policy_dict['mu'] - policy_dict['beta'] > target_dict['mu'] + target_dict['beta']:
                # Lower bound greater than target upper bound: envy
                # Compute cost of audit
                cost = t * target_dict['mu'] - target_dict['r'] - r
                return True, t, cost
            if policy_dict['mu'] + policy_dict['beta'] <= target_dict['mu'] - target_dict['beta'] + epsilon:
                # Upper bound smaller than target lower bound: remove policy from list
                policy_dicts.remove(policy_dict)
        else:
            temp_list = []
            for other_dict in policy_dicts:
                # Check for each policy if upper bound is still greater than target lower bound
                if policy_dict['mu'] - policy_dict['beta'] + epsilon < other_dict['mu'] + other_dict['beta']:
                    temp_list.append(other_dict)
            policy_dicts = temp_list
    # List of other policies empty, not epsilon-envious
    # Compute cost of audit
    cost = t * target_dict['mu'] - target_dict['r'] - r
    return False, t, cost

def AUDIT():
    pass
