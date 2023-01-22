import numpy as np
import random
from tqdm import tqdm

def utility(m, n, policies, probability_policies, rewards, expec_rewards):
    return np.sum(probability_policies[n] * expec_rewards[m])

# Basic definition of envy-freeness in a system (see 3.1 paper)
def envy_free_basic(policies, probability_policies, rewards, expec_rewards, epsilon=0):
    users, items = policies.shape

    # Envy per user
    envy_users = np.zeros(users)
    # One-hot encoding of envious users
    envious_users = np.zeros(users)

    # The users
    users = np.arange(users)    

    # Determine envy-freeness in whole system by checking for user-sided utility inequality
    for user in tqdm(range(len(users))):
    
        # Determine utility of own policy for current user
        u_mm = utility(user, user, policies, probability_policies, rewards, expec_rewards)
        
        # Max utility difference experienced by user
        u_delta_max = 0

        other_users = users[users != user]

        # Determine utilities of other users' policies for current user
        for other_user in other_users:
            u_mn = utility(user, other_user, policies, probability_policies, rewards, expec_rewards)
            
            # Update to track the maximum envy experiences by the current user
            u_delta = u_mn - u_mm
            if u_delta > u_delta_max:
                u_delta_max = u_delta

            if u_mm <= epsilon + u_mm:
                continue
            else:
                envious_users[user] = True

        # Save envy of user
        envy_users[user] = max(u_delta_max, 0) 

    print(np.count_nonzero(envy_users > epsilon))
    exit()
    return True

def determine_envy_freeness(policies, probability_policies, rewards, expec_rewards, mode_envy="basis"):
    # Basic definition of envy-freeness
    if mode_envy == "basic":
        envy_free = envy_free_basic(policies, probability_policies, rewards, expec_rewards)
        print(envy_free)
    # Algorithm 1: OCEF (Online Certification of Envy-Freeness) algorithm
    elif mode_envy == "OCEF":
        exit()
    # Algorithm 2: AUDIT algorithm
    elif mode_envy == "AUDIT":
        exit()
    # Default is basic envy-freeness
    else:
        envy_free_basic(policies, probability_policies, rewards, expec_rewards)

def beta(n, m):
    return

def OCEF(policies, rewards, m, K, conf_delta, conserv_explore_alpha, envy_epsilon):
    """
    OCEF (Online Certification of Envy-Freeness) algorithm

    :policies:              recommendation policies
    :rewards:               binary rewards
    :m:                     index of current user m
    :K:                     number of other users to select (subset of all existing users except m)
    :conf_delta:            confidence parameter δ
    :conserv_explore_alpha: conservative exploration parameter α
    :envy_epsilon:          envy parameter epsilon
    """
    num_users, num_items = policies.shape
    users = np.delete(np.arange(num_users), [m])
    # Take random subset of size K of all users except user m
    S_0 = np.random.choice(users, size=K, replace=False)
    print(S_0)

    for t in range(1, 2):
        # Randomly draw index of other user
        l = random.choice(S_0)
        print(l)
        if beta(0, t-1):
            break
    
    return