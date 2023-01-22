import numpy as np
import random

def utility(m, n, policies, probability_policies, rewards, expec_rewards):
    _, items = policies.shape

    u = 0
    # Sum through possible items 
    for item_a in range(items):    
        u += probability_policies[n][item_a] * expec_rewards[m][item_a]
    return u

# Basic definition of envy-freeness in a system (see 3.1 paper)
def envy_free_basic(policies, probability_policies, rewards, expec_rewards):
    users, items = policies.shape
    
    # Only try for one user
    for user in range(1):
        u = utility(user, user, policies, probability_policies, rewards, expec_rewards)
        print(u)
    return

def determine_envy_freeness(policies, probability_policies, rewards, expec_rewards, mode_envy="basis"):
    # Basic definition of envy-freeness
    if mode_envy == "basic":
        envy_free_basic(policies, probability_policies, rewards, expec_rewards)
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