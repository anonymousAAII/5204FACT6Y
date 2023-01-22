import numpy as np
import random

# Basic definition of envy-freeness in a system (see 3.1 paper)
def envy_free_basic(policies):
    return

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