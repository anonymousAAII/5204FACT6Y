import numpy as np
import random
from tqdm import tqdm

def utility(m, n, policies, probability_policies, rewards, expec_rewards):
    return np.sum(probability_policies[n] * expec_rewards[m])

# Basic definition of envy-freeness in a system (see 3.1 paper)
def envy_free_basic(policies, probability_policies, rewards, expec_rewards, epsilon=0.05, gamma=0.5, lamb=0.5):
    print("Auditing envy through basic definition...")
    users, items = policies.shape

    # RELAXATION CONDITIONS: such that we do not have to try ALL users and ALL policies
    # Threshold of when an user is considered envious
    # The larger gamma the more policies of other users should give the user envy before he become envious
    envious_user_threshold = int(gamma * users)
    # Threshold of users that should not be envious
    envy_free_users_threshold = users * (1 - lamb)

    # Envy per user
    envy_users = np.zeros(users)

    # The users
    users = np.arange(users)    

    num_envy_free_users = 0
    envious_system = True

    # Determine envy-freeness in whole system by checking for user-sided utility inequality
    for user in tqdm(range(len(users))):
    
        # Determine utility of own policy for current user
        u_mm = utility(user, user, policies, probability_policies, rewards, expec_rewards)
        
        # Max utility difference experienced by user
        u_delta_max = 0

        envious_count = 0

        other_users = users[users != user]
        # Randomly shuffle since users should come from a discrete uniform distribution
        np.random.shuffle(other_users)

        envious_user = False

        # Determine utilities of other users' policies for current user
        for other_user in other_users:
            u_mn = utility(user, other_user, policies, probability_policies, rewards, expec_rewards)
            
            # Update to track the maximum envy experienced by the current user
            u_delta = u_mn - u_mm
            if u_delta > u_delta_max:
                u_delta_max = u_delta

            # User has envy
            if u_mm + epsilon < u_mn:
                envious_count += 1                    

            # RELAXATION: when at least this number of other policies make the user envy consider him envious 
            if envious_count > envious_user_threshold:
                envious_user = True
                break

        # Save envy of user
        envy_users[user] = max(u_delta_max, 0) 

        if envious_user == False:
            num_envy_free_users += 1

        # RELAXATION: check whether system is envy-free
        if num_envy_free_users == envy_free_users_threshold:
            envious_system = False
            break            

    envy_free = not envious_system
    M = len(envy_users)
    average_envy_per_user = np.sum(envy_users) / M
    proportion_envious_users = np.count_nonzero(envy_users > epsilon) / M
    return {"envy_free": envy_free, "avg_envy_user": average_envy_per_user, "prop_envious_users": proportion_envious_users}

def determine_envy_freeness(policies, probability_policies, rewards, expec_rewards, mode_envy="basis"):
    # Basic definition of envy-freeness
    if mode_envy == "basic":
        envy_results = envy_free_basic(policies, probability_policies, rewards, expec_rewards)
    # Algorithm 1: OCEF (Online Certification of Envy-Freeness) algorithm
    elif mode_envy == "OCEF":
        exit()
    # Algorithm 2: AUDIT algorithm
    elif mode_envy == "AUDIT":
        exit()
    # Default is basic envy-freeness
    else:
        envy_results = envy_free_basic(policies, probability_policies, rewards, expec_rewards)
    
    print(envy_results)
    return envy_results

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