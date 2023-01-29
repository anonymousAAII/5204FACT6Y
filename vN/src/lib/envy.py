####
# vN/src/lib/envy.py
# 
# This file contains all the functionality to audit for envy(-freeness) in a recommender system 
####
import numpy as np
from tqdm import tqdm

# def utility(m, n, probability_policies, expec_rewards):
#     return np.sum(probability_policies[n] * expec_rewards[m])

# Basic definition of envy-freeness in a system (see 3.1 paper)
def envy_free_basic(recommendations, policies, expec_rewards, preferences, epsilon=0.05, gamma=0.5, lamb=0.5, relax_criterion=False):
    """
    Checks whether a system is envy-free according to the basic definition.

    :recommendations:       recommender system's recommendation
    :policies:              probability distribution over the items of getting picked/recommended -> policies
    :rewards:               binary rewards
    :expec_rewards:         the expectation of the rewards
    :epsilon:               determines how much deviation the utility u_mm and u_mn are allowed to have to be still considered equal
    :gamma:                 determines the envious threshold for a USER. The larger gamma the more policies of other users should give 
                            the user envy before he is considered envious
    :lamb:                  determines the envy threshold for the SYSTEM. A system is considered envy-free if at leas (1 - lambda) users are not envious
    :returns:               data = {"envy_free": <envy_free>, "avg_envy_user": <average_envy_per_user>, "prop_envious_users": <proportion_envious_users>}
    """
    print("Checking for envy through basic definition...")
    users, _ = recommendations.shape

    # # RELAXATION CONDITIONS: such that we do not have to try ALL users and ALL policies
    # # Threshold of when an user is considered envious
    # envious_user_threshold = int(gamma * users)
    # # Threshold of users that should not be envious
    # envy_free_users_threshold = users * (1 - lamb)

    # This whole section belows makes use of the properties of lineair algebra to compute sources of envy

    # According to definition utility = policy * expectation rewards
    utilities = policies @ expec_rewards.T

    # Column index = index of user policy, row index = index of user to which the policy was applied
    delta_utilities = utilities - np.repeat([utilities.diagonal()], users, axis=0)
    
    # Get maximum envy experienced by users (i.e. for each user)
    max_delta_utilities = delta_utilities.max(axis=0, keepdims=True)
    delta_envy = np.maximum(max_delta_utilities, np.zeros(users))

    # Envy for each user
    envy_users = delta_envy.flatten()

    average_envy_per_user = np.mean(envy_users)
    proportion_envious_users = np.mean(envy_users > epsilon)

    # <envy_free> temporarily None
    return {"envy_free": None, "avg_envy_user": average_envy_per_user, "prop_envious_users": proportion_envious_users}

def determine_envy_freeness(recommendations, policies, rewards, expec_rewards, preferences, mode_envy="basis"):
    """
    Determines envy-freeness in a system according to different methods

    :recommendations:       recommender system's recommendations
    :policies:              probability distribution over the items of getting picked/recommended -> policies
    :rewards:               binary rewards
    :expec_rewards:         the expectation of the rewards
    :mode_envy:             modus that specifies which method should be applied to audit envy-freeness in the system
    :returns:               audit results 
    """
    # Basic definition of envy-freeness
    if mode_envy == "basic":
        envy_results = envy_free_basic(recommendations, policies, expec_rewards, preferences)
    # Algorithm 1: OCEF (Online Certification of Envy-Freeness) algorithm
    elif mode_envy == "OCEF":
        # TO DO
        exit()
    # Algorithm 2: AUDIT algorithm
    elif mode_envy == "AUDIT":
        # TO DO
        exit()
    # Default is basic envy-freeness
    else:
        envy_results = envy_free_basic(recommendations, policies, expec_rewards, preferences)
    
    print(envy_results)
    return envy_results

# def beta(n, m):
#     return

# def OCEF(policies, rewards, m, K, conf_delta, conserv_explore_alpha, envy_epsilon):
#     """
#     OCEF (Online Certification of Envy-Freeness) algorithm

#     :policies:              recommendation policies
#     :rewards:               binary rewards
#     :m:                     index of current user m
#     :K:                     number of other users to select (subset of all existing users except m)
#     :conf_delta:            confidence parameter δ
#     :conserv_explore_alpha: conservative exploration parameter α
#     :envy_epsilon:          envy parameter epsilon
#     """
#     num_users, num_items = policies.shape
#     users = np.delete(np.arange(num_users), [m])
#     # Take random subset of size K of all users except user m
#     S_0 = np.random.choice(users, size=K, replace=False)
#     print(S_0)

#     for t in range(1, 2):
#         # Randomly draw index of other user
#         l = random.choice(S_0)
#         print(l)
#         if beta(0, t-1):
#             break
    
#     return