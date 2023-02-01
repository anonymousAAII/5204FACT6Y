import numpy as np
from tqdm import tqdm

class Audit(object):
    def __init__(self, recommender):
        """
        Audit object can perform an audit for envy-ness in its given recommender system

        Inputs:
            recommender     - recommender system object
        """
        self.recommender = recommender   
        # Envy results of audit peforming different audit methods
        self.envy = {
                "basic": None,
                "prob": None,
                "OCEF": None
            }

        # Used parameters during audit for different audit methods
        self.params = {
                "basic": {
                    "epsilon": 0.05
                }
            }

    # Basic definition of envy-freeness in a system (see 3.1 paper)
    def __envy_free_basic(self):
        policies = self.recommender.policies
        expec_rewards = self.recommender.expec_rewards
        users, _ = policies.shape

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
        return delta_envy.flatten()

    def audit_envy(self, audit_mode="basic"):
        """
        Audit function
        """
        if audit_mode=="basic":
            return self.__envy_free_basic()
