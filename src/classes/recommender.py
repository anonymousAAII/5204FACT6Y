import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler

class Recommender(object):
    def __init__(self, data_set, normalized, model, model_type, params, perf, metric, ground_truth, temperature):
        """
        Initialize recommender system object
        
        Inputs:
            
        """
        self.data_set = data_set
        self.normalized = normalized
        self.model = model
        self.model_type = model_type
        self.params = params
        
        self.metric = metric
        self.perf = perf

        self.ground_truth = ground_truth
        
        self.set_preferences()
        self.temperature = temperature

        self.set_policies()
        self.set_recommendations()

        self.set_rewards()
        self.set_expec_rewards()
        
        self.set_u()
        self.set_u_OPT()
        self.set_u_m()

    def set_preferences(self):
        print("Initializing preferences...")

        # Funky SVD
        if self.model_type == "svd":
            row, col = self.ground_truth.shape
            u_id = np.repeat(np.arange(0, row), col)
            i_id = np.tile(np.arange(0, col), row)
            
            # Construct all user-item pairs for the recommender system model
            df = pd.DataFrame({"u_id": u_id, "i_id": i_id})
            
            # Calculate estimated preference scores
            pred = self.model.predict(df)

            self.preferences = np.array(pred).reshape(self.ground_truth.shape)
        # ALS
        else:
            self.preferences = self.model.user_factors @ self.model.item_factors.T

    def set_policies(self):
        print("Generating policies...")

        # Apply temperature parameter   
        divider = np.full(self.preferences.shape[0], self.temperature)
        policies = np.divide(self.preferences.T, divider).T
        
        # Compute the softmax transformation along the second axis (i.e., the rows)
        self.policies = scipy.special.softmax(policies, axis=1)

    # This whole section belows makes use of the properties of lineair algebra to compute sources of envy
    def set_u(self):
        # According to definition utility = policy * expectation rewards
        utilities = self.policies @ self.expec_rewards.T
        self.u = utilities

    def set_u_OPT(self):
        """
            Finds the utilies of unconstrainted optimal policies
        """    
        # Contains the OPT utilities per user in the matrix form 
        # Non zero elements contain the OPT per user -> coordinate = <policy_index>, <user_index>
        # so each column has exactly one none zero element
        self.u_OPT = self.u * (self.u >= np.sort(self.u, axis=0)[[-1],:]).astype(int)

    def set_u_m(self):
        """
            Find utilities u_m for policy m (so the utility for each user of its own policy) 
        """
        self.u_m = np.zeros(self.u.shape)
        np.fill_diagonal(self.u_m, self.u.diagonal())

    def set_u_EUU(self):
        """
            Find the Equal User Utility using the Frank-Wolfe algorithm
        """
        # Use the policies as the initial guess of the Frank-Wolfe algorithm
        p_init = self.policies

    def set_recommendations(self):
        print("Generating recommendations...")

        # According to a given policy i.e. a probability distribution 
        # select an item by drawing from this distribution -> is the recommendation
        def select_policy(distribution, indices):
            i_drawn_policy = np.random.choice(indices, 1, p=distribution)
            recommendation = np.zeros(len(indices))
            recommendation[i_drawn_policy] = 1
            return recommendation

        # Since stationary policies pick a recommendation for an user only once
        indices = np.arange(len(self.policies[0]))
        self.recommendations = np.apply_along_axis(select_policy, 1, self.policies, indices)
        
    def set_rewards(self):
        print("Generating binary rewards...")

        rewards = np.zeros(self.ground_truth.shape)

        # Based on the approximated interest, the rewards are drawn from the binomial distribution. 
        # The higher the interest the more likely it is to order the service and vice-versa.
        def draw_from_bernoulli(x):
            return np.random.binomial(1, 0.5)
        
        apply_bernoulli = np.vectorize(draw_from_bernoulli)   
        self.rewards = apply_bernoulli(rewards)

    def set_expec_rewards(self):
        print("Setting expectation of the rewards...")
        # When ground truth relevance scores have not been normalized
        # Scale to range [0, 1] to derive the expecation of the rewards (i.e. from Bernoulli dist.)
        if self.normalized:
            self.expec_rewards = np.copy(self.ground_truth)
        else:
            scaler = MinMaxScaler()
            model = scaler.fit(self.ground_truth)
            self.expec_rewards = model.transform(self.ground_truth)

        
 