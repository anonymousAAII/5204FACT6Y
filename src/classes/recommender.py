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

        
 