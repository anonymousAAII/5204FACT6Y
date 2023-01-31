import numpy as np
import torch
import torch.nn.functional as F

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k

class Recommender(object):
    def __init__(self, model, model_type, params, ground_truth, preference_est, performance, pred_metric, 
                    recommendations, policies, rewards, expec_rewards):
        """
        Initialize recommender system object
        
        Inputs:
            
        """
        self.model = model
        self.model_type = model_type
        self.params = params
        
        self.pred_metric = pred_metric
        self.performance = performance

        self.ground_truth = ground_truth
        self.preferences = preference_est

        self.policies = policies
        self.recommendations = recommendations

        self.rewards = rewards
        self.expec_rewards = expec_rewards

    def recommend(self, user_id):
        """
        Inputs:

        Outputs:
 
        """
        return self.recommendations[user_id]

    def get_policy(self, user_id):
        """
        Inputs:

        Outputs:
 
        """
        return self.policies[user_id] 