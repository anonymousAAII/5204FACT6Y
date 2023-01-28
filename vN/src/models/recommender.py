import numpy as np
import torch
import torch.nn.functional as F

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k

class Recommender(object):
    def __init__(self, model, model_type, params, ground_truth, preference_est, pred_perf, pred_metric, 
                    recommendations, policies, rewards, expec_rewards):
        """
        Implements an alternating least squares model as a recommender system.
        Inputs:
            
        """
        self.model = model
        self.model_type = model_type
        self.params = params
        self.ground_truth = ground_truth
        self.preference_est = preference_est
        self.pred_perf = pred_perf
        self.pred_metric = pred_metric
        self.recommendations = recommendations
        self.policies = policies

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