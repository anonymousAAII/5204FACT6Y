import numpy as np
import torch
import torch.nn.functional as F
from implicit.als import AlternatingLeastSquares

class Recommender(object):
    def __init__(self, factors=100, regularization=0.01, alpha=1.0,
                 iterations=15, temperature=1, compute_dense_matrix=False,
                 random_state=None):
        """
        Implements an alternating least squares model as a recommender system.
        Inputs:
            factors - Number of hidden factors of the model
            regularization - Regularization parameter of the model
            alpha - Confidence weights of the model
            iterations - Number of training iterations
            temperature - Softmax temperature
            compute_dense_matrix - If set to True, computes the full dense
                                   matrix. Not feasible for very large datasets
                                   due to memory usage.
        """
        self.model = AlternatingLeastSquares(factors=factors,
                                             regularization=regularization,
                                             alpha=alpha,
                                             iterations=iterations,
                                             random_state=random_state)
        self.temperature = temperature
        self.dense_matrix = compute_dense_matrix
    
    def fit_model(self, users_items):
        """
        Trains the model.
        Inputs:
            users_items - A sparse matrix were rows represent user, columns
                          represent items and values represent confidence.
        """
        self.model.fit(users_items, show_progress=False)
        if self.dense_matrix:
            self.preferences = self.model.user_factors @ self.model.item_factors.T
            self.policies = F.softmax(
                torch.Tensor(self.preferences) / self.temperature,
                dim=1).numpy()
            # Normalize so sum is close enough to 1
            self.policies /= np.sum(self.policies, axis=1, keepdims=True)
    
    def recommend(self, user_id):
        """
        Samples an item to recommend to a user using a softmax policy.
        Inputs:
            user_id - Index of the user to recommend an item to.
        Outputs:
            item - Recommended item
        """
        if self.dense_matrix:
            policy = self.policies[user_id]
        else:
            preferences = self.model.user_factors[user_id] @ self.model.item_factors.T
            policy = F.softmax(torch.Tensor(preferences) / self.temperature,
                               dim=0).numpy()
            policy /= np.sum(policy) # normalize so sum is close enough to 1
        item = np.random.choice(self.model.item_factors.shape[0], p=policy)
        return item
