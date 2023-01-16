import numpy as np
import torch
import torch.nn.functional as F
from implicit.als import AlternatingLeastSquares

class Recommender(object):
    def __init__(self, factors=100, regularization=0.01, alpha=1.0,
                 iterations=15, temperature=1, compute_dense_matrix=False):
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
                                             iterations=iterations)
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
            user_item_matrix = self.model.user_factors @ self.model.item_factors.T
            self.user_item_logits = torch.Tensor(user_item_matrix)
            self.user_item_probs = F.softmax(self.user_item_logits / self.temperature,
                                             dim=1)
    
    def recommend(self, user_id, size=1):
        """
        Samples items (without replacement) to a user to recommend using a
        softmax policy.
        Inputs:
            user_id - Index of the user to recommend an item to.
        Outputs:
            items - Array of recommended items.
        """
        if self.dense_matrix:
            item_probs = self.user_item_probs[user_id].numpy()
        else:
            item_logits = self.model.user_factors[user_id] @ self.model.item_factors.T
            item_logits = torch.Tensor(item_logits)
            item_probs = F.softmax(item_logits / self.temperature, dim=0).numpy()
        item_probs /= np.sum(item_probs) # normalize so sum is close enough to 1
        items = np.random.choice(self.model.item_factors.shape[0], size=size,
                                p=item_probs)
        return items
