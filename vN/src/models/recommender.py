import numpy as np
import torch
import torch.nn.functional as F

from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k

class Recommender(object):
    def __init__(self, model, params, ground_truth, preference_est, pred_perf, pred_metric, recommendations, policies):
        """
        Implements an alternating least squares model as a recommender system.
        Inputs:
            
        """
        self.model = model
        self.params = params
        self.ground_truth = ground_truth
        self.preference_est = preference_est
        self.pred_perf = pred_perf
        self.pred_metric = pred_metric
        self.recommendations = recommendations
        self.policies = policies
    
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

def grid_search(train_matrix, val_matrix, factors, regularization,
                confidence_weights, temperature=1, seed=None):
    """
    Performs grid search over the hyperparameters for a recommender using the
    precision_at_k metric.
    Inputs:
        train_matrix - Sparse user-item matrix of training data
        val_matrix - Sparse user-item matrix of validation data
        factors - List of number of latent factors to test
        regularization - List regularization factors to test
        confidence_weights - List of confidence weight values to test
    Outputs:
        results - Matrix where entry (i,j,k) gives the performance for number
                  of factors i, regularization factor j and confidence weight k
        best_rec - Recommender model with hyperparameter setting that performed
                   best on the validation data
    """
    results = np.zeros((len(factors), len(regularization), len(confidence_weights)))
    best_rec = None
    best_performance = 0
    for fi, ri, ci in np.ndindex(results.shape):
        rec = Recommender(factors=factors[fi], regularization=regularization[ri],
                          alpha=confidence_weights[ci], temperature=temperature,
                          compute_dense_matrix=True, random_state=seed)
        rec.fit_model(train_matrix)
        performance = precision_at_k(rec.model, train_matrix, val_matrix,
                                     show_progress=False)
        if performance > best_performance:
            best_performance = performance
            best_rec = rec
        results[fi, ri, ci] = performance
    return results, best_rec