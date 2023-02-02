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
        # Utilities of own policies
        u_m = self.recommender.u_m
        # Utilities of OPT
        u_OPT = self.recommender.u_OPT

        # Only keep <u_OPT> that involve policies other than those of the user
        mask = np.full(u_OPT.shape, 1)
        np.fill_diagonal(mask, 0)
        u_n = mask * u_OPT
        delta_u = (u_n.flatten() - u_m.flatten())
        envy = delta_u.clip(min=0)
        
        # Envy for each user
        return envy

    def audit_envy(self, audit_mode="basic"):
        """
        Audit function
        """
        if audit_mode=="basic":
            return self.__envy_free_basic()
