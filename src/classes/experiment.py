import numpy as np
from tqdm import tqdm
import time

#1st party imports
from classes.audit import Audit
from lib import io

class Experiment(object):
    def __init__(self, log_path, experiment_choices, recommenders):
        self.log_path = log_path
        self.experiment_free_pass = True if experiment_choices == "all" else False
        self.experiment_choices = experiment_choices
        self.experiment_results = {experiment: {} for experiment in experiment_choices}

        self.recommenders = recommenders
        # Tracks the audit per recommender model (mapping indices)
        self.audits = {i: None for i in range(len(recommenders))}

    def exp_5_1(self):
        info = self.recommenders[0].data_set["label"] + " " + self.recommenders[0].model_type
        print("Performing experiment 5.1...{}".format(info))
        avg_envy_users = {}
        prop_envious_users = {}
        
        start = time.time()
        for i in tqdm(range(len(self.recommenders))):
            recommender = self.recommenders[i]
            audit = Audit(recommender)
            envy_per_user = audit.get_envy()

            # Average envy per user 
            latent_factor = recommender.params["latent_factor"]
            avg_envy_users[latent_factor] = np.mean(envy_per_user)

            # Proportion envious users
            epsilon = audit.params["basic"]["epsilon"]
            prop_envious_users[latent_factor] = np.mean(envy_per_user > epsilon)

            self.audits[i] = audit
            
        end = time.time() - start
        io.write_to_file(self.log_path, "Performing experiment 5.1...{} {}".format(info, str(end)))

        # Sort in ascending order of <latent_factor>
        avg_envy_users = dict(sorted(avg_envy_users.items()))
        prop_envious_users = dict(sorted(prop_envious_users.items())) 

        self.experiment_results["5.1"]["avg_envy_users"] = avg_envy_users
        self.experiment_results["5.1"]["prop_envious_users"] = prop_envious_users



       