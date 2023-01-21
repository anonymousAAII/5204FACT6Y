from scipy import sparse
from sklearn.model_selection import train_test_split
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, AUC_at_k 
from tqdm import tqdm
import os

# 1st party imports
from lib import io
import constant

# Create recommendation system using the given "ground truth"
def create_recommendation_system(ground_truth, hyperparameter_configurations, ground_truth_coordinates, split={"train": 0.7, "validation": 0.1}):
    R = sparse.csc_matrix(ground_truth)
    # print(R)

    # Create data split 
    train, validation_test = train_test_split(np.array(list(ground_truth_coordinates.keys())), train_size=split["train"])
    validation, test = train_test_split(validation_test, train_size=split["validation"]/(1 - split["train"]))
    
    R_train = generate_masked_matrix(ground_truth, ground_truth_coordinates, validation_test, mask_mode="zero")
    R_train_csr = sparse.csr_matrix(R_train).astype(np.float64)
    # print(R_train)

    R_validation = generate_masked_matrix(ground_truth, ground_truth_coordinates, validation, mask_mode="value")
    R_validation_csr = sparse.csr_matrix(R_validation).astype(np.float64)
    # print(R_validation)
    
    performance = {}

    # Train low-rank matrix completion algorithm (Bell and Sejnowski 1995)
    for i in tqdm(range(len(hyperparameter_configurations))):
        params = hyperparameter_configurations[i]
 
        # Create model
        model = AlternatingLeastSquares(factors=16,#params["latent_factor"],
                                        regularization=params["reg"],
                                        alpha=params["alpha"])

        # Train model
        model.fit(R_train_csr)

        # Validate model
        precision = AUC_at_k(model, R_train_csr, R_validation_csr, K=1000, show_progress=False, num_threads=4)
        performance[precision] = {"i_params": i, "params": params, "p_val": precision}

    print(performance)
    
    return

if __name__ == "__main__":
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Generate ground truth of artist preferences
    filename = "user_artist.py"
    exec(compile(open(filename, "rb").read(), filename, "exec"))

    # Load ground truth
    io.load("ground_truth_fm", globals())
    print(ground_truth_fm)

    