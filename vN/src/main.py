import pickle
from scipy import sparse
from sklearn.model_selection import train_test_split
import implicit
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import precision_at_k, AUC_at_k 
from tqdm import tqdm

def save(filename, *args):
    # Get global dictionary
    glob = globals()
    d = {}
    for v in args:
        # Copy over desired values
        d[v] = glob[v]
    with open(filename, 'wb') as f:
        # Put them in the file 
        pickle.dump(d, f)

def load(filename):
    # Get global dictionary
    glob = globals()
    with open(filename, 'rb') as f:
        for k, v in pickle.load(f).items():
            # Set each global variable to the value from the file
            glob[k] = v

def generate_hyperparameter_configurations(latent_factors, regularization, confidence_weighting):
    configurations = {}
    
    # Initialize with all possible hyperparameter combinations
    i = 0
    for latent_factor in latent_factors:
        for reg in regularization:
            for alpha in confidence_weighting:
                configurations[i] = {"latent_factor": latent_factor, "reg": reg, "alpha": alpha}
                i+=1
                    
    return configurations

def generate_masked_matrix(R, index2coordinates_R, indices_mask, mask):
    # Copy filled array and set mask to zero
    if mask == "zero":
        R_masked = np.copy(R)
    # Create zero array and set mask to value
    else:
        R_masked = np.zeros(R.shape)

    # Loop through indices of mask
    for i in indices_mask:
        # Get mask's coordinates in R
        coordinates = index2coordinates_R[i]

        # According to <mask> mode set r_ui to zero       
        if mask == "zero":
            R_masked[coordinates["r"]][coordinates["c"]] = 0
            continue

        # Set r_ui to value
        R_masked[coordinates["r"]][coordinates["c"]] = R[coordinates["r"]][coordinates["c"]]

    return R_masked

# Create recommendation system using the given "ground truth"
def create_recommendation_system(ground_truth, hyperparameter_configurations, index2coordinates, split={"train": 0.7, "validation": 0.1}):
    R = sparse.csc_matrix(ground_truth)
    # print(R)

    # Create data split 
    train, validation_test = train_test_split(np.array(list(index2coordinates.keys())), train_size=split["train"])
    validation, test = train_test_split(validation_test, train_size=split["validation"]/(1 - split["train"]))
    
    R_train = generate_masked_matrix(ground_truth, index2coordinates, validation_test, mask="zero")
    R_train_csr = sparse.csr_matrix(R_train).astype(np.float64)
    print(R_train)
    is_nan = np.any(np.isnan(self.user_factors), axis=None)
    # print(R_train)

    R_validation = generate_masked_matrix(ground_truth, index2coordinates, validation, mask="value")
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
        performance[precision] = {"i_params": i, "p_val": precision, "params": params}

    print(performance)
    
    return

if __name__ == "__main__":
    # Folder location of variables that can be (re)loaded (in another script)
    VARIABLE_FILES_LOCATION = "variables/"

    # Generate ground truth of artist preferences
    filename = 'artist_recommendation.py'
    exec(compile(open(filename, "rb").read(), filename, 'exec'))

    # Load ground truth
    load(VARIABLE_FILES_LOCATION + "ground_truth_fm")
    print("Ground_truth_fm:\n", ground_truth_fm)
