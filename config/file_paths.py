import os

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to raw and processed data
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# Paths for test train splits
SPLITS_DATA_DIR = os.path.join(BASE_DIR, "data", "splits")


# Paths for metadata and results
META_DATA_DIR = os.path.join(BASE_DIR, "data", "meta")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# EXPERIMENT_TRACKING
PREDICT_INDEPENDENT_PATH = os.path.join(RESULTS_DIR, "independent", "predictions")
PREDICT_TRACKING_MIXED_PATH = os.path.join(RESULTS_DIR, "mixed", "predictions")
PREDICT_TRACKING_PROBABILISTIC_PATH = os.path.join(RESULTS_DIR, "probabilistic", "predictions")

# Paths for experiment tracking
EXPERIMENT_TRACKING_INDEPENDENT_PATH = os.path.join(RESULTS_DIR,'independent', "experiment_tracking_independent.xlsx")
EXPERIMENT_TRACKING_MIXED_PATH = os.path.join(RESULTS_DIR,'mixed', "experiment_tracking_mixed.xlsx")
EXPERIMENT_TRACKING_PROBABILISTIC_PATH = os.path.join(RESULTS_DIR,'probabilistic', "experiment_tracking_probabilistic.xlsx")



def get_experiment_tracking_path(experiment_type):
    if experiment_type == "independent":
        return EXPERIMENT_TRACKING_INDEPENDENT_PATH
    elif experiment_type == "mixed":
        return EXPERIMENT_TRACKING_MIXED_PATH
    elif experiment_type == "probabilistic":
        return EXPERIMENT_TRACKING_PROBABILISTIC_PATH

def get_predict_tracking_path(experiment_type):
    if experiment_type == "independent":
        return PREDICT_INDEPENDENT_PATH
    elif experiment_type == "mixed":
        return PREDICT_TRACKING_MIXED_PATH
    elif experiment_type == "probabilistic":
        return PREDICT_TRACKING_PROBABILISTIC_PATH

def get_patient_raw_path(patient):
    return os.path.join(RAW_DATA_DIR, patient)

# Example usage
if __name__ == "__main__":
    print("Base Directory:", BASE_DIR)
    print("Raw Data Path:", RAW_DATA_DIR)
    print("Processed Data Path:", PROCESSED_DATA_DIR)
    print("Metadata Path:", META_DATA_DIR)
    print("Results Path:", RESULTS_DIR)