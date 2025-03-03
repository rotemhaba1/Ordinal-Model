from validation_utils import *
from config.file_paths import *
from src.utils.setup_logger import evaluation_logger

if __name__ == "__main__":
    experiments_to_update=find_experiments_to_update(EXPERIMENT_TRACKING_MIXED_PATH, EXPERIMENT_SUMMARY_MIXED_PATH)
    experiments_valid=evaluate_experiments(experiments_to_update, PREDICT_TRACKING_MIXED_PATH)
    update_experiments_file(experiments_valid,EXPERIMENT_SUMMARY_MIXED_PATH)



