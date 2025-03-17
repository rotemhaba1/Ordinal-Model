from src.validation.validation_utils import *
from config.file_paths import *
from config.hyperparams import param_ensemble
from src.utils.setup_logger import evaluation_logger
from src.preprocessing.save_processed_data import patient_info

def run_pipeline_validation(experiment_types=['mixed', 'independent']):
    Patients, Patients_level_3 = patient_info()
    for experiment_type in experiment_types:
        if experiment_type == "mixed":
            experiments_to_update = find_experiments_to_update(EXPERIMENT_TRACKING_MIXED_PATH,
                                                               EXPERIMENT_SUMMARY_MIXED_PATH, param_ensemble)
            experiments_valid = evaluate_experiments(experiments_to_update, PREDICT_TRACKING_MIXED_PATH)
            update_experiments_file(experiments_valid, EXPERIMENT_SUMMARY_MIXED_PATH)
            summary_results_mixed(RESULTS_DIR, EXPERIMENT_SUMMARY_MIXED_PATH)
        elif experiment_type == "independent":
            experiments_to_update = find_experiments_to_update(EXPERIMENT_TRACKING_INDEPENDENT_PATH,
                                                               EXPERIMENT_SUMMARY_INDEPENDENT_PATH, param_ensemble)
            experiments_valid = evaluate_experiments(experiments_to_update, PREDICT_TRACKING_INDEPENDENT_PATH,
                                                     Patients_level_3)
            update_experiments_file(experiments_valid, EXPERIMENT_SUMMARY_INDEPENDENT_PATH)
            summary_results_independent(RESULTS_DIR, EXPERIMENT_SUMMARY_INDEPENDENT_PATH)








