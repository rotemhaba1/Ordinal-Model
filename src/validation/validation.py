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

        elif experiment_type == "affine":
            experiments_to_update = find_experiments_to_update(EXPERIMENT_TRACKING_AFFINE,
                                                               EXPERIMENT_SUMMARY_AFFINE_PATH, param_ensemble)
            if experiments_to_update.__len__()>0:
                experiments_valid,auc_per_fold_df= evaluate_experiments(experiments_to_update, PREDICT_TRACKING_AFFINE_PATH)
                update_experiments_file(experiments_valid,auc_per_fold_df, EXPERIMENT_SUMMARY_AFFINE_PATH)

            summary_results_affine(RESULTS_DIR, EXPERIMENT_SUMMARY_AFFINE_PATH)
            knee_point(RESULTS_DIR)
            t_test_point(RESULTS_DIR,EXPERIMENT_SUMMARY_AFFINE_PATH,on='auc_avg')
            t_test_point(RESULTS_DIR, EXPERIMENT_SUMMARY_AFFINE_PATH,on='auc_weighted')
            patient_auc(RESULTS_DIR,on='auc_avg')
            patient_auc(RESULTS_DIR, on='auc_weighted')

        """ 
        elif experiment_type == "probabilistic_step_2": 
            experiments_to_update = find_experiments_to_update(EXPERIMENT_TRACKING_PROBABILISTIC_STEP_2_PATH,
                                                               EXPERIMENT_SUMMARY_PROBABILISTIC_PATH, param_ensemble)
            experiments_valid = evaluate_experiments_prob(experiments_to_update, PREDICT_TRACKING_PROBABILISTIC_PATH,
                                                     Patients_level_3)
            update_experiments_file(experiments_valid, EXPERIMENT_SUMMARY_INDEPENDENT_PATH)
            summary_results_independent(RESULTS_DIR, EXPERIMENT_SUMMARY_INDEPENDENT_PATH)
        """








