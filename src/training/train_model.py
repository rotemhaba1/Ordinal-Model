from src.utils.setup_logger import training_logger
from src.training.models import predict_models
from src.training.train_utils import *
import pandas as pd
import os
import time
from itertools import product
from config.hyperparams import param_grids,params
import warnings
from src.preprocessing.save_processed_data import patient_info
warnings.simplefilter("ignore", category=FutureWarning)
import multiprocessing


"""
from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier(WIGR_power=1)
"""

def get_param_combinations():
    all_combinations = {}

    for model, params in param_grids.items():
        if params:  # If there are parameters to iterate over
            keys, values = zip(*params.items())
            combinations = [dict(zip(keys, v)) for v in product(*values)]
        else:  # If the parameter grid is empty, use an empty dictionary (default parameters)
            combinations = [{}]

        all_combinations[model] = combinations

    return all_combinations

def train_experiment_mixed(params,experiment_id,path):
    X,Y,split_train_test = load_data_experiment_mixed(params)
    cv_probabilities = {}
    for cv_i in ['cv_1'  , 'cv_2',   'cv_3' ,  'cv_4',   'cv_5']:
        #training_logger.info(f"Start : {cv_i}/cv_5")
        train_indices = split_train_test[split_train_test[cv_i] == True].index
        test_indices = split_train_test[split_train_test[cv_i] == False].index
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
        X_train, Y_train = preprocess_data(X_train, Y_train,params)
        clf = predict_models(params)
        clf = clf.fit(X_train, Y_train["level_int"])
        X_test = X_test.drop(columns=["Patient_NO",'Respiratory cycle'])
        y_predict_proba = clf.predict_proba(X_test)
        y_predict_proba_df = pd.DataFrame(y_predict_proba, index=Y_test.index,
                                          columns=["prob_class_1", "prob_class_2", "prob_class_3"])

        Y_test = pd.concat([Y_test, y_predict_proba_df], axis=1)
        cv_probabilities[cv_i] = Y_test

    save_path = os.path.join(path, f"cv_probabilities_{experiment_id}.parquet")

    cv_results_df = pd.concat(cv_probabilities, names=["cv_fold"])
    cv_results_df.to_parquet(save_path, engine="pyarrow")

def train_experiment_independent(params,experiment_id,path,Patients_level_3):
    for p_id in Patients_level_3:
        X,Y,split_train_test = load_data_experiment_independent(p_id)
        cv_probabilities = {}
        #training_logger.info(f" Start :P_{p_id}")
        for cv_i in ['cv_1'  , 'cv_2',   'cv_3' ,  'cv_4',   'cv_5']:
            train_indices = split_train_test[split_train_test[cv_i] == True].index
            test_indices = split_train_test[split_train_test[cv_i] == False].index
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
            X_train, Y_train = preprocess_data(X_train, Y_train,params)
            clf = predict_models(params)
            clf = clf.fit(X_train, Y_train["level_int"])
            X_test = X_test.drop(
                columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X_test.columns])
            y_predict_proba = clf.predict_proba(X_test)
            num_classes = y_predict_proba.shape[1]
            column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else ['prob_class_1','prob_class_3']
            y_predict_proba_df = pd.DataFrame(y_predict_proba, index=Y_test.index, columns=column_names)

            Y_test = pd.concat([Y_test, y_predict_proba_df], axis=1)
            cv_probabilities[cv_i] = Y_test

        save_path = os.path.join(path, f"cv_probabilities_P{p_id}_{experiment_id}.parquet")

        cv_results_df = pd.concat(cv_probabilities, names=["cv_fold"])
        cv_results_df.to_parquet(save_path, engine="pyarrow")

def train(experiment_type,params,Patients_level_3,retrain=True):
    experiment_tracking_path=get_experiment_tracking_path(experiment_type)
    experiment_id,exists  = get_index(params, experiment_type, experiment_tracking_path)
    if (retrain==False) & (exists==True):
        training_logger.info(
            f"SKIP Experiment: {experiment_type.upper():<12} | Experiment ID: {experiment_id:<2} | "
            f"Model: {params['model']:<21} | Hyperparameters: {params['combo']}"
        )
        return ''

    training_logger.info(
        f"Experiment: {experiment_type.upper():<12} | Experiment ID: {experiment_id:<2} | "
        f"Model: {params['model']:<21} | Hyperparameters: {params['combo']}"
    )

    start_time = time.time()
    if experiment_type == "mixed":
        train_experiment_mixed(params,experiment_id,get_predict_tracking_path(experiment_type))
    elif experiment_type == "independent":
        train_experiment_independent(params, experiment_id, get_predict_tracking_path(experiment_type),Patients_level_3)
    else:
        raise ValueError("Invalid methodology selected")

    end_time = time.time()
    training_logger.debug(f"Training completed in {end_time - start_time:.2f} seconds.")
    save_index(experiment_id, params, experiment_type, experiment_tracking_path)
    training_logger.debug(f"Model saved successfully: {experiment_tracking_path}")


def run_experiment(experiment_type, params,Patients, Patients_level_3,param_combinations,retrain):

    training_logger.info(f"Training started for experiment_type: {experiment_type}")
    training_logger.info(f"Model Hyperparameters: {params}")

    for model, combinations in param_combinations.items():
        for combo in combinations:
            params['model'] = model
            params['combo'] = combo
            train(experiment_type, params, Patients_level_3,retrain)


def run_in_parallel():
    experiment_types = ['mixed', 'independent']
    retrain=False
    Patients, Patients_level_3 = patient_info()
    param_combinations = get_param_combinations()
    with multiprocessing.Pool(processes=len(experiment_types)) as pool:
        pool.starmap(run_experiment, [(experiment_type, params,Patients, Patients_level_3,param_combinations,retrain) for experiment_type in experiment_types])


def run_in_sequence():
    experiment_types = ['independent', 'mixed']
    retrain = False
    Patients, Patients_level_3 = patient_info()
    param_combinations = get_param_combinations()

    for experiment_type in experiment_types:
        run_experiment(experiment_type, params,Patients, Patients_level_3,param_combinations,retrain)


if __name__ == "__main__":
    # run_in_parallel()
    run_in_sequence()

