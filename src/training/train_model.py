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
        training_logger.info(f"Start : {cv_i}/cv_5")
        train_indices = split_train_test[split_train_test[cv_i] == True].index
        test_indices = split_train_test[split_train_test[cv_i] == False].index
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
        X_train, Y_train = preprocess_data(X_train, Y_train,params)
        clf = predict_models(params)
        if params['smote']:
            X_train,Y_train=load_data_experiment_mixed_smote(X_train,Y_train[["level_int"]],params,cv_i)
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
            if params['smote']:
                X_train, Y_train = load_data_experiment_independent_smote(X_train, Y_train[["level_int"]], params, cv_i,p_id)
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

def train_experiment_probabilistic_step_1(params,experiment_id,path,Patients_level_3):

    for p_id in Patients_level_3:
        X, Y = load_data_experiment_probabilistic_step_1(params)
        Y=Y[Y["Patient_NO"] != f"P_{p_id}"]
        X=X.loc[Y.index]
        p_id_second_test_probabilities = {}

        for p_id_second_test in set(Patients_level_3) - {p_id}:
            train_indices = Y[Y["Patient_NO"] != f"P_{p_id_second_test}"].index
            test_indices = Y[Y["Patient_NO"] == f"P_{p_id_second_test}"].index
            X_train, X_test = X.loc[train_indices], X.loc[test_indices]
            Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
            X_train, Y_train = preprocess_data(X_train, Y_train, params)
            clf = predict_models(params)
            if params['smote']:
                X_train, Y_train = load_data_experiment_probabilistic_step_1_smote(X_train, Y_train[["level_int"]], params, p_id_second_test,p_id)
            clf = clf.fit(X_train, Y_train["level_int"])
            X_test = X_test.drop(
                columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X_test.columns])
            y_predict_proba = clf.predict_proba(X_test)
            num_classes = y_predict_proba.shape[1]
            column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else ['prob_class_1',
                                                                                               'prob_class_3']
            y_predict_proba_df = pd.DataFrame(y_predict_proba, index=Y_test.index, columns=column_names)

            Y_test = pd.concat([Y_test, y_predict_proba_df], axis=1)
            p_id_second_test_probabilities[f"P_{p_id_second_test}"] = Y_test

        # test matrix
        X, Y = load_data_experiment_probabilistic_step_1(params)
        train_indices = Y[Y["Patient_NO"] != f"P_{p_id}"].index
        test_indices = Y[Y["Patient_NO"] == f"P_{p_id}"].index
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        Y_train, Y_test = Y.loc[train_indices], Y.loc[test_indices]
        X_train, Y_train = preprocess_data(X_train, Y_train, params)
        clf = predict_models(params)
        if params['smote']:
            X_train, Y_train = load_data_experiment_probabilistic_step_1_test_smote(X_train, Y_train[["level_int"]], params, p_id)
        clf = clf.fit(X_train, Y_train["level_int"])
        X_test = X_test.drop(
            columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X_test.columns])
        y_predict_proba = clf.predict_proba(X_test)
        num_classes = y_predict_proba.shape[1]
        column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else ['prob_class_1',
                                                                                           'prob_class_3']
        y_predict_proba_df = pd.DataFrame(y_predict_proba, index=Y_test.index, columns=column_names)

        Y_test = pd.concat([Y_test, y_predict_proba_df], axis=1)
        p_id_second_test_probabilities[f"P_{p_id}"] = Y_test



        save_path = os.path.join(path, f"probabilities_step_1_P{p_id}_{experiment_id}.parquet")

        cv_results_df = pd.concat(p_id_second_test_probabilities, names=["cv_probabilities_P"])
        cv_results_df.to_parquet(save_path, engine="pyarrow")

def train_experiment_probabilistic_step_2(params,experiment_id_step_1,experiment_id,path_step_1,path,Patients_level_3):
    for p_id in Patients_level_3:
        train,test = load_data_experiment_probabilistic_step_2(path_step_1,p_id,experiment_id_step_1)

        X_train, X_test = train.filter(like="prob_train_class_"), test.filter(like="prob_train_class_")
        Y_train, Y_test = train,test
        X_train, Y_train = preprocess_data(X_train, Y_train, params)
        clf = predict_models(params)
        if params['smote']:
            X_train, Y_train = load_data_experiment_probabilistic_step_2_smote(X_train, Y_train[["level_int"]])
        clf = clf.fit(X_train, Y_train["level_int"])
        y_predict_proba = clf.predict_proba(X_test)
        num_classes = y_predict_proba.shape[1]
        column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else ['prob_class_1',
                                                                                           'prob_class_3']
        y_predict_proba_df = pd.DataFrame(y_predict_proba, index=Y_test.index, columns=column_names)

        Y_test = pd.concat([Y_test, y_predict_proba_df], axis=1)
        save_path = os.path.join(path_step_1, f"probabilities_step_2_P{p_id}_{experiment_id_step_1}_{experiment_id}.parquet")
        Y_test.to_parquet(save_path, engine="pyarrow")


def train(experiment_type,params,Patients_level_3,retrain=True):
    experiment_tracking_path=get_experiment_tracking_path(experiment_type)
    if experiment_type=='probabilistic_step_2':
        experiment_id, exists = get_index_step_2(params, experiment_type, experiment_tracking_path)
        experiment_step_1 = get_experiment_step_1(experiment_tracking_path)
    else:
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
    elif experiment_type == "probabilistic":
        train_experiment_probabilistic_step_1(params, experiment_id, get_predict_tracking_path(experiment_type),Patients_level_3)
    elif experiment_type == "probabilistic_step_2":
        for index, row in experiment_step_1.iterrows():
            train_experiment_probabilistic_step_2(params, row['index'],experiment_id, get_predict_tracking_path("probabilistic") ,get_predict_tracking_path(experiment_type),Patients_level_3)
            end_time = time.time()
            training_logger.debug(f"Training completed in {end_time - start_time:.2f} seconds.")
            save_index_step_2(row['index'],experiment_id,row['params'], params, experiment_type, experiment_tracking_path)
            training_logger.debug(f"Model saved successfully: {experiment_tracking_path}")
        return
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
            try:
                train(experiment_type, params, Patients_level_3, retrain)
            except Exception as e:
                training_logger.error(f"Error in training with model={model}, combo={combo}: {str(e)}", exc_info=True)


def run_in_parallel(experiment_types=['mixed', 'independent']):
    retrain=False
    Patients, Patients_level_3 = patient_info()
    #Patients= [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 46, 47, 49, 50, 51]
    #Patients_level_3=[1, 5, 6, 7, 10, 17, 18, 32, 47, 51]
    param_combinations = get_param_combinations()
    with multiprocessing.Pool(processes=len(experiment_types)) as pool:
        pool.starmap(run_experiment, [(experiment_type, params,Patients, Patients_level_3,param_combinations,retrain) for experiment_type in experiment_types])


def run_in_sequence(experiment_types=['mixed', 'independent','probabilistic']):
    retrain = False
    Patients, Patients_level_3 = patient_info()
    param_combinations = get_param_combinations()

    for experiment_type in experiment_types:
        run_experiment(experiment_type, params,Patients, Patients_level_3,param_combinations,retrain)




