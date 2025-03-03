from src.utils.setup_logger import training_logger
from src.training.models import predict_models
from src.training.train_utils import *
import pandas as pd
import os
import time
from itertools import product
from config.hyperparams import param_grids
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


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

def train(experiment_type,params):
    experiment_tracking_path=get_experiment_tracking_path(experiment_type)
    experiment_id = get_index(params, experiment_type, experiment_tracking_path)
    training_logger.info(f"ID-{experiment_id} --Algo Hyperparameters: {params['model']}, {params['combo']}")
    start_time = time.time()
    if experiment_type == "mixed":
        train_experiment_mixed(params,experiment_id,get_predict_tracking_path(experiment_type))
        end_time = time.time()
        training_logger.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        save_index(experiment_id, params, experiment_type, experiment_tracking_path)
        training_logger.info(f"Model saved successfully: {experiment_tracking_path}")
    else:
        raise ValueError("Invalid methodology selected")



if __name__ == "__main__":
    params = {
        "min_diff": 1.5,
        "max_diff": 9,
        "min_length": 1.5,
        "max_length": 8,
        "remove_level": ["Inhalation"],
        'downsampling': False,
        'smote': True,
    }


    experiment_type="mixed"
    param_combinations = get_param_combinations()
    training_logger.info("Training started.")
    training_logger.info(f"experiment_type: {experiment_type}")
    training_logger.info(f"Model Hyperparameters: {params}")
    for model, combinations in param_combinations.items():
        for combo in combinations:
            params['model']=model
            params['combo'] = combo
            train(experiment_type,params)