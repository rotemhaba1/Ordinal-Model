import os
import pandas as pd
from config.file_paths import *
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from datetime import datetime


def load_data_experiment_mixed(params):
    Y = pd.read_parquet(f"{PROCESSED_DATA_DIR}/target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
    f"min_length{params['min_length']}max_length{params['max_length']}"
    f"remove_level_{params['remove_level'][0]}.parquet")

    X = pd.read_parquet(f"{PROCESSED_DATA_DIR}/EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
    f"min_length{params['min_length']}max_length{params['max_length']}"
    f"remove_level_{params['remove_level'][0]}.parquet")

    split_train_test = pd.read_parquet(f"{SPLITS_DATA_DIR}/split_train_test_min_diff{params['min_diff']}max_diff{params['max_diff']}"
    f"min_length{params['min_length']}max_length{params['max_length']}"
    f"remove_level_{params['remove_level'][0]}.parquet")


    return X,Y,split_train_test



def preprocess_data(X_train, Y_train, algorithm_params):
    level_mapping = {
        'FEV1 [-10,inf)': 0,
        'FEV1 [-20,-10)': 1,
        'FEV1 [-inf,-20)': 2
    }

    Y_train["level_int"] = Y_train["level"].map(level_mapping)
    X_train = X_train.drop(columns=["Patient_NO",'Respiratory cycle'])

    if algorithm_params['downsampling']:
        rus = RandomUnderSampler(random_state=42)
        X_train, Y_train_level = rus.fit_resample(X_train, Y_train["level_int"])
        Y_train = Y_train.loc[Y_train.index.isin(X_train.index)]
        Y_train["level_int"] = Y_train_level

    if algorithm_params['smote']:
        smote = SMOTE(random_state=42)
        X_train, Y_train_level = smote.fit_resample(X_train, Y_train["level_int"])
        Y_train = pd.DataFrame({"level_int": Y_train_level})



    return X_train, Y_train

def get_index(params, algorithm_params, experiment, excel_path):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    params_str = str(params)
    algorithm_params_str = str(algorithm_params)

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=["index", "params", "algorithm_params", "experiment", "timestamp"])

    existing_run = df[
        (df["params"] == params_str) &
        (df["algorithm_params"] == algorithm_params_str) &
        (df["experiment"] == experiment)
        ]

    if not existing_run.empty:
        return existing_run["index"].values[0]

    new_index = df["index"].max() + 1 if not df.empty else 1

    return new_index


def save_index(new_index,params, algorithm_params, experiment, experiment_tracking_path):
    params_str = str(params)
    algorithm_params_str = str(algorithm_params)

    new_run = pd.DataFrame([{
        "index": new_index,
        "params": params_str,
        "algorithm_params": algorithm_params_str,
        "experiment": experiment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])
    df = pd.read_excel(experiment_tracking_path)

    df = pd.concat([df, new_run], ignore_index=True)
    df.to_excel(experiment_tracking_path, index=False)