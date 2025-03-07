import os
import pandas as pd
from config.file_paths import *
import  numpy as np

from datetime import datetime


def load_data_experiment_mixed(params):
    # Define file paths
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}.parquet"
    )

    split_train_test_path = os.path.join(
        SPLITS_DATA_DIR,
        f"split_train_test_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}.parquet"
    )

    # Read parquet files
    Y = pd.read_parquet(target_path)
    X = pd.read_parquet(eeg_df_path)
    split_train_test = pd.read_parquet(split_train_test_path)


    return X,Y,split_train_test

def load_data_experiment_independent(p_id):
    # Define file paths
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"Respiratory_cycle_df_P_{p_id}.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"fft_data_model_STFT_P_{p_id}.parquet"
    )

    split_train_test_path = os.path.join(
        SPLITS_DATA_DIR,
        f"split_train_test_P_{p_id}.parquet"
    )

    # Read parquet files
    Y = pd.read_parquet(target_path)
    X = pd.read_parquet(eeg_df_path)
    split_train_test = pd.read_parquet(split_train_test_path)


    return X,Y,split_train_test

def preprocess_data(X_train, Y_train, params):
    level_mapping = {
        'FEV1 [-10,inf)': 0,
        'FEV1 [-20,-10)': 1,
        'FEV1 [-inf,-20)': 2
    }

    Y_train["level_int"] = Y_train["level"].map(level_mapping)
    unique_classes = np.sort(Y_train["level_int"].unique())
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(unique_classes)}
    Y_train["level_int"] = Y_train["level_int"].map(class_mapping)

    X_train = X_train.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X_train.columns])

    if params['downsampling']:
        """
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_train, Y_train_level = rus.fit_resample(X_train, Y_train["level_int"])
        Y_train = Y_train.loc[Y_train.index.isin(X_train.index)]
        Y_train["level_int"] = Y_train_level
        """


    if params['smote']:
        """
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train, Y_train_level = smote.fit_resample(X_train, Y_train["level_int"])
        Y_train = pd.DataFrame({"level_int": Y_train_level})
        """
        from sklearn.utils import resample
        X_train, Y_train_level = resample(X_train, Y_train["level_int"],
                                                            replace=True,
                                                            n_samples=len(X_train),
                                                            random_state=42)

        Y_train = pd.DataFrame({"level_int": Y_train_level})



    return X_train, Y_train

def get_index(params, experiment, excel_path):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    params_str = str(params)

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:
        df = pd.DataFrame(columns=["index", "params", "experiment", "timestamp"])

    existing_run = df[
        (df["params"] == params_str) &
        (df["experiment"] == experiment)
        ]

    if not existing_run.empty:
        return existing_run["index"].values[0],True

    new_index = df["index"].max() + 1 if not df.empty else 1

    return new_index,False


def save_index(new_index, params, experiment, experiment_tracking_path):
    params_str = str(params)


    new_run = pd.DataFrame([{
        "index": new_index,
        "params": params_str,
        "experiment": experiment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(experiment_tracking_path):
        df = pd.read_excel(experiment_tracking_path)
    else:
        df = pd.DataFrame(columns=["index", "params", "experiment", "timestamp"])

    df = pd.concat([df, new_run], ignore_index=True)
    df = df.sort_values(by='timestamp', ascending=False)
    df = df.drop_duplicates(subset='index', keep='first')
    df = df.reset_index(drop=True)
    df.to_excel(experiment_tracking_path, index=False)