import os
import pandas as pd
from config.file_paths import *
import  numpy as np
from src.utils.smote import synthetic_oversample,random_oversample
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

def load_data_experiment_mixed_smote(X,Y,params,cv_i):

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}_{cv_i}_smote.parquet"
    )

    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}_{cv_i}_smote.parquet"
    )

    if os.path.exists(eeg_df_path) and os.path.exists(target_path):
        X = pd.read_parquet(eeg_df_path)
        Y = pd.read_parquet(target_path)
        return X, Y

    X, Y = synthetic_oversample(X, Y)
    X.to_parquet(eeg_df_path, index=False)
    Y.to_parquet(target_path, index=False)
    return X, Y

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

def load_data_experiment_independent_smote(X, Y, params, cv_i, p_id):
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"Respiratory_cycle_df_P_{p_id}_{cv_i}_smote.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"fft_data_model_STFT_P_{p_id}_{cv_i}_smote.parquet"
    )

    if os.path.exists(eeg_df_path) and os.path.exists(target_path):
        X = pd.read_parquet(eeg_df_path)
        Y = pd.read_parquet(target_path)
        return X, Y

    X, Y = synthetic_oversample(X, Y)
    X.to_parquet(eeg_df_path, index=False)
    Y.to_parquet(target_path, index=False)
    return X, Y

def load_data_experiment_probabilistic_step_1(params):
    # Define file paths
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic.parquet"
    )

    # Read parquet files
    Y = pd.read_parquet(target_path)
    X = pd.read_parquet(eeg_df_path)



    return X,Y

def load_data_experiment_probabilistic_step_1_smote(X, Y, params, p_id_second_test,p_id):
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic_{p_id_second_test}_{p_id}_smote.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic_{p_id_second_test}_{p_id}_smote.parquet"
    )

    if os.path.exists(eeg_df_path) and os.path.exists(target_path):
        X = pd.read_parquet(eeg_df_path)
        Y = pd.read_parquet(target_path)
        return X, Y

    X, Y = synthetic_oversample(X, Y)
    X.to_parquet(eeg_df_path, index=False)
    Y.to_parquet(target_path, index=False)
    return X, Y

def load_data_experiment_probabilistic_step_1_test_smote(X, Y, params,p_id):
    target_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"target_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic_{p_id}_smote.parquet"
    )

    eeg_df_path = os.path.join(
        PROCESSED_DATA_DIR,
        f"EEG_df_min_diff{params['min_diff']}max_diff{params['max_diff']}"
        f"min_length{params['min_length']}max_length{params['max_length']}"
        f"remove_level_{params['remove_level'][0]}probabilistic_{p_id}_smote.parquet"
    )

    if os.path.exists(eeg_df_path) and os.path.exists(target_path):
        X = pd.read_parquet(eeg_df_path)
        Y = pd.read_parquet(target_path)
        return X, Y

    X, Y = synthetic_oversample(X, Y)
    X.to_parquet(eeg_df_path, index=False)
    Y.to_parquet(target_path, index=False)
    return X, Y

def add_previous_probabilities(df, previous_number):
    df = df.copy()  # Avoid modifying original DF
    df = df.sort_values(by=['Patient_NO', 'Respiratory cycle']).reset_index(drop=True)
    for t in range(1, previous_number + 1):
        for col in ['prob_class_1', 'prob_class_2', 'prob_class_3']:
            df[f'{col}_t{t}'] = df.groupby('Patient_NO')[col].shift(t)  # Shift past values
    df = df.dropna(subset=[f'{col}_t{t}']).reset_index(drop=True)

    return df

def load_data_experiment_probabilistic_step_2_smote(X, Y):
    X, Y = synthetic_oversample(X, Y)
    return X, Y

def load_data_experiment_probabilistic_step_2(path,p_id,experiment_id):
    # Define file paths
    probabilistic_step1_path = os.path.join(path, f"probabilities_step_1_P{p_id}_{experiment_id}.parquet")
    all_data = pd.read_parquet(probabilistic_step1_path)
    all_data = all_data.reset_index(drop=True)
    all_data=add_previous_probabilities(all_data, previous_number=9)
    all_data = all_data.rename(columns=lambda col: col.replace("prob_class_", "prob_train_class_") if "prob_class_" in col else col)

    train = all_data[all_data["Patient_NO"] != f"P_{p_id}"]
    test = all_data[all_data["Patient_NO"] == f"P_{p_id}"]

    return train,test

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


def get_index_step_2(params, experiment, excel_path):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)

    params_str = str(params)

    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
    else:

        df = pd.DataFrame(columns=['index', 'index_step1', 'index_step2', 'params_step1', 'params_step2','experiment', 'timestamp'])

    existing_run = df[
        (df["params_step2"] == params_str) &
        (df["experiment"] == experiment)
        ]

    if not existing_run.empty:
        return existing_run["index"].values[0], True

    new_index = df["index_step2"].max() + 1 if not df.empty else 1

    return new_index, False

def get_experiment_step_1(excel_path):
    return  pd.read_excel(excel_path.replace('_step_2', ""))

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

def save_index_step_2(index_step1,index_step2,params_step1, params_step2, experiment, experiment_tracking_path):
    params_step1= str(params_step1)
    params_step2 = str(params_step2)



    new_run = pd.DataFrame([{
        "index": f"{index_step1}_{index_step2}",
        "index_step1": index_step1,
        "index_step2": index_step2,
        "params_step1": params_step1,
        "params_step2":params_step2,
        "experiment": experiment,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }])

    if os.path.exists(experiment_tracking_path):
        df = pd.read_excel(experiment_tracking_path)
    else:
        df = pd.DataFrame(columns=["index","index_step1","index_step2", "params_step1", "params_step2", "experiment", "timestamp"])

    df = pd.concat([df, new_run], ignore_index=True)
    df = df.sort_values(by='timestamp', ascending=False)
    df = df.drop_duplicates(subset='index', keep='first')
    df = df.reset_index(drop=True)
    df.to_excel(experiment_tracking_path, index=False)