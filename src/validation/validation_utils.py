import pandas as pd
import os
from src.utils.setup_logger import evaluation_logger
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score
import numpy as np
import ast
from datetime import datetime

def filter_params(row, model_x, combo_x):
    try:
        params_dict = ast.literal_eval(row['params'])  # Convert string to dictionary
        return params_dict.get('model') == model_x and params_dict.get('combo') == combo_x
    except (ValueError, SyntaxError):
        return False

def add_ensemble(param_ensemble,tracking_df):
    index_ensemble=[]
    for i in param_ensemble:
        filtered_rows = tracking_df[tracking_df.apply(lambda row: filter_params(row, param_ensemble[i]['model'], param_ensemble[i]['combo']), axis=1)]
        if len(filtered_rows)==1:
            index_ensemble.append(filtered_rows['index'].iloc[0])
        else:
            evaluation_logger.warning(f"Missing / duplicate index_ensemble for {param_ensemble[i]}")
            return tracking_df

    ensemble_row =  pd.DataFrame([{'index': 'ensemble', 'params': index_ensemble, 'experiment': 'mixed',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}])
    return pd.concat([tracking_df, ensemble_row], ignore_index=True)

def find_experiments_to_update(tracking_path, summary_path,param_ensemble):
    if os.path.exists(tracking_path):
        tracking_df = pd.read_excel(tracking_path)
        tracking_df=add_ensemble(param_ensemble, tracking_df)
    else:
        evaluation_logger.error(f"Tracking file not found: {tracking_path}")
        return pd.DataFrame(columns=["experiment_id"])

    if not os.path.exists(summary_path):
        evaluation_logger.info("Summary file not found, all experiments need to be updated.")
        return tracking_df

    summary_df = pd.read_excel(summary_path)

    missing_cols = [col for col in tracking_df.columns if col not in summary_df.columns]
    if missing_cols:
        summary_df = summary_df.reindex(columns=summary_df.columns.tolist() + missing_cols, fill_value=None)

    missing_experiments = tracking_df[~tracking_df["index"].astype(str).isin(summary_df["index"].astype(str))]
    summary_df=summary_df[summary_df.columns.intersection(tracking_df.columns)]

    merged_df = summary_df.merge(tracking_df[['index', 'timestamp']], on='index', suffixes=('_summary', '_tracking'))
    diff_date_df = merged_df[merged_df['timestamp_summary'] != merged_df['timestamp_tracking']]
    diff_date_df = diff_date_df.rename(columns={'timestamp_summary': 'timestamp'})
    diff_date_df = diff_date_df[summary_df.columns]
    experiments_to_update=pd.concat([missing_experiments,diff_date_df])


    evaluation_logger.info(f"Experiments to update: {len(experiments_to_update)}")

    return experiments_to_update

def maps_levels(df):
    level_mapping = {
        'FEV1 [-10,inf)': 1,
        'FEV1 [-20,-10)': 2,
        'FEV1 [-inf,-20)': 3
    }
    df["level_int"] = df["level"].map(level_mapping)
    return df

def predict_ensemble(experiment_params,predict_dir):
    df_prob_1=pd.DataFrame()
    df_prob_2 = pd.DataFrame()
    df_prob_3 = pd.DataFrame()

    for run_number, ex_id in enumerate(experiment_params, start=1):
        prediction_file = os.path.join(predict_dir, f"cv_probabilities_{ex_id}.parquet")
        df = pd.read_parquet(prediction_file)
        df_prob_1[run_number] = df['prob_class_1']
        df_prob_2[run_number] = df['prob_class_2']
        df_prob_3[run_number] = df['prob_class_3']

    df_prob_1['avg_all_columns'] = df_prob_1.mean(axis=1)
    df_prob_2['avg_all_columns'] = df_prob_2.mean(axis=1)
    df_prob_3['avg_all_columns'] = df_prob_3.mean(axis=1)
    df['prob_class_1'] = df_prob_1['avg_all_columns']
    df['prob_class_2'] = df_prob_2['avg_all_columns']
    df['prob_class_3'] = df_prob_3['avg_all_columns']

    return df



def evaluate_experiments(experiments_to_update, predict_dir):
    results = []

    for _, experiment in experiments_to_update.iterrows():
        experiment_id = experiment["index"]
        prediction_file = os.path.join(predict_dir, f"cv_probabilities_{experiment_id}.parquet")


        if (not os.path.exists(prediction_file)) & (experiment_id!='ensemble'):
            evaluation_logger.warning(f"Prediction file missing: {prediction_file}")
            continue

        try:
            if experiment_id!='ensemble':
                df = pd.read_parquet(prediction_file)
                model_name= ast.literal_eval(experiment["params"])['model']
            else:
                if isinstance(experiment['params'], str):
                    experiment_params = ast.literal_eval(experiment['params'])
                else:
                    experiment_params = experiment['params']
                df = predict_ensemble(experiment_params,predict_dir)
                model_name=experiment_id

            maps_levels(df)

            metrics_scores = {
                "auc": {1: [], 2: [], 3: []},
                "mse": {1: [], 2: [], 3: []},
                "accuracy": {1: [], 2: [], 3: []},
                "f1": {1: [], 2: [], 3: []},
                "sensitivity": {1: [], 2: [], 3: []}
            }
            num_samples = {1: 0, 2: 0, 3: 0}

            auc_avg_scores, auc_weighted_scores = [], []
            mse_avg_scores, mse_weighted_scores = [], []
            accuracy_avg_scores, accuracy_weighted_scores = [], []
            f1_avg_scores, f1_weighted_scores = [], []
            sensitivity_avg_scores, sensitivity_weighted_scores = [], []

            for cv_fold, group in df.groupby(level="cv_fold"):
                try:
                    y_true = pd.get_dummies(group["level_int"])
                    y_pred = group[["prob_class_1", "prob_class_2", "prob_class_3"]]
                    y_pred_labels = y_pred.idxmax(axis=1).str.extract(r'(\d)').astype(int)
                    y_true_labels = group["level_int"]
                    y_pred_labels_ = y_pred.idxmax(axis=1).apply(lambda x: int(x[-1]))

                    for i, class_name in enumerate([1, 2, 3]):
                        if class_name in y_true.columns:
                            num_samples[class_name] += y_true.iloc[:, i].sum()

                            metrics_scores["auc"][class_name].append(
                                roc_auc_score(y_true.iloc[:, i], y_pred.iloc[:, i])
                            )
                            metrics_scores["mse"][class_name].append(
                                mean_squared_error(y_true.iloc[:, i], y_pred.iloc[:, i])
                            )
                            metrics_scores["accuracy"][class_name].append(
                                accuracy_score(y_true.iloc[:, i], y_pred_labels[0] == class_name)
                            )
                            metrics_scores["f1"][class_name].append(
                                f1_score(y_true.iloc[:, i], y_pred_labels[0] == class_name)
                            )
                            metrics_scores["sensitivity"][class_name].append(
                                recall_score(y_true.iloc[:, i], y_pred_labels[0] == class_name)
                            )

                    auc_avg_scores.append(roc_auc_score(y_true, y_pred, multi_class="ovr"))
                    auc_weighted_scores.append(roc_auc_score(y_true, y_pred, average='weighted', multi_class="ovr"))

                    mse_avg_scores.append(mean_squared_error(y_true, y_pred))
                    mse_weighted_scores.append(mean_squared_error(y_true, y_pred, multioutput='uniform_average'))

                    accuracy_avg_scores.append(accuracy_score(y_true_labels, y_pred_labels_))
                    accuracy_weighted_scores.append(accuracy_score(y_true_labels, y_pred_labels_))

                    f1_avg_scores.append(f1_score(y_true_labels, y_pred_labels_, average='macro'))
                    f1_weighted_scores.append(f1_score(y_true_labels, y_pred_labels_, average='weighted'))

                    sensitivity_avg_scores.append(recall_score(y_true_labels, y_pred_labels_, average='macro'))
                    sensitivity_weighted_scores.append(recall_score(y_true_labels, y_pred_labels_, average='weighted'))

                except Exception as e:
                    evaluation_logger.warning(f"Failed to compute metrics for experiment {experiment_id}, cv_fold {cv_fold}: {e}")

            if auc_avg_scores:
                results.append({
                    "index": experiment_id,
                    "model":model_name,
                    "auc_avg": np.mean(auc_avg_scores),
                    "auc_weighted_avg": np.mean(auc_weighted_scores),
                    "mse_avg": np.mean(mse_avg_scores),
                    "mse_weighted_avg": np.mean(mse_weighted_scores),
                    "accuracy_avg": np.mean(accuracy_avg_scores),
                    "accuracy_weighted_avg": np.mean(accuracy_weighted_scores),
                    "f1_avg": np.mean(f1_avg_scores),
                    "f1_weighted_avg": np.mean(f1_weighted_scores),
                    "sensitivity_avg": np.mean(sensitivity_avg_scores),
                    "sensitivity_weighted_avg": np.mean(sensitivity_weighted_scores),
                    "auc_class_1": np.mean(metrics_scores["auc"][1]) if metrics_scores["auc"][1] else None,
                    "auc_class_2": np.mean(metrics_scores["auc"][2]) if metrics_scores["auc"][2] else None,
                    "auc_class_3": np.mean(metrics_scores["auc"][3]) if metrics_scores["auc"][3] else None,
                    "mse_class_1": np.mean(metrics_scores["mse"][1]) if metrics_scores["mse"][1] else None,
                    "mse_class_2": np.mean(metrics_scores["mse"][2]) if metrics_scores["mse"][2] else None,
                    "mse_class_3": np.mean(metrics_scores["mse"][3]) if metrics_scores["mse"][3] else None,
                    "accuracy_class_1": np.mean(metrics_scores["accuracy"][1]) if metrics_scores["accuracy"][1] else None,
                    "accuracy_class_2": np.mean(metrics_scores["accuracy"][2]) if metrics_scores["accuracy"][2] else None,
                    "accuracy_class_3": np.mean(metrics_scores["accuracy"][3]) if metrics_scores["accuracy"][3] else None,
                    "f1_class_1": np.mean(metrics_scores["f1"][1]) if metrics_scores["f1"][1] else None,
                    "f1_class_2": np.mean(metrics_scores["f1"][2]) if metrics_scores["f1"][2] else None,
                    "f1_class_3": np.mean(metrics_scores["f1"][3]) if metrics_scores["f1"][3] else None,
                    "sensitivity_class_1": np.mean(metrics_scores["sensitivity"][1]) if metrics_scores["sensitivity"][1] else None,
                    "sensitivity_class_2": np.mean(metrics_scores["sensitivity"][2]) if metrics_scores["sensitivity"][2] else None,
                    "sensitivity_class_3": np.mean(metrics_scores["sensitivity"][3]) if metrics_scores["sensitivity"][3] else None,
                    "num_samples_class_1": num_samples[1],
                    "num_samples_class_2": num_samples[2],
                    "num_samples_class_3": num_samples[3],
                })


        except Exception as e:
            evaluation_logger.error(f"Error evaluating experiment {experiment_id}: {e}")

    results_df = pd.DataFrame(results)
    return experiments_to_update.merge(results_df, on="index", how="left")

def update_experiments_file(experiments_valid,summary_path):
    if not os.path.exists(summary_path):
        evaluation_logger.info("Summary file not found, all experiments updated.")
        experiments_valid.to_excel(summary_path,index=False)
    else:
        summary_df=pd.read_excel(summary_path)
        summary_df = summary_df[~summary_df["index"].isin(experiments_valid["index"].to_list())]
        pd.concat([summary_df,experiments_valid]).to_excel(summary_path,index=False)

