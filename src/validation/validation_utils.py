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

    experiments_to_update = experiments_to_update.sort_values(by='timestamp', ascending=False)
    experiments_to_update = experiments_to_update.drop_duplicates(subset='index', keep='first')
    experiments_to_update = experiments_to_update.reset_index(drop=True)

    evaluation_logger.info(f"Experiments to update: {len(experiments_to_update)}")

    return experiments_to_update

def maps_levels(df):
    num_classes = len(df["level"].unique())
    level_mapping = {
        'FEV1 [-10,inf)': 1,
        'FEV1 [-20,-10)': 2,
        'FEV1 [-inf,-20)': 3
    }
    df["level_int"] = df["level"].map(level_mapping)

    metrics_scores = {metric: {i: [] for i in range(1, 4)} for metric in
                      ["auc", "mse", "accuracy", "f1", "sensitivity"]}
    num_samples = {i: 0 for i in range(1, 4)}

    if num_classes == 2:
        for metric in metrics_scores.values():
            metric.pop(2, None)
        num_samples.pop(2, None)

    return df,metrics_scores,num_samples

def predict_ensemble(experiment_params,predict_dir,p_i=''):
    df_prob_1=pd.DataFrame()
    df_prob_2 = pd.DataFrame()
    df_prob_3 = pd.DataFrame()

    for run_number, ex_id in enumerate(experiment_params, start=1):

        prediction_file = os.path.join(predict_dir, f"cv_probabilities{p_i}_{ex_id}.parquet")
        df = pd.read_parquet(prediction_file)
        num_classes = len(df["level"].unique())
        df_prob_1[run_number] = df['prob_class_1']
        if num_classes==3:
            df_prob_2[run_number] = df['prob_class_2']
        df_prob_3[run_number] = df['prob_class_3']

    df_prob_1['avg_all_columns'] = df_prob_1.mean(axis=1)
    if num_classes == 3:
        df_prob_2['avg_all_columns'] = df_prob_2.mean(axis=1)
    df_prob_3['avg_all_columns'] = df_prob_3.mean(axis=1)
    df['prob_class_1'] = df_prob_1['avg_all_columns']
    if num_classes == 3:
        df['prob_class_2'] = df_prob_2['avg_all_columns']
    df['prob_class_3'] = df_prob_3['avg_all_columns']

    return df

def evaluate_experiments(experiments_to_update, predict_dir,Patients_level_3=['']):
    results = []
    name_p = '' if Patients_level_3[0]=='' else '_P'
    for p_i in Patients_level_3:
        for _, experiment in experiments_to_update.iterrows():
            experiment_id = experiment["index"]
            prediction_file = os.path.join(predict_dir, f"cv_probabilities{name_p}{p_i}_{experiment_id}.parquet")


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
                    p_i_param=(f'_P{p_i}' if p_i!='' else '')
                    df = predict_ensemble(experiment_params,predict_dir,p_i=p_i_param)
                    model_name=experiment_id

                df,metrics_scores,num_samples=maps_levels(df)

                auc_avg_scores, auc_weighted_scores = [], []
                mse_avg_scores, mse_weighted_scores = [], []
                accuracy_avg_scores, accuracy_weighted_scores = [], []
                f1_avg_scores, f1_weighted_scores = [], []
                sensitivity_avg_scores, sensitivity_weighted_scores = [], []

                for cv_fold, group in df.groupby(level="cv_fold"):
                    try:
                        y_true = pd.get_dummies(group["level_int"])
                        num_classes = y_true.shape[1]
                        column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else [
                            'prob_class_1', 'prob_class_3']
                        y_pred = group[column_names]
                        y_pred_labels = y_pred.idxmax(axis=1).str.extract(r'(\d)').astype(int)
                        y_true_labels = group["level_int"]
                        y_pred_labels_ = y_pred.idxmax(axis=1).apply(lambda x: int(x[-1]))
                        class_list  = [i + 1 for i in range(3)] if num_classes == 3 else [1,3]
                        for i, class_name in enumerate(class_list):
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

                        accuracy_avg_scores.append(accuracy_score(y_true_labels, y_pred_labels_))
                        accuracy_weighted_scores.append(accuracy_score(y_true_labels, y_pred_labels_))

                        f1_avg_scores.append(f1_score(y_true_labels, y_pred_labels_, average='macro'))
                        f1_weighted_scores.append(f1_score(y_true_labels, y_pred_labels_, average='weighted'))

                        sensitivity_avg_scores.append(recall_score(y_true_labels, y_pred_labels_, average='macro'))
                        sensitivity_weighted_scores.append(recall_score(y_true_labels, y_pred_labels_, average='weighted'))

                    except Exception as e:
                        evaluation_logger.warning(f"Failed to compute metrics for experiment {experiment_id}, cv_fold {cv_fold}: {e}")

                if auc_avg_scores:
                    ll={
                        "index": experiment_id,
                        "model":model_name,
                        "Patients":str(p_i),
                        "auc_avg": np.mean(auc_avg_scores),
                        "auc_weighted_avg": np.mean(auc_weighted_scores),
                        "mse_avg": np.mean(mse_avg_scores),
                        "accuracy_avg": np.mean(accuracy_avg_scores),
                        "accuracy_weighted_avg": np.mean(accuracy_weighted_scores),
                        "f1_avg": np.mean(f1_avg_scores),
                        "f1_weighted_avg": np.mean(f1_weighted_scores),
                        "sensitivity_avg": np.mean(sensitivity_avg_scores),
                        "sensitivity_weighted_avg": np.mean(sensitivity_weighted_scores),}
                    for i in class_list:
                        ll[f'auc_class_{i}']=np.mean(metrics_scores["auc"][i]) if metrics_scores["auc"][i] else None
                        ll[f'mse_class_{i}'] = np.mean(metrics_scores["mse"][i]) if metrics_scores["mse"][i] else None
                        ll[f'accuracy_class_{i}'] = np.mean(metrics_scores["accuracy"][i]) if metrics_scores["accuracy"][i] else None
                        ll[f'f1_class_{i}'] = np.mean(metrics_scores["f1"][i]) if metrics_scores["f1"][i] else None
                        ll[f'sensitivity_class_{i}'] = np.mean(metrics_scores["sensitivity"][i]) if metrics_scores["sensitivity"][i] else None
                        ll[f'num_samples_class_{i}'] = num_samples[i]


                    results.append(ll)




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

def summary_results_mixed(result_path,summary_path):
    summary_df=pd.read_excel(summary_path)
    selected_columns = ['model', 'auc_weighted_avg', 'mse_avg', 'accuracy_weighted_avg',
                        'f1_weighted_avg', 'sensitivity_weighted_avg']
    filtered_df = summary_df[selected_columns]
    best_models_df = filtered_df.loc[filtered_df.groupby('model')['auc_weighted_avg'].idxmax()]
    model_order = ['DecisionTrees', 'DecisionTrees_Ordinal', 'AdaBoost', 'AdaBoost_Ordinal',
                   'RandomForest', 'RandomForest_Ordinal', 'catboost', 'XGBoost', 'ensemble']
    best_models_df = best_models_df.set_index('model').loc[model_order].reset_index()
    best_models_df.to_excel(f'{result_path}/mix_results.xlsx', index=False)

def summary_results_independent(result_path,summary_path):
    summary_df=pd.read_excel(summary_path)
    summary_df['Have 3 classes']=np.where(summary_df['num_samples_class_2']>0,True,False)
    selected_columns = ['Patients','model', 'auc_weighted_avg', 'mse_avg', 'accuracy_weighted_avg',
                        'f1_weighted_avg', 'sensitivity_weighted_avg','Have 3 classes']

    filtered_df = summary_df[selected_columns]
    best_models_df = filtered_df.loc[filtered_df.groupby(['Patients', 'model'])['auc_weighted_avg'].idxmax()]


    model_order = ['DecisionTrees', 'DecisionTrees_Ordinal', 'AdaBoost', 'AdaBoost_Ordinal',
                   'RandomForest', 'RandomForest_Ordinal', 'catboost', 'XGBoost', 'ensemble']
    best_models_df['model'] = pd.Categorical(best_models_df['model'], categories=model_order, ordered=True)
    best_models_df = best_models_df.sort_values(by=['Patients', 'model']).reset_index(drop=True)

    best_models_df.to_excel(f'{result_path}/independent_results.xlsx', index=False)