import pandas as pd
import os
from src.utils.setup_logger import evaluation_logger
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score, f1_score, recall_score,cohen_kappa_score
import numpy as np
import ast
from datetime import datetime
from scipy.stats import ttest_ind, ttest_rel,wasserstein_distance


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
        filtered_rows=filtered_rows[filtered_rows['params'].apply(lambda x: ast.literal_eval(x).get('smote') ==  False)]
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
        if 'probabilistic' not in tracking_path:
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

def add_stats(metric_list, name, ll):
    mean = np.mean(metric_list)
    std = np.std(metric_list)
    cv = 100 * std / mean if mean != 0 else 0
    ll[f"{name}_std"] = std
    ll[f"{name}_cv_percent"] = cv

    return ll


def metrics_per_class(y_true, y_pred, y_pred_labels, class_list, metrics_scores, num_samples):

    for i, class_name in enumerate(class_list):
        if class_name in y_true.columns:
            y_true_col = y_true.iloc[:, i]
            y_pred_col = y_pred.iloc[:, i]
            pred_class_match = (y_pred_labels[0] == class_name)

            num_samples[class_name] += y_true_col.sum()

            metrics_scores["auc"][class_name].append(
                roc_auc_score(y_true_col, y_pred_col)
            )
            metrics_scores["mse"][class_name].append(
                mean_squared_error(y_true_col, y_pred_col)
            )
            metrics_scores["accuracy"][class_name].append(
                accuracy_score(y_true_col, pred_class_match)
            )
            metrics_scores["f1"][class_name].append(
                f1_score(y_true_col, pred_class_match)
            )
            metrics_scores["sensitivity"][class_name].append(
                recall_score(y_true_col, pred_class_match)
            )

    return metrics_scores, num_samples

def ordinal_auc(y_true_labels, y_pred_proba):
    y_score = np.dot(y_pred_proba, np.arange(y_pred_proba.shape[1]))
    count, total = 0, 0
    for i in range(len(y_true_labels)):
        for j in range(i + 1, len(y_true_labels)):
            if y_true_labels[i] < y_true_labels[j]:
                total += 1
                if y_score[i] < y_score[j]:
                    count += 1
            elif y_true_labels[i] > y_true_labels[j]:
                total += 1
                if y_score[i] > y_score[j]:
                    count += 1
    return count / total if total > 0 else np.nan




def emd_all(y_true, y_pred_proba):
    emd_scores = []
    classes = np.arange(y_pred_proba.shape[1])
    for i, true_class in enumerate(y_true):
        true_dist = np.zeros_like(classes, dtype=float)
        true_dist[true_class] = 1.0
        pred_dist = y_pred_proba[i]
        emd = wasserstein_distance(classes, classes, true_dist, pred_dist)
        emd_scores.append(emd)
    return np.mean(emd_scores)


def evaluate_experiments(experiments_to_update, predict_dir,Patients_level_3=['']):
    results = []
    auc_per_fold_results = []
    name_p = '' if Patients_level_3[0]=='' else '_P'
    affine = True if 'affine' in predict_dir else False
    for p_i in Patients_level_3:
        for idx, experiment in experiments_to_update.iterrows():
            if idx % 100 == 0:
                evaluation_logger.info(f"Experiments start : {idx} / {len(experiments_to_update)}")
            experiment_id = experiment["index"]
            prediction_file = os.path.join(predict_dir, f"cv_probabilities{name_p}{p_i}_{experiment_id}.parquet")
            if affine:
                if os.path.exists(os.path.join(predict_dir, f"cv_probabilities{name_p}{p_i}_True_{experiment_id}.parquet")):
                    prediction_file = os.path.join(predict_dir,
                                                   f"cv_probabilities{name_p}{p_i}_True_{experiment_id}.parquet")
                else:
                    prediction_file = os.path.join(predict_dir,
                                                   f"cv_probabilities{name_p}{p_i}_False_{experiment_id}.parquet")

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
                metrics_scores_per_p=metrics_scores.copy()
                num_samples_per_p = num_samples.copy()
                auc_avg_scores, auc_weighted_scores = [], []
                mse_avg_scores, mse_weighted_scores = [], []
                accuracy_scores = []
                f1_avg_scores, f1_weighted_scores = [], []
                sensitivity_avg_scores, sensitivity_weighted_scores = [], []
                ordinal_auc_scores = []
                emd_scores = []
                qwk_scores = []

                p_ll={}
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
                        metrics_scores, num_samples = metrics_per_class(y_true, y_pred, y_pred_labels, class_list, metrics_scores, num_samples)


                        auc_avg_scores.append(roc_auc_score(y_true, y_pred, multi_class="ovr"))
                        auc_weighted_scores.append(roc_auc_score(y_true, y_pred, average='weighted', multi_class="ovr"))

                        mse_avg_scores.append(mean_squared_error(y_true, y_pred))

                        accuracy_scores.append(accuracy_score(y_true_labels, y_pred_labels_))

                        f1_avg_scores.append(f1_score(y_true_labels, y_pred_labels_, average='macro'))
                        f1_weighted_scores.append(f1_score(y_true_labels, y_pred_labels_, average='weighted'))

                        sensitivity_avg_scores.append(recall_score(y_true_labels, y_pred_labels_, average='macro'))
                        sensitivity_weighted_scores.append(recall_score(y_true_labels, y_pred_labels_, average='weighted'))

                        ordinal_auc_scores.append(ordinal_auc(y_true_labels.values-1, y_pred.values))
                        emd_scores.append(emd_all(y_true_labels.values-1, y_pred.values))
                        qwk_scores.append(cohen_kappa_score(y_true_labels.values, y_pred_labels_, weights='quadratic'))






                    except Exception as e:
                        evaluation_logger.warning(f"Failed to compute metrics for experiment {experiment_id}, cv_fold {cv_fold}: {e}")

                if 'mixed' not in predict_dir:
                    for p_i_test in df["Patient_NO"].unique():
                        df_p = df[df["Patient_NO"] == p_i_test]
                        y_true = pd.get_dummies(df_p["level_int"])
                        column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else [
                            'prob_class_1', 'prob_class_3']
                        y_pred = df_p[column_names]

                        p_ll[f"Patient_NO_{p_i_test}_auc_avg"] = roc_auc_score(y_true, y_pred, multi_class="ovr")
                        p_ll[f"Patient_NO_{p_i_test}_auc_weighted"] = roc_auc_score(y_true, y_pred,
                                                                                              average='weighted',
                                                                                              multi_class="ovr")
                        p_ll[f"Patient_NO_{p_i_test}_samples"] =df_p['level_int'].value_counts().to_dict()

                auc_per_fold_results.append({
                    'experiment_id': experiment_id,
                    'model': model_name,
                    'Patients': str(p_i),
                    'auc_weighted_cv1': auc_weighted_scores[0],
                    'auc_weighted_cv2': auc_weighted_scores[1],
                    'auc_weighted_cv3': auc_weighted_scores[2],
                    'auc_weighted_cv4': auc_weighted_scores[3],
                    'auc_weighted_cv5': auc_weighted_scores[4],
                    'auc_avg_cv1': auc_avg_scores[0],
                    'auc_avg_cv2': auc_avg_scores[1],
                    'auc_avg_cv3': auc_avg_scores[2],
                    'auc_avg_cv4': auc_avg_scores[3],
                    'auc_avg_cv5': auc_avg_scores[4],

                })

                if auc_avg_scores:
                    ll={
                        "index": experiment_id,
                        "model":model_name,
                        "Patients":str(p_i),
                        "auc_avg": np.mean(auc_avg_scores),
                        "auc_weighted_avg": np.mean(auc_weighted_scores),
                        "mse_avg": np.mean(mse_avg_scores),
                        "accuracy_avg": np.mean(accuracy_scores),
                        "f1_avg": np.mean(f1_avg_scores),
                        "f1_weighted_avg": np.mean(f1_weighted_scores),
                        "sensitivity_avg": np.mean(sensitivity_avg_scores),
                        "sensitivity_weighted_avg": np.mean(sensitivity_weighted_scores),
                        "auc_ordinal_avg": np.mean(ordinal_auc_scores),
                    "emd_avg": np.mean(emd_scores ),
                    "qwk_avg": np.mean(qwk_scores ),}

                    ll = add_stats(auc_avg_scores, "auc_avg", ll)
                    ll = add_stats(auc_weighted_scores, "auc_weighted", ll)
                    ll = add_stats(mse_avg_scores, "mse", ll)
                    ll = add_stats(accuracy_scores, "accuracy_avg", ll)
                    ll = add_stats(f1_avg_scores, "f1_avg", ll)
                    ll = add_stats(f1_weighted_scores, "f1_weighted", ll)
                    ll = add_stats(sensitivity_avg_scores, "sensitivity_avg", ll)
                    ll = add_stats(sensitivity_weighted_scores, "sensitivity_weighted", ll)
                    ll = add_stats(ordinal_auc_scores, "auc_ordinal", ll)
                    ll = add_stats(emd_scores, "emd_avg", ll)
                    ll = add_stats(qwk_scores, "qwk_avg", ll)

                    for i in class_list:
                        ll[f'auc_class_{i}']=np.mean(metrics_scores["auc"][i]) if metrics_scores["auc"][i] else None
                        ll[f'mse_class_{i}'] = np.mean(metrics_scores["mse"][i]) if metrics_scores["mse"][i] else None
                        ll[f'accuracy_class_{i}'] = np.mean(metrics_scores["accuracy"][i]) if metrics_scores["accuracy"][i] else None
                        ll[f'f1_class_{i}'] = np.mean(metrics_scores["f1"][i]) if metrics_scores["f1"][i] else None
                        ll[f'sensitivity_class_{i}'] = np.mean(metrics_scores["sensitivity"][i]) if metrics_scores["sensitivity"][i] else None
                        ll[f'num_samples_class_{i}'] = num_samples[i]



                    results.append({**ll, **p_ll})




            except Exception as e:
                evaluation_logger.error(f"Error evaluating experiment {experiment_id}: {e}")

    results_df = pd.DataFrame(results)
    auc_per_fold_df = pd.DataFrame(auc_per_fold_results)
    if results_df.empty:
        return  experiments_to_update,auc_per_fold_df
    else:
        return experiments_to_update.merge(results_df, on="index", how="left"),auc_per_fold_df

def evaluate_experiments_independet(experiments_to_update, predict_dir,Patients_level_3=['']):
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

def evaluate_experiments_prob(experiments_to_update, predict_dir,Patients_level_3=['']):
    results = []
    name_p='_P'
    for _, experiment in experiments_to_update.iterrows():
        auc_avg_scores, auc_weighted_scores = [], []
        mse_avg_scores, mse_weighted_scores = [], []
        accuracy_avg_scores, accuracy_weighted_scores = [], []
        f1_avg_scores, f1_weighted_scores = [], []
        sensitivity_avg_scores, sensitivity_weighted_scores = [], []
        for p_i in Patients_level_3:
            experiment_id = experiment["index"]
            prediction_file = os.path.join(predict_dir, f"probabilities_step_2{name_p}{p_i}_{experiment_id}.parquet")
            try:
                df = pd.read_parquet(prediction_file)
                model_name_step1= ast.literal_eval(experiment["params_step1"])['model']
                model_name_step2 = ast.literal_eval(experiment["params_step2"])['model']
                df,metrics_scores,num_samples=maps_levels(df)



                try:
                    y_true = pd.get_dummies(df["level_int"])
                    num_classes = y_true.shape[1]
                    column_names = [f"prob_class_{i + 1}" for i in range(3)] if num_classes == 3 else [
                        'prob_class_1', 'prob_class_3']
                    y_pred = df[column_names]
                    y_pred_labels = y_pred.idxmax(axis=1).str.extract(r'(\d)').astype(int)
                    y_true_labels = df["level_int"]
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
                    evaluation_logger.warning(f"Failed to compute metrics for experiment {experiment_id}: {e}")

                if auc_avg_scores:
                    ll={
                        "index": experiment_id,
                        "model_step1":model_name_step1,
                        "model_step2": model_name_step2,
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

def update_experiments_file(experiments_valid, auc_per_fold_df, summary_path):
    if not os.path.exists(summary_path):
        evaluation_logger.info("Summary file not found, all experiments updated.")
        with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
            experiments_valid.to_excel(writer, sheet_name='Sheet1', index=False)
            auc_per_fold_df.to_excel(writer, sheet_name='auc_per_fold_df', index=False)
    else:
        summary_df = pd.read_excel(summary_path, sheet_name='Sheet1')
        summary_df = summary_df[~summary_df["index"].isin(experiments_valid["index"].to_list())]
        updated_df = pd.concat([summary_df, experiments_valid], ignore_index=True)

        summary_df = pd.read_excel(summary_path, sheet_name='auc_per_fold_df')
        summary_df = summary_df[~summary_df["experiment_id"].isin(auc_per_fold_df["experiment_id"].to_list())]
        updated_auc_df = pd.concat([summary_df, auc_per_fold_df], ignore_index=True)

        with pd.ExcelWriter(summary_path, engine='openpyxl', mode='w') as writer:
            updated_df.to_excel(writer, sheet_name='Sheet1', index=False)
            updated_auc_df.to_excel(writer, sheet_name='auc_per_fold_df', index=False)

def update_experiments_file_independent(experiments_valid,summary_path):
    if not os.path.exists(summary_path):
        evaluation_logger.info("Summary file not found, all experiments updated.")
        experiments_valid.to_excel(summary_path,index=False)
    else:
        summary_df=pd.read_excel(summary_path)
        summary_df = summary_df[~summary_df["index"].isin(experiments_valid["index"].to_list())]
        pd.concat([summary_df,experiments_valid]).to_excel(summary_path,index=False)

def summary_results_mixed(result_path,summary_path):
    summary_df=pd.read_excel(summary_path)
    selected_columns = ['index','params','model'
    , 'auc_weighted_avg', 'auc_weighted_std', 'auc_weighted_cv_percent'
    , 'auc_ordinal_avg', 'auc_ordinal_std', 'auc_ordinal_cv_percent'
    , 'emd_avg', 'emd_avg_std', 'emd_avg_cv_percent'
    , 'qwk_avg', 'qwk_avg_std', 'qwk_avg_cv_percent'
    , 'auc_avg', 'auc_avg_std', 'auc_avg_cv_percent'
    , 'mse_avg', 'mse_std', 'mse_cv_percent'
    , 'accuracy_avg', 'accuracy_avg_std', 'accuracy_avg_cv_percent'
    , 'f1_weighted_avg', 'f1_weighted_std', 'f1_weighted_cv_percent'
    , 'sensitivity_weighted_avg', 'sensitivity_weighted_std', 'sensitivity_weighted_cv_percent'
    , 'auc_class_1', 'auc_class_2', 'auc_class_3'
    , 'num_samples_class_1', 'num_samples_class_2', 'num_samples_class_3'
    , 'total_time']
    filtered_df = summary_df[selected_columns]
    best_models_df = filtered_df.loc[filtered_df.groupby('model')['auc_weighted_avg'].idxmax()]
    model_order = ['DecisionTrees', 'DecisionTrees_Ordinal', 'AdaBoost', 'AdaBoost_Ordinal',
                   'RandomForest', 'RandomForest_Ordinal', 'catboost', 'XGBoost', 'ensemble']
    best_models_df = best_models_df.set_index('model').loc[model_order].reset_index()
    best_models_df.to_excel(f'{result_path}/mixed_results.xlsx', index=False)

def compute_total_time(row):
    if row['type'] == 'None':
        return row['fit_model_time_seconds']
    elif row['type'] == 'DR':
        return row['fit_model_time_seconds'] + row['dr_time_seconds']
    elif row['type'] == 'DR_AFFINE':
        return row['fit_model_time_seconds'] + row['dr_time_seconds'] + row['affine_time_seconds']
    else:
        return 0

def summary_results_affine(result_path,summary_path):
    summary_df=pd.read_excel(summary_path)
    summary_df['params2'] = summary_df['params'].apply(ast.literal_eval)

    summary_df['affine'] = summary_df['params2'].apply(
        lambda x: x.get('affine') if isinstance(x, dict) else None
    )

    summary_df['p_anchor'] = summary_df['params2'].apply(lambda x: x.get('p_anchor') if isinstance(x, dict) else None)
    summary_df['affine_transform'] = summary_df['params2'].apply(
        lambda x: x.get('affine_transform') if isinstance(x, dict) else None
    )
    summary_df['dimensional_reduction_transform'] = summary_df['params2'].apply(
        lambda x: x.get('dimensional_reduction_transform') if isinstance(x, dict) else None
    )
    summary_df['dimensional_reduction'] = summary_df['params2'].apply(
        lambda x: x.get('dimensional_reduction') if isinstance(x, dict) else None
    )
    summary_df['number_dimensional']=summary_df['dimensional_reduction'].str.extract(r'(\d+)$').astype(int)
    summary_df['type'] = summary_df.apply(
        lambda row:
        'FULL_with_AFFINE' if 'FULL_with_AFFINE' in row['dimensional_reduction']
        else 'DR_AFFINE' if row['affine_transform'] and row['dimensional_reduction_transform']
        else 'AFFINE' if row['affine_transform']
        else 'DR' if row['dimensional_reduction_transform']
        else 'None',
        axis=1
    )

    summary_df['type'] = np.where(
        summary_df['affine'] != 'regular',
        summary_df['type'] + '_' + summary_df['affine'],
        summary_df['type']
    )

    selected_columns = ['index'
        ,'params','model','affine_transform','dimensional_reduction_transform','dimensional_reduction','number_dimensional','type'
        ,'p_anchor'
                        ,'affine'
        , 'auc_weighted_avg','auc_weighted_std','auc_weighted_cv_percent'
        , 'auc_ordinal_avg', 'auc_ordinal_std', 'auc_ordinal_cv_percent'
        , 'emd_avg', 'emd_avg_std', 'emd_avg_cv_percent'
        , 'qwk_avg', 'qwk_avg_std', 'qwk_avg_cv_percent'
        , 'auc_avg', 'auc_avg_std', 'auc_avg_cv_percent'
        , 'mse_avg','mse_std','mse_cv_percent'
        , 'accuracy_avg','accuracy_avg_std','accuracy_avg_cv_percent'
        , 'f1_weighted_avg','f1_weighted_std','f1_weighted_cv_percent'
        , 'sensitivity_weighted_avg','sensitivity_weighted_std','sensitivity_weighted_cv_percent'
        ,'auc_class_1','auc_class_2','auc_class_3'
        , 'num_samples_class_1', 'num_samples_class_2', 'num_samples_class_3'
        ,'total_time']
    selected_columns += [col for col in summary_df.columns if "Patient_NO_P" in col]
    filtered_df = summary_df[selected_columns]
    best_models_df = filtered_df.loc[filtered_df.groupby(['model','affine_transform','dimensional_reduction_transform'
                                                             ,'dimensional_reduction','p_anchor','type'])['auc_weighted_avg'].idxmax()]
    model_order = ['DecisionTrees', 'DecisionTrees_Ordinal', 'AdaBoost', 'AdaBoost_Ordinal',
                   'RandomForest', 'RandomForest_Ordinal', 'catboost', 'XGBoost', 'ensemble']
    available_models = [m for m in model_order if m in best_models_df['model'].values]
    best_models_df = best_models_df.set_index('model').loc[available_models].reset_index()

    time_df = pd.read_excel(r'C:\Users\user\OneDrive - Bar-Ilan University - Students\PHD Rotem Haba\Ordinal-Model\data\processed\dimensional_reduction_times.xlsx')
    best_models_df=best_models_df.merge(
        time_df,
        on='dimensional_reduction',
        how='left'
    )
    best_models_df.rename(columns={'total_time': 'fit_model_time_seconds'}, inplace=True)
    best_models_df['total_time'] = best_models_df.apply(compute_total_time, axis=1)

    best_models_df.to_excel(f'{result_path}/affine_results.xlsx', index=False)

def summary_results_independent(result_path,summary_path):
    summary_df=pd.read_excel(summary_path)
    summary_df['Have 3 classes']=np.where(summary_df['num_samples_class_2']>0,True,False)
    selected_columns = ['Patients','model', 'auc_avg', 'auc_weighted_avg', 'mse_avg', 'accuracy_weighted_avg',
                        'f1_weighted_avg', 'sensitivity_weighted_avg','Have 3 classes','auc_class_1','auc_class_2','auc_class_3']

    filtered_df = summary_df[selected_columns]
    best_models_df = filtered_df.loc[filtered_df.groupby(['Patients', 'model'])['auc_weighted_avg'].idxmax()]


    model_order = ['DecisionTrees', 'DecisionTrees_Ordinal', 'AdaBoost', 'AdaBoost_Ordinal',
                   'RandomForest', 'RandomForest_Ordinal', 'catboost', 'XGBoost', 'ensemble']
    best_models_df['model'] = pd.Categorical(best_models_df['model'], categories=model_order, ordered=True)
    best_models_df = best_models_df.sort_values(by=['Patients', 'model']).reset_index(drop=True)

    best_models_df.to_excel(f'{result_path}/independent_results.xlsx', index=False)

def find_elbow(x, y):
    x = np.array(x)
    y = np.array(y)

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    numerator = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator

    elbow_idx = distances.argmax()
    return x[elbow_idx], y[elbow_idx]

def knee_point(result_path):
    file_path = os.path.join(result_path, "affine_results.xlsx")

    # Load data
    df = pd.read_excel(file_path)

    # Sort data to ensure elbow method works correctly
    df_sorted = df.sort_values(by=['model', 'type', 'number_dimensional'])

    results = []
    for (model, type_), group in df_sorted.groupby(['model', 'type']):
        x = group['number_dimensional'].values
        y = group['auc_weighted_avg'].values

        if len(x) < 5:
            continue  # not enough points

        elbow_x, elbow_y = find_elbow(x, y)
        results.append({
            'model': model,
            'type': type_,
            'elbow_number_dimensional': elbow_x,
            'auc_at_elbow': elbow_y
        })

    elbow_df = pd.DataFrame(results)

    # Save to new sheet in the same file
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        elbow_df.to_excel(writer, sheet_name='elbow_points', index=False)

    print("Elbow points saved to 'elbow_points' sheet in affine_results.xlsx")

def  t_test_point(result_path,summary_path,on='auc_avg',number_dimensional=7):

    summary_df = pd.read_excel(summary_path,sheet_name='auc_per_fold_df')
    auc_columns = [f'{on}_cv{i}' for i in range(1, 6)]
    summary_df = summary_df[['experiment_id']+auc_columns]
    folder_name = os.path.basename(os.path.dirname(summary_path))
    result_df = pd.read_excel(f'{result_path}/{folder_name}_results.xlsx')
    merged_df = result_df.merge(summary_df, left_on="index", right_on="experiment_id", how="left")
    merged_df['dimensional_reduction'] = merged_df['dimensional_reduction'].str.replace('all_', '', regex=False)

    results = []

    for model_i in merged_df['model'].unique():
        df_model = merged_df[merged_df['model'] == model_i]

        for dr in df_model['dimensional_reduction'].unique():
            df = df_model[df_model['dimensional_reduction'] == dr]
            types = df['type'].unique()

            if len(types) < 2:
                continue  # can't compare less than 2 types

            type_auc = {}
            for t in types:
                auc_values = df[df['type'] == t][auc_columns].values.flatten()
                type_auc[t] = auc_values

            row = {'model': model_i, 'dimensional_reduction': dr}
            for i in range(len(types)):
                for j in range(i + 1, len(types)):
                    t1, t2 = types[i], types[j]
                    try:

                        stat, p_val = ttest_rel(type_auc[t1], type_auc[t2])
                        row[f't_test_pval_{t1}_vs_{t2}'] = p_val

                    except Exception:
                        row[f't_test_pval_{t1}_vs_{t2}'] = None

            results.append(row)
    if len(results) > 0:

        t_test_df = pd.DataFrame(results)
        t_test_df['dimensional_reduction'] = t_test_df['dimensional_reduction'].str.replace('PLS_range_', '').astype(int)
        first_cols = ['model', 'dimensional_reduction']
        pval_cols = [col for col in t_test_df.columns if 't_test_pval' in col]
        t_test_df = t_test_df[first_cols + pval_cols ]

        excel_file = os.path.join(result_path, folder_name+"_results.xlsx")

        with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            t_test_df.to_excel(writer, sheet_name=f't_test_{on}', index=False)
            merged_df[['model', 'type', 'number_dimensional'] + auc_columns].to_excel(writer,
                                                                                      sheet_name=f't_test_{on}_data',
                                                                                      index=False)

def  patient_auc(result_path,on,number_dimensional=7):
    summary_df = pd.read_excel(f'{result_path}/affine_results.xlsx')
    columns_to_keep=['model','type','number_dimensional']
    for p_ in ['P_1','P_5','P_6','P_7','P_10','P_17','P_18','P_47']:
        columns_to_keep.append(f'Patient_NO_{p_}_{on}')

    for p_ in ['P_1','P_5','P_6','P_7','P_10','P_17','P_18','P_47']:
        columns_to_keep.append(f'Patient_NO_{p_}_samples')

    summary_df= summary_df[columns_to_keep]
    excel_file = os.path.join(result_path, "affine_results.xlsx")
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name=f'patient_{on}', index=False)


