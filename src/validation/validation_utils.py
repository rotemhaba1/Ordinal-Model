import pandas as pd
import os
from src.utils.setup_logger import evaluation_logger

def find_experiments_to_update(tracking_path, summary_path):
    if os.path.exists(tracking_path):
        tracking_df = pd.read_excel(tracking_path)
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

    missing_experiments = tracking_df[~tracking_df["experiment_id"].astype(str).isin(summary_df["experiment_id"].astype(str))]

    merged_df = tracking_df.merge(summary_df, on="experiment_id", how="inner", suffixes=("_tracking", "_summary"))

    columns_to_check = [col for col in tracking_df.columns if col != "experiment_id"]

    different_experiments = merged_df[
        merged_df.apply(lambda row: any(row[f"{col}_tracking"] != row[f"{col}_summary"] for col in columns_to_check), axis=1)
    ][tracking_df.columns]

    experiments_to_update = pd.concat([missing_experiments, different_experiments]).drop_duplicates()

    evaluation_logger.info(f"Experiments to update: {len(experiments_to_update)}")

    return experiments_to_update

def evaluate_experiments(experiments_to_update, predict_dir):
    metrics = ["experiment_id", "auc", "f1_score", "precision", "recall"]
    results = []

    for _, experiment in experiments_to_update.iterrows():
        experiment_id = experiment["index"]
        prediction_file = os.path.join(predict_dir, f"cv_probabilities_{experiment_id}.parquet")

        if not os.path.exists(prediction_file):
            evaluation_logger.warning(f"Prediction file missing: {prediction_file}")
            continue

        try:
            df = pd.read_parquet(prediction_file)


            auc = roc_auc_score(df["y_true"], df["y_pred"])
            f1 = f1_score(df["y_true"], df["y_pred"], average="macro")
            precision = precision_score(df["y_true"], df["y_pred"], average="macro")
            recall = recall_score(df["y_true"], df["y_pred"], average="macro")

            results.append({
                "experiment_id": experiment_id,
                "auc": auc,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
        except Exception as e:
            evaluation_logger.error(f"Error evaluating {experiment_id}: {e}")

    return pd.DataFrame(results, columns=metrics)
