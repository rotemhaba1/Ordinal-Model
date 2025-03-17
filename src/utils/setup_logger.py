import logging
import os

def setup_logger(log_name, log_file, log_level=logging.INFO):
    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger(log_name)
    logger.setLevel(log_level)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def log_experiment(logger, experiment_type, experiment_id, model, combo):
    formatted_experiment_type = f"{experiment_type:<12}"
    formatted_experiment_id = f"ID-{experiment_id:<2}"
    formatted_model = f"{model:<21}"
    formatted_combo = f"{combo}"
    logger.info(f"Experiment: {formatted_experiment_type} | {formatted_experiment_id} | Model: {formatted_model} | Hyperparameters: {formatted_combo}")

training_logger = setup_logger("training", "logs/training.log", logging.INFO)
preprocessing_logger = setup_logger("preprocessing", "logs/preprocessing.log", logging.INFO)
evaluation_logger = setup_logger("evaluation", "logs/evaluation.log", logging.WARNING)


