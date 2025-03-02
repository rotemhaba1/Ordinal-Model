import logging
import os


def setup_logger(log_name, log_file):
    os.makedirs("logs", exist_ok=True)  # Ensure logs folder exists

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if not logger.handlers:
        # File handler (writes to a file)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler (shows logs in PyCharm)
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')  # Shorter format for console
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


# Create loggers
training_logger = setup_logger("training", "logs/training.log")
preprocessing_logger = setup_logger("preprocessing", "logs/preprocessing.log")
evaluation_logger = setup_logger("evaluation", "logs/evaluation.log")
