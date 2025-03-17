
from  src.preprocessing.save_processed_data import run_pipeline_processed
from  src.training.train_model import run_in_sequence
from  src.validation.validation import run_pipeline_validation


if __name__ == "__main__":
    experiment_types=['mixed']
    run_pipeline_processed(experiment_types)
    run_in_sequence(experiment_types)
    run_pipeline_validation(experiment_types)