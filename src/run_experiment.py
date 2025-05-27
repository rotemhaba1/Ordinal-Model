from Cython import nonecheck
import itertools
from  src.preprocessing.save_processed_data import run_pipeline_processed,save_time
from  src.training.train_model import run_in_sequence
from  src.validation.validation import run_pipeline_validation


if __name__ == "__main__":

    experiment_types=['affine'] # 'probabilistic', 'independent','mixed','probabilistic','probabilistic_step_2','affine'
    #run_pipeline_processed(experiment_types)
    #run_in_sequence(experiment_types)
    #run_pipeline_validation(experiment_types)

    from config.hyperparams import params
    from src.preprocessing.save_processed_data import run_pipeline_processed,save_time
    from src.training.train_model import run_in_sequence
    from src.validation.validation import run_pipeline_validation


    if __name__ == "__main__":
        affine_transform_opt=[True,False]
        dimensional_reduction_op = [True, False]
        dimensional_reduction_name_op = ['PLS_range_','FULL_with_AFFINE_PLS_range_']
        affine_op = [None,'SMOTE']
        n_components_range = range(2, 5)

        experiment_types = ['affine']
        results_times = []
        for n_components in n_components_range:
            params['dimensional_reduction'] = f'PLS_range_{n_components}'
            dr_time, affine_time = run_pipeline_processed(experiment_types)
            results_times.append({
                "i": n_components,
                "dimensional_reduction": params['dimensional_reduction'],
                "dr_time_seconds": dr_time,
                "affine_time_seconds": affine_time
            })

        save_time(results_times)


        for affine_transform, dim_reduction_transform, dim_reduction_name,affine,n_components \
                in itertools.product(
                affine_transform_opt,
                dimensional_reduction_op,
                dimensional_reduction_name_op,affine_op,n_components_range):
            print(affine_transform, dim_reduction_transform, dim_reduction_name,affine,n_components )
            if n_components < 2 :
                continue
            params['dimensional_reduction'] = f'{dim_reduction_name}{n_components}'
            params['affine_transform'] = affine_transform
            params['affine'] = affine
            params['dimensional_reduction_transform'] = dim_reduction_transform
            print(f"\n--- Running pipeline with dimensional_reduction = {params['dimensional_reduction']} ---")
            run_in_sequence(experiment_types)

        run_pipeline_validation(experiment_types)

