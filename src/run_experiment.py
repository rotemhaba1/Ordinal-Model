
from  src.preprocessing.save_processed_data import run_pipeline_processed,save_time
from  src.training.train_model import run_in_sequence
from  src.validation.validation import run_pipeline_validation


if __name__ == "__main__":

    experiment_types=['affine'] # 'probabilistic', 'independent','mixed','probabilistic','probabilistic_step_2','affine'
    #run_pipeline_processed(experiment_types)
    #run_in_sequence(experiment_types)
    run_pipeline_validation(experiment_types)

    from config.hyperparams import params
    from src.preprocessing.save_processed_data import run_pipeline_processed,save_time
    from src.training.train_model import run_in_sequence
    from src.validation.validation import run_pipeline_validation


    if __name__ == "__main__":
        if False:
            experiment_types = ['affine']  # can include: 'probabilistic', 'independent', etc.
            results_times = []
            for i in range(6,9): # range(2,101):
                if i == 1:
                    continue
                params['dimensional_reduction'] = f'PLS_range_{i}'
                params['dimensional_reduction'] = f'all_PLS_range_{i}'

                print(f"\n--- Running pipeline with dimensional_reduction = {params['dimensional_reduction']} ---")
                """
                dr_time,affine_time=run_pipeline_processed(experiment_types)
                results_times.append({
                    "i": i,
                    "dimensional_reduction": params['dimensional_reduction'],
                    "dr_time_seconds": dr_time,
                    "affine_time_seconds": affine_time
                })
    
            save_time(results_times)
    
                """
                for affine_transform_flag in [True]:
                    params['affine_transform'] = affine_transform_flag
                    for  dimensional_reduction_transform_flag in [False]:
                        params['dimensional_reduction_transform'] = dimensional_reduction_transform_flag
                        #if (affine_transform_flag==True) & (dimensional_reduction_transform_flag==False):
                            #continue
                        run_in_sequence(experiment_types)

            run_pipeline_validation(experiment_types)


