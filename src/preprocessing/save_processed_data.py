from scipy.ndimage import affine_transform
from scipy.spatial.distance import mahalanobis
from src.preprocessing.prepare_data import read_data, filter_data,get_data,transform_data
from config.file_paths import *
from config.hyperparams import *
from src.utils.setup_logger import preprocessing_logger
from src.preprocessing.data_split import *
from src.training.train_utils import load_data_experiment_affine,load_data_experiment_affine_dr
from src.preprocessing.AffineTransform  import SimpleAffineTransform
from src.preprocessing.AffineNetWrapper2  import SimpleMLPTransform
from src.preprocessing.PCALDATransform  import PCALDATransform
#from src.preprocessing.data_visualization  import plot_lda_outliers
import pandas as pd
from scipy.stats import zscore
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.decomposition import PCA
import numpy as np
from src.preprocessing.extension_PLS import PLSCorePoints
import time


def get_pca_lda_pipeline(X, y, n_pca_components=10, lda_components_requested=2, use_shrinkage=True):
    n_classes = len(np.unique(y))
    max_lda_components = min(n_pca_components, n_classes - 1)
    lda_components = min(lda_components_requested, max_lda_components)

    pca = PCA(n_components=n_pca_components)

    if use_shrinkage:
        lda = LDA(n_components=lda_components, solver='lsqr', shrinkage='auto')
    else:
        lda = LDA(n_components=lda_components)

    pipeline = Pipeline([
        ('pca', pca),
        ('lda', lda)
    ])

    return pipeline

def patient_info():
    summary_data = []

    for i in range(1, 52):
        path =get_patient_raw_path(i)
        file_path = os.path.join(path, "challenge_test_report.xlsx")

        if os.path.exists(file_path):
            df = pd.read_excel(file_path)

            if "FEV1" in df.columns:
                above_neg_10 = (df["FEV1"] > -10).sum()
                between_neg_10_20 = ((df["FEV1"] <= -10) & (df["FEV1"] > -20)).sum()
                below_neg_20 = (df["FEV1"] <= -20).sum()

                summary_data.append({
                    "Path": f"P_{i}",
                    "Above -10": above_neg_10,
                    "Between -10 and -20": between_neg_10_20,
                    "Below -20": below_neg_20
                })
            else:
                preprocessing_logger.warning(f"Warning: 'FEV1' column not found in {file_path}")
        else:
            preprocessing_logger.critical(f"File not found: {file_path}")

    summary_df = pd.DataFrame(summary_data)



    Patients = summary_df["Path"].str.extract(r'P_(\d+)')[0].astype(int).tolist()
    Patients_level_3 =summary_df[summary_df["Below -20"] > 0]["Path"].str.extract(r'P_(\d+)')[0].astype(int).tolist()

    preprocessing_logger.info(f"All patients: {Patients}")
    preprocessing_logger.info(f"Patients with -20% FEV1 ranges: {Patients_level_3}")


    return Patients,Patients_level_3

def save_data(Patients,min_diff_Option=None,max_diff_Option=None,min_length_Option=None,max_length_Option=None
              ,remove_level_Option=[['Inhalation']],type=['everyone','independent','probabilistic'],add_3_class=False,title=""):
    # step 1 -SAVE TABLES for all p_ together
    if 'everyone' in type:
        for remove_level in remove_level_Option:
            for min_diff in min_diff_Option:
                for max_diff in max_diff_Option:
                    for min_length in min_length_Option:
                        for max_length in max_length_Option:
                            target_list=[]
                            EEG_df_list=[]
                            clear_p = []
                            for patient in Patients:
                                Patient_NO = 'P_'+str(patient)
                                file_location = get_patient_raw_path(str(patient))

                                Respiratory_cycle_df, data = get_data(eeg_file_name=file_location + r'/EEG_' + Patient_NO + '.txt',
                                                                           times_file_name=file_location + r'/Measuring Time.xlsx',
                                                                           challenge_test_file_name=file_location + r'/challenge_test_report.xlsx')
                                min_FEV1=Respiratory_cycle_df['FEV1'].min()
                                EEG_df,target=filter_data(Respiratory_cycle_df=Respiratory_cycle_df.copy(),EEG_df= data, Percentage_of_next_level=0.2,
                                                                                               breath_type='quiet_breath',remove_class=[''], remove_level=remove_level,
                                                                                               min_diff=min_diff, max_diff=max_diff, min_length=min_length, max_length=max_length)
                                #if sum(target=='FEV1 [-20,-10)')<15:
                                #    clear_p.append(Patient_NO)
                                if sum(target == 'FEV1 [-10,inf)') < 20:
                                    clear_p.append(Patient_NO)
                                if add_3_class: # target=='FEV1 [-20,-10)'
                                    if sum(target=='FEV1 [-20,-10)')<15:
                                        mask = target == 'FEV1 [-10,inf)'
                                        indices = target[mask].index
                                        n_to_change = max(int(0.05 * len(indices)),15)
                                        indices_to_change = indices[-n_to_change:]

                                        target.loc[indices_to_change] = 'FEV1 [-20,-10)'


                                EEG_df['Patient_NO']= Patient_NO
                                target=pd.DataFrame(target)
                                target['Patient_NO']=Patient_NO
                                target=target.reset_index(drop=True)
                                if min_FEV1<-20:
                                    target['binary'] = 1
                                else:
                                    target['binary'] = 0

                                if len(EEG_df['Respiratory cycle'].unique())>1:
                                    target_list.append(target)
                                    EEG_df_list.append(EEG_df)
                                else:
                                    preprocessing_logger.error(f"Error - Only one Respiratory cycle point for patient {patient}")

                            target=pd.concat(target_list)
                            EEG_df=pd.concat(EEG_df_list)

                            EEG_df=transform_data(EEG_df,'model_STFT')
                            EEG_df=EEG_df[0].reset_index()

                            EEG_df=EEG_df.sort_values(by=['Patient_NO', 'Respiratory cycle'])
                            target['colFromIndex'] = target.index
                            target=target.sort_values(by=['Patient_NO','colFromIndex'])
                            target['Respiratory cycle']=EEG_df['Respiratory cycle']
                            target=target.drop(['colFromIndex'], axis=1)
                            target = target.reset_index(drop=True)

                            target = target[~target['Patient_NO'].isin(clear_p)].reset_index(drop=True)
                            EEG_df = EEG_df[~EEG_df['Patient_NO'].isin(clear_p)].reset_index(drop=True)

                            level_mapping = {
                                'FEV1 [-10,inf)': 1,
                                'FEV1 [-20,-10)': 2,
                                'FEV1 [-inf,-20)': 3
                            }
                            target["level_int"] = target["level"].map(level_mapping)

                            preprocessing_logger.info(
                                f"min_diff: {min_diff}, max_diff: {max_diff}, min_length: {min_length}, max_length: {max_length}")

                            base_name = f"min_diff{min_diff}max_diff{max_diff}min_length{min_length}max_length{max_length}remove_level_{remove_level[0]}{title}"

                            target.to_parquet(
                                os.path.join(PROCESSED_DATA_DIR, f'target_{base_name}.parquet'),
                                engine='pyarrow', compression='snappy', index=False
                            )

                            EEG_df.to_parquet(
                                os.path.join(PROCESSED_DATA_DIR, f'EEG_df_{base_name}.parquet'),
                                engine='pyarrow', compression='snappy', index=False
                            )

    if 'independent' in type:
        for patient in Patients:
            for remove_level in remove_level_Option:
                for min_diff in min_diff_Option:
                    for max_diff in max_diff_Option:
                        for min_length in min_length_Option:
                            for max_length in max_length_Option:

                                Patient_NO = 'P_' + str(patient)
                                file_location = get_patient_raw_path(str(patient))
                                Respiratory_cycle_df, fft_data = read_data(file_location + r'/EEG_' + Patient_NO + '.txt',
                                                                           file_location + r'/Measuring Time.xlsx',
                                                                           file_location + r'/challenge_test_report.xlsx')





                                EEG_df, target = filter_data(Respiratory_cycle_df=Respiratory_cycle_df.copy(),
                                                             EEG_df=fft_data[1], Percentage_of_next_level=0.2,
                                                             breath_type='quiet_breath', remove_class=[''],
                                                             remove_level=remove_level,
                                                             min_diff=min_diff, max_diff=max_diff,
                                                             min_length=min_length, max_length=max_length)

                                target.to_frame().to_parquet(PROCESSED_DATA_DIR + r'/Respiratory_cycle_df_' + Patient_NO + '.parquet',
                                engine='pyarrow', compression='snappy', index=False)

                                EEG_df.to_parquet(PROCESSED_DATA_DIR + r'/fft_data_model_STFT_' + Patient_NO + title + '.parquet',
                                                       engine='pyarrow', compression='snappy', index=False)


                                """
                                Respiratory_cycle_df.to_parquet(PROCESSED_DATA_DIR + r'/Respiratory_cycle_df_' + Patient_NO + '.parquet',
                                engine='pyarrow', compression='snappy', index=False)
                                                                
                                fft_data[0].to_parquet(PROCESSED_DATA_DIR + r'/fft_data_model_FFT_' + Patient_NO + title + '.parquet',
                                           engine='pyarrow', compression='snappy', index=False)
                                                       
                                fft_data[1].to_parquet(PROCESSED_DATA_DIR + r'/fft_data_model_STFT_' + Patient_NO + title + '.parquet',
                                                       engine='pyarrow', compression='snappy', index=False)
                    
                                fft_data[2].to_parquet(PROCESSED_DATA_DIR + r'/fft_data_row_data_' + Patient_NO + title + '.parquet',
                                                       engine='pyarrow', compression='snappy', index=False)
                                """


def clean_lda(df, threshold=3, n_components=2):
    lda_cols = [f'col_{i}' for i in range(1, n_components + 1)]

    z_scores = df[lda_cols].apply(zscore)
    mask = (np.abs(z_scores) > threshold).any(axis=1)

    outliers_idx = df.index[mask]
    return outliers_idx

def dimensional_reduction_function(params):
    method = params['dimensional_reduction']
    if method == 'LDA':
        n_components = 2
        model = LDA(n_components=n_components, solver='eigen', shrinkage='auto')
    elif method == 'PLS':
        n_components = 5
        model = PLSRegression(n_components=n_components)
    elif method == 'PLS2':
        n_components = 10
        model = PLSRegression(n_components=n_components)
    elif method == 'PLS3':
        n_components = 30
        model = PLSRegression(n_components=n_components)
    elif method == 'PLSCorePoints':
        n_components = 10
        model = PLSCorePoints(n_components=n_components)
    elif method == 'NCA':
        n_components = 5
        model = NCA(n_components=n_components, random_state=0)
    elif 'PLS_range_' in method :
        n_components = int(method.replace('PLS_range_', ''))
        if n_components>0:
            model = PLSRegression(n_components=n_components)
        else:
            model =''
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    params['n_components'] = n_components
    return dimensional_reduction_preproses(params, model)


def dimensional_reduction_preproses(params,model):
    start_time = time.time()
    n_components=params['n_components']
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X, Y, split_train_test = load_data_experiment_affine(params)
        level_mapping = {
            'FEV1 [-10,inf)': 1,
            'FEV1 [-20,-10)': 2,
            'FEV1 [-inf,-20)': 3
        }
        Y["level_int"] = Y["level"].map(level_mapping)
        x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
        X_after_lda = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_lda = Y.copy()
        for i in range(1, n_components+1):
            X_after_lda.loc[:, f'col_{i}'] = np.nan
        for p_n in Y["Patient_NO"].unique().tolist():
            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_n)].index
            p_indices= split_train_test[split_train_test['Patient_NO']==p_n].index

            model.fit_transform(x_df.loc[train_indices], Y['level_int'].loc[train_indices])
            X_after_lda.loc[p_indices, [f'col_{i}' for i in range(1, n_components+1)]] = model.transform(x_df.loc[p_indices])
            outliers_idx = clean_lda(X_after_lda.loc[p_indices], n_components=n_components)

            outliers_to_remove = []
            outliers_in_train = list(set(outliers_idx).intersection(train_indices))

            for p_n_level in Y.loc[train_indices, "level_int"].unique().tolist():
                level_train_indices = Y.loc[train_indices][Y.loc[train_indices]['level_int'] == p_n_level].index
                level_outliers = list(set(outliers_in_train).intersection(level_train_indices))
                max_to_remove = int(0.3 * len(level_train_indices))
                outliers_to_remove.extend(level_outliers[:max_to_remove])

            X_after_lda = X_after_lda.drop(index=outliers_to_remove)
            Y_after_lda = Y_after_lda.drop(index=outliers_to_remove)
            X = X.drop(index=outliers_to_remove)
            Y = Y.drop(index=outliers_to_remove)


        end_time = time.time()
        elapsed_seconds = end_time - start_time

        filename_base = (
            f"min_diff{params['min_diff']}_max_diff{params['max_diff']}_"
            f"min_length{params['min_length']}_max_length{params['max_length']}_"
            f"{params['dimensional_reduction']}_{cv_i}"
        )

        
        Y_after_lda.to_parquet(
            f"{PROCESSED_DATA_DIR}/target_{filename_base}.parquet",
            engine='pyarrow', compression='snappy'
        )

        X_after_lda.to_parquet(
            f"{PROCESSED_DATA_DIR}/EEG_df_{filename_base}.parquet",
            engine='pyarrow', compression='snappy'
        )

        Y.to_parquet(
            f"{PROCESSED_DATA_DIR}/target_{filename_base}_outliers.parquet",
            engine='pyarrow', compression='snappy'
        )

        X.to_parquet(
            f"{PROCESSED_DATA_DIR}/EEG_df_{filename_base}_outliers.parquet",
            engine='pyarrow', compression='snappy'
        )


        return elapsed_seconds

def learn_affine(source, target):
    ones = np.ones((source.shape[0], 1))
    X_ext = np.hstack([source, ones])
    coeffs, _, _, _ = np.linalg.lstsq(X_ext, target, rcond=None)
    A = coeffs[:2, :].T
    b = coeffs[2, :]
    return A, b


def affine_transform_data(params):
    start_time = time.time()
    _, _2, split_train_test = load_data_experiment_affine(params)
    p_anchor=params['p_anchor']
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X,Y=load_data_experiment_affine_dr(params, cv_i)
        n_components = sum('col_' in col for col in X.columns)
        x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
        dr_cols = [f'lda_{i}' for i in range(1, n_components + 1)]
        X_after_affine = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_affine = Y.copy()
        for p_n in Y["Patient_NO"].unique():
            if p_n == p_anchor:
                continue
            anchor_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_anchor)].index.intersection(x_df.index)
            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_n)].index.intersection(x_df.index)
            all_indices = split_train_test[split_train_test['Patient_NO'] == p_n].index.intersection(x_df.index)

            #model = SimpleAffineTransform()
            model = SimpleMLPTransform(hidden_layer_sizes=(64,), max_iter=1000)


            model.fit_transform(
                X_anchor=x_df.loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject=x_df.loc[train_indices],
                y_subject=Y['level'].loc[train_indices],
                affine_method=params['affine']
            )

            X_affine_train=model.transform(x_df.loc[all_indices])
            X_after_affine.loc[all_indices, dr_cols] = X_affine_train.values

        all_anchor_indices = split_train_test[split_train_test['Patient_NO'] == p_anchor].index.intersection(x_df.index)
        X_after_affine.loc[all_anchor_indices, dr_cols] = x_df.loc[all_anchor_indices].values

        end_time = time.time()
        elapsed_seconds = end_time - start_time

        filename_base = (
            f"min_diff{params['min_diff']}_max_diff{params['max_diff']}_"
            f"min_length{params['min_length']}_max_length{params['max_length']}_"
            f"{p_anchor}_affine_{params['dimensional_reduction']}_affine_{params['affine']}_{cv_i}"
        )

        Y_after_affine.to_parquet(
            f"{PROCESSED_DATA_DIR}/target_{filename_base}.parquet",
            engine='pyarrow', compression='snappy'
        )

        X_after_affine.to_parquet(
            f"{PROCESSED_DATA_DIR}/EEG_df_{filename_base}.parquet",
            engine='pyarrow', compression='snappy'
        )

    return elapsed_seconds





def split_train_test(Patients=[],type=['everyone','independent','affine']):
    if 'everyone' in type:
        df=pd.read_parquet(
            PROCESSED_DATA_DIR + r'/target_min_diff' + str(1.5) + 'max_diff' + str(
                9) + 'min_length' + str(1.5) + 'max_length' + str(
                8) + 'remove_level_' + 'Inhalation' + '' + '.parquet')
        df_split = create_splits(df, 'level')
        df_split.to_parquet(SPLITS_DATA_DIR + r'/split_train_test_min_diff' + str(1.5) + 'max_diff' + str(
                9) + 'min_length' + str(1.5) + 'max_length' + str(
                8) + 'remove_level_' + 'Inhalation' + '' + '.parquet',
                            engine='pyarrow', compression='snappy', index=False)

    elif  'independent' in type:
        for patient in Patients:
            Patient_NO = 'P_' + str(patient)
            df=pd.read_parquet(PROCESSED_DATA_DIR + r'/Respiratory_cycle_df_' + Patient_NO + '.parquet' )
            df_split=create_splits(df,'level')
            df_split.to_parquet(SPLITS_DATA_DIR + r'/split_train_test_' + Patient_NO + '.parquet',
                              engine='pyarrow', compression='snappy', index=False)
    elif 'affine' in type:
        base_name = (
            f"min_diff{params['min_diff']}"
            f"max_diff{params['max_diff']}"
            f"min_length{params['min_length']}"
            f"max_length{params['max_length']}"
            f"remove_level_Inhalationaffine"
        )

        df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, f"target_{base_name}.parquet"))

        df_split = create_splits_affine(df, 'level')

        df_split.to_parquet(
            os.path.join(SPLITS_DATA_DIR, f"split_train_test_{base_name}.parquet"),
            engine='pyarrow', compression='snappy', index=False
        )

def save_time(results):
    save_path = os.path.join(PROCESSED_DATA_DIR, "dimensional_reduction_times.xlsx")
    pd.DataFrame(results).to_excel(save_path, index=False)

def run_pipeline_processed(experiment_types=['mixed', 'independent','probabilistic']):
    Patients, Patients_level_3 = patient_info()

    min_diff_Option=[params['min_diff']]
    max_diff_Option=[params['max_diff']]
    min_length_Option=[params['min_length']]
    max_length_Option=[params['max_length']]
    remove_level_Option=[['Inhalation']]

    for experiment_type in experiment_types:
        if experiment_type == 'mixed':
            save_data(Patients, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,
                      remove_level_Option, type=['everyone'], title="")
            split_train_test(type=['everyone'])

        elif experiment_type == 'independent':
            save_data(Patients_level_3, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,
                      remove_level_Option, type=['independent'], title="")
            split_train_test(Patients_level_3, type=['independent'])

        elif experiment_type == 'probabilistic':
            save_data(Patients_level_3, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,
                      remove_level_Option, type=['everyone'], title="probabilistic")

        elif experiment_type == 'affine':
            dr_time, affine_time = '', ''
            #save_data(Patients_level_3, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,remove_level_Option, type=['everyone'],add_3_class=True, title="affine")
            #split_train_test(type=['affine'])
            #dr_time= dimensional_reduction_function(params)
            affine_time= affine_transform_data(params)
            return dr_time,affine_time



"""
Save independent:
    save_data(Patients_level_3,min_diff_Option,max_diff_Option,min_length_Option,max_length_Option,type=['independent'],title="")
    
Save mixed:
    save_data(Patients, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option, remove_level_Option,type=['everyone'], title="")
    
Save mixed for selected patient:
    Patients = [5,6,7]
    save_data(Patients, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option, remove_level_Option,type=['everyone'], title="p5_p6_p7")
    
Options for remove_level_Option:
    remove_level_Option=[['Inhalation'],['metacholin']]
    
min_diff_Option=[1.5]
max_diff_Option=[9,15]
min_length_Option=[1.5]
max_length_Option=[8,15]
"""

