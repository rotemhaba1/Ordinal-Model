from scipy.ndimage import affine_transform

from src.preprocessing.prepare_data import read_data, filter_data,get_data,transform_data
from config.file_paths import *
from config.hyperparams import *
from src.utils.setup_logger import preprocessing_logger
from src.preprocessing.data_split import *
from src.training.train_utils import load_data_experiment_affine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from src.preprocessing.AffineTransform  import SimpleAffineTransform
#from src.preprocessing.data_visualization  import plot_lda_outliers

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
                                if add_3_class: # target=='FEV1 [-20,-10)'
                                    if sum(target=='FEV1 [-20,-10)')<15:
                                        mask = target == 'FEV1 [-10,inf)'
                                        indices = target[mask].index
                                        n_to_change = int(0.05 * len(indices))
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

                            preprocessing_logger.info(
                                f"min_diff: {min_diff}, max_diff: {max_diff}, min_length: {min_length}, max_length: {max_length}")
                            target.to_parquet(
                                PROCESSED_DATA_DIR + r'/target_min_diff' + str(min_diff) + 'max_diff' + str(
                                    max_diff) + 'min_length' + str(min_length) + 'max_length' + str(
                                    max_length) + 'remove_level_' + remove_level[0] + title + '.parquet',
                                engine='pyarrow', compression='snappy', index=False)

                            EEG_df.to_parquet(
                                PROCESSED_DATA_DIR + r'/EEG_df_min_diff' + str(min_diff) + 'max_diff' + str(
                                    max_diff) + 'min_length' + str(min_length) + 'max_length' + str(
                                    max_length) + 'remove_level_' + remove_level[0] + title + '.parquet',
                                engine='pyarrow', compression='snappy', index=False)

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


def clean_lda(df, threshold=3):
    from scipy.stats import zscore
    z1 = zscore(df['lda_1'])
    z2 = zscore(df['lda_2'])

    outliers_idx = df.index[(np.abs(z1) > threshold) | (np.abs(z2) > threshold)]

    return outliers_idx

def clean_mahalanobis_outliers(df, train_indices, threshold=3):
    from scipy.spatial.distance import mahalanobis
    train_data = df.loc[train_indices, ['lda_1', 'lda_2']].dropna().values

    cov = np.cov(train_data, rowvar=False)
    inv_covmat = np.linalg.inv(cov)
    mean_vec = np.mean(train_data, axis=0)

    test_data = df[['lda_1', 'lda_2']].dropna()
    dists = np.array([mahalanobis(row, mean_vec, inv_covmat) for row in test_data.values])

    outliers_idx = test_data.index[dists > threshold]

    return outliers_idx

def dimensional_reduction_LDA(params):
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X, Y, split_train_test = load_data_experiment_affine(params)
        x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
        X_after_lda = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_lda = Y.copy()
        X_after_lda.loc[:,'lda_1'] = np.nan
        X_after_lda.loc[:,'lda_2'] = np.nan
        for p_n in Y["Patient_NO"].unique().tolist():
            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_n)].index
            p_indices= split_train_test[split_train_test['Patient_NO']==p_n].index
            lda = LDA(n_components=2)
            lda.fit_transform(x_df.loc[train_indices], Y['level'].loc[train_indices])
            X_after_lda.loc[p_indices, ['lda_1', 'lda_2']] = lda.transform(x_df.loc[p_indices])
            outliers_idx=clean_lda(X_after_lda.loc[p_indices])
            X_after_lda = X_after_lda.drop(index=outliers_idx)
            Y_after_lda = Y_after_lda.drop(index=outliers_idx)
            X = X.drop(index=outliers_idx)
            Y = Y.drop(index=outliers_idx)

        Y_after_lda.to_parquet(
            PROCESSED_DATA_DIR + r'/target_min_diff' + str(params['min_diff']) + 'max_diff' +str(params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length']) +  'lda_' +cv_i+ '.parquet',
            engine='pyarrow', compression='snappy')

        X_after_lda.to_parquet(
            PROCESSED_DATA_DIR + r'/EEG_df_min_diff' + str(params['min_diff']) + 'max_diff' +str(params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length'])  +  'lda_' +cv_i+ '.parquet',
            engine='pyarrow', compression='snappy')

        Y.to_parquet(
            PROCESSED_DATA_DIR + r'/target_min_diff' + str(params['min_diff']) + 'max_diff' +str(params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length']) +  'outliers_lda_' +cv_i+ '.parquet',
            engine='pyarrow', compression='snappy')

        X.to_parquet(
            PROCESSED_DATA_DIR + r'/EEG_df_min_diff' + str(params['min_diff']) + 'max_diff' +str(params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length'])  +  'outliers_lda_' +cv_i+ '.parquet',
            engine='pyarrow', compression='snappy')

def learn_affine(source, target):
    ones = np.ones((source.shape[0], 1))
    X_ext = np.hstack([source, ones])
    coeffs, _, _, _ = np.linalg.lstsq(X_ext, target, rcond=None)
    A = coeffs[:2, :].T
    b = coeffs[2, :]
    return A, b

def affine_transform_data(params):
    _, _2, split_train_test = load_data_experiment_affine(params)
    p_anchor=params['p_anchor']
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X= pd.read_parquet(os.path.join(PROCESSED_DATA_DIR + r'/EEG_df_min_diff' + str(params['min_diff']) + 'max_diff' + str(
                params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length']) + 'lda_' + cv_i + '.parquet'))
        Y= pd.read_parquet(os.path.join(PROCESSED_DATA_DIR + r'/target_min_diff' + str(params['min_diff']) + 'max_diff' + str(
                params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length']) + 'lda_' + cv_i + '.parquet'))
        x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
        X_after_affine = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_affine = Y.copy()
        for p_n in Y["Patient_NO"].unique():
            if p_n == p_anchor:
                continue
            anchor_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_anchor)].index.intersection(x_df.index)

            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO']==p_n)].index.intersection(x_df.index)

            test_indices = split_train_test[(split_train_test[f'{cv_i}'] == False) & (split_train_test['Patient_NO']==p_n)].index.intersection(x_df.index)

            all_indices = split_train_test[split_train_test['Patient_NO'] == p_n].index.intersection(x_df.index)

            model = SimpleAffineTransform()

            model.fit_transform(
                X_anchor=x_df.loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject=x_df.loc[train_indices],
                y_subject=Y['level'].loc[train_indices]
            )

            X_affine_train=model.transform(x_df.loc[all_indices])

            """
            model.plot_affine_alignment(
                X_anchor=x_df[['lda_1', 'lda_2']].loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject=x_df[['lda_1', 'lda_2']].loc[train_indices],
                y_subject=Y['level'].loc[train_indices],
                X_subject_test=model.transform(x_df.loc[test_indices]),
                y_subject_test=Y['level'].loc[test_indices],
                title=f"Affine Transform | Patient {p_n}"
            )
            """


            X_after_affine.loc[all_indices, ['lda_1', 'lda_2']] = X_affine_train.values

        all_anchor_indices = split_train_test[split_train_test['Patient_NO'] == p_anchor].index.intersection(x_df.index)
        X_after_affine.loc[all_anchor_indices, ['lda_1', 'lda_2']] = x_df.loc[all_anchor_indices].values

        Y_after_affine.to_parquet(
            PROCESSED_DATA_DIR + r'/target_min_diff' + str(params['min_diff']) + 'max_diff' + str(
                params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length']) + '_'+p_anchor+'affine_' + cv_i + '.parquet',
            engine='pyarrow', compression='snappy')

        X_after_affine.to_parquet(
            PROCESSED_DATA_DIR + r'/EEG_df_min_diff' + str(params['min_diff']) + 'max_diff' + str(
                params['max_diff']) + 'min_length'
            + str(params['min_length']) + 'max_length' + str(params['max_length'])+ '_'+p_anchor+'affine_' + cv_i + '.parquet',
            engine='pyarrow', compression='snappy')




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
        df=pd.read_parquet(
            PROCESSED_DATA_DIR + r'/target_min_diff' + str(params['min_diff']) + 'max_diff' + str(
                params['max_diff']) + 'min_length' + str(params['min_length']) + 'max_length' + str(
                params['max_length']) + 'remove_level_' + 'Inhalation' + 'affine' + '.parquet')
        df_split = create_splits_affine(df, 'level')
        df_split.to_parquet(SPLITS_DATA_DIR + r'/split_train_test_min_diff' + str(params['min_diff']) + 'max_diff' + str(params['max_diff']) + 'min_length' + str(params['min_length'])
                            + 'max_length' + str(params['max_length']) + 'remove_level_' + 'Inhalation' + 'affine' + '.parquet',
                            engine='pyarrow', compression='snappy', index=False)

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
            save_data(Patients_level_3, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,remove_level_Option, type=['everyone'],add_3_class=True, title="affine")
            split_train_test(type=['affine'])
            dimensional_reduction_LDA(params)
            affine_transform_data(params)



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

