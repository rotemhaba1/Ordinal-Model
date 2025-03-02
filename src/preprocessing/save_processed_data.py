from prepare_data import read_data, filter_data,get_data,transform_data
from config.file_paths import *
from src.utils.setup_logger import preprocessing_logger
from data_split import *

def patient_info():
    BASE_DIR = os.getcwd()
    summary_data = []

    for i in range(1, 51):
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
              ,remove_level_Option=[['Inhalation']],type=['everyone','independent'],title=""):
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


def split_train_test(Patients=[],type=['everyone','independent']):
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


Patients,Patients_level_3=patient_info()

min_diff_Option=[1.5]
max_diff_Option=[9]
min_length_Option=[1.5]
max_length_Option=[8]
remove_level_Option=[['Inhalation']]
#save_data(Patients_level_3, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option,remove_level_Option,type=['independent'], title="")
#save_data(Patients, min_diff_Option, max_diff_Option, min_length_Option, max_length_Option, remove_level_Option,type=['everyone'], title="")

split_train_test(type=['everyone'])
split_train_test(Patients_level_3,type=['independent'])

