import pandas as pd
import numpy as np
from scipy.signal import stft
import re
import os
np.random.seed(42)

def get_data(eeg_file_name, times_file_name, challenge_test_file_name):
    '''

    :param eeg_file_name: EEG_P1.txt
    :param times_file_name: Measuring Time.xlsx
    :param times_file_name: challenge_test_report.xlsx
    :return:
    1.Add times to EEG signals
    2.Combines operations with data
    3.Respiratory cycles
    4.Percentage of the level

    Column explanation:
        'Respiration' -A breath belt signal
        , 'EEG (.5 - 35 Hz)' -EEG signal
        , 'EEG (.5 - 35 Hz).1' - EEG signal
        , 'eeg_seconds' -The second measurement of the signal
        ,'step' -{'beginning', 'spirometry', 'start', 'end'}
        , 'level' -{'base', 'Saline', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5','Inhalation'}
        , 'taken' -If 1 the spirometry was taken
        , 'time' -The time the researcher measured as the beginning of step and level
        , 'start_seconds' - The start_seconds of step and level
        , 'id'
        , 'end_seconds' - The end_seconds of step and level
        ,'Respiratory cycle' -Respiratory cycle by positive and negative area distribution
        , 'NUM'
        , 'marker'
        , 'Length Respiratory cycle'
        ,'max_Respiratory_cycle'- the max value in this Respiratory_cycle
        ,'min_Respiratory_cycle' - the min value in this Respiratory_cycle
        ,'min max diff Respiratory cycle'
        , 'quiet_breath'
        , 'force_breath'
        , 'rank_by_Respiratory_cycle' -In what percentage is the respiratory cycle
        , 'rank_by_seconds' -In what percentage is the respiratory cycle


    '''

    # ---------------------------------------
    # ------1.Add times to EEG signals------
    # --------------------------------------
    if os.path.exists(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.txt', eeg_file_name)):
        eeg_data = pd.read_csv(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.txt', eeg_file_name), delimiter="\t", low_memory=False)
    elif os.path.exists(eeg_file_name):
        eeg_data = pd.read_csv(eeg_file_name, delimiter="\t", low_memory=False)
    elif re.sub(r'EEG_P_(\d+)\.txt', r'P\1.xlsx', eeg_file_name):
        eeg_data = pd.read_excel(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.xlsx', eeg_file_name))


    eeg_data = eeg_data[['Respiration', 'EEG (.5 - 35 Hz)', 'EEG (.5 - 35 Hz).1']]
    freq = 1000
    N = eeg_data.shape[0]
    t = np.linspace(0, (1 / freq) * (N), N)
    eeg_data['eeg_seconds'] = t

    # ---------------------------------------
    # ---2.Combines operations with data----
    # --------------------------------------
    times_data = pd.read_excel(times_file_name)
    times_data = times_data[times_data['time'].notna()]
    times_data['id'] = 1
    times_data['end_seconds'] = times_data.groupby('id').seconds.shift(-1)
    times_data['end_seconds'] = times_data['end_seconds'].fillna(np.inf)
    times_data = times_data.rename(columns={"seconds": "start_seconds"})

    a = eeg_data.eeg_seconds.values
    bh = times_data.end_seconds.values
    bl = times_data.start_seconds.values

    i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))

    df = pd.DataFrame(
        np.column_stack([eeg_data.values[i], times_data.values[j]]),
        columns=eeg_data.columns.append(times_data.columns)
    )

    # ---------------------------------------
    # --------3. Respiratory cycles---------
    # --------------------------------------
    challenge_test_data = pd.read_excel(challenge_test_file_name)
    df = df.merge(challenge_test_data, on='level')


    # ---------------------------------------
    # --------4. Respiratory cycles---------
    # --------------------------------------

    df['Respiratory cycle'] = np.where(df['Respiration'] >= 0, 1, -1)
    df['NUM'] = (df['Respiratory cycle'] != df['Respiratory cycle'].shift()).cumsum()
    df['Respiratory cycle'] = np.where(df['Respiration'] >= 0, df['NUM'], df['NUM'] - 1)
    df['Respiratory cycle'] = (df['Respiratory cycle'] / 2).astype(int) + 1
    df.loc[:, 'marker'] = (df.loc[:, 'Respiratory cycle'] == df.loc[:, 'Respiratory cycle'].shift(-1))

    df['Length Respiratory cycle'] = df.groupby('Respiratory cycle')['eeg_seconds'].transform('max') - \
                                     df.groupby('Respiratory cycle')['eeg_seconds'].transform('min')
    df['min_Respiratory_cycle'] = df.groupby('Respiratory cycle')['Respiration'].transform('min')
    df['max_Respiratory_cycle'] = df.groupby('Respiratory cycle')['Respiration'].transform('max')
    df['min max diff Respiratory cycle'] = df['max_Respiratory_cycle'] - df['min_Respiratory_cycle']

    df['eeg_seconds2']=df['eeg_seconds']
    Respiratory_cycle_df=df.groupby(['Respiratory cycle']).agg({ 'step': 'first',
                                            'level': 'first',
                                            'eeg_seconds':'min',
                                             'eeg_seconds2': 'max',
                                             'FEV1': 'first',
                                            'min max diff Respiratory cycle': 'min',
                                            'Length Respiratory cycle': 'min',
                                             'max_Respiratory_cycle': 'min',
                                           }).reset_index()
    Respiratory_cycle_df = Respiratory_cycle_df.rename(columns={'eeg_seconds': 'start_seconds', 'eeg_seconds2': 'end_seconds'})


    return Respiratory_cycle_df,df

def transform_data(df,signal_model):
    # ---------------------------------------
    # ---- 5.fourier_transformations --------
    # ---------------------------------------

    data_to_fft=[]
    data_signal1 = df.groupby(['Patient_NO','Respiratory cycle'])['EEG (.5 - 35 Hz)'].apply(list)
    data_to_fft_signal1 = fourier_transformations(EEG_df=data_signal1, signal_model=signal_model)

    data_signal2 = df.groupby(['Patient_NO','Respiratory cycle'])['EEG (.5 - 35 Hz).1'].apply(list)
    data_to_fft_signal2 = fourier_transformations(EEG_df=data_signal2, signal_model=signal_model)

    data_to_fft.append(data_to_fft_signal1.merge(data_to_fft_signal2, on=['Patient_NO','Respiratory cycle']))
    return data_to_fft

def filter_data(Respiratory_cycle_df,EEG_df,Percentage_of_next_level,breath_type,remove_class,remove_level=['Inhalation'],min_diff=2,max_diff=7,min_length=2,max_length=7):
    """

    :param Respiratory_cycle_df: all the data on Respiratory_cycle
    :param EEG_df: the variebls
    :param Percentage_of_next_level: treshold of x% from next level
    :param breath_type: quiet,force,all
    :param algorithm:
    :param level_type:
    :param smote: True/False
    :param CV:
    :param min_diff:
    :param max_diff:
    :param min_length:
    :param max_length:
    :return:
    """
    if 'metacholin' in remove_level:
        remove_level=['stage1', 'stage2', 'stage3', 'stage4', 'stage5', 'stage6','Inhalation']
    Respiratory_cycle_df = level_steps(Respiratory_cycle_df, [-10, -20],remove_level)
    Respiratory_cycle_df=Respiratory_types(Respiratory_cycle_df, min_diff, max_diff, min_length, max_length)
    Respiratory_cycle_df = Percentage_level(Respiratory_cycle_df,Percentage_of_next_level)
    if 'quiet_breath' in breath_type:
        Respiratory_cycle_df=Respiratory_cycle_df[Respiratory_cycle_df['quiet_breath']=='1']
    elif 'force_breath'  in breath_type:
        Respiratory_cycle_df = Respiratory_cycle_df[Respiratory_cycle_df['force_breath'] == '1']

    Respiratory_cycle_df=Respiratory_cycle_df[~Respiratory_cycle_df['level'].isin(remove_level)]
    Respiratory_cycle_df = Respiratory_cycle_df[~Respiratory_cycle_df['level'].isin(remove_class)]
    target = Respiratory_cycle_df.groupby('Respiratory cycle')['level'].agg(lambda x: x.value_counts().index[0])

    filter=Respiratory_cycle_df['Respiratory cycle'].unique()
    if 'Respiratory cycle' in EEG_df.columns:
        EEG_df=EEG_df[EEG_df['Respiratory cycle'].isin(filter)]
    else:
        EEG_df = EEG_df[EEG_df.index.isin(filter)]

    return EEG_df,target

def read_data(eeg_file_name, times_file_name, challenge_test_file_name):
    '''

    :param eeg_file_name: EEG_P1.txt
    :param times_file_name: Measuring Time.xlsx
    :param times_file_name: challenge_test_report.xlsx
    :return:
    1.Add times to EEG signals
    2.Combines operations with data
    3.Respiratory cycles
    4.Percentage of the level

    Column explanation:
        'Respiration' -A breath belt signal
        , 'EEG (.5 - 35 Hz)' -EEG signal
        , 'EEG (.5 - 35 Hz).1' - EEG signal
        , 'eeg_seconds' -The second measurement of the signal
        ,'step' -{'beginning', 'spirometry', 'start', 'end'}
        , 'level' -{'base', 'Saline', 'stage1', 'stage2', 'stage3', 'stage4', 'stage5','Inhalation'}
        , 'taken' -If 1 the spirometry was taken
        , 'time' -The time the researcher measured as the beginning of step and level
        , 'start_seconds' - The start_seconds of step and level
        , 'id'
        , 'end_seconds' - The end_seconds of step and level
        ,'Respiratory cycle' -Respiratory cycle by positive and negative area distribution
        , 'NUM'
        , 'marker'
        , 'Length Respiratory cycle'
        ,'max_Respiratory_cycle'- the max value in this Respiratory_cycle
        ,'min_Respiratory_cycle' - the min value in this Respiratory_cycle
        ,'min max diff Respiratory cycle'
        , 'quiet_breath'
        , 'force_breath'
        , 'rank_by_Respiratory_cycle' -In what percentage is the respiratory cycle
        , 'rank_by_seconds' -In what percentage is the respiratory cycle


    '''

    # ---------------------------------------
    # ------1.Add times to EEG signals------
    # --------------------------------------
    if os.path.exists(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.txt', eeg_file_name)):
        eeg_data = pd.read_csv(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.txt', eeg_file_name), delimiter="\t", low_memory=False)
    elif os.path.exists(eeg_file_name):
        eeg_data = pd.read_csv(eeg_file_name, delimiter="\t", low_memory=False)
    elif re.sub(r'EEG_P_(\d+)\.txt', r'P\1.xlsx', eeg_file_name):
        eeg_data = pd.read_excel(re.sub(r'EEG_P_(\d+)\.txt', r'P\1.xlsx', eeg_file_name))


    eeg_data = eeg_data[['Respiration', 'EEG (.5 - 35 Hz)', 'EEG (.5 - 35 Hz).1']]
    freq = 1000
    N = eeg_data.shape[0]
    t = np.linspace(0, (1 / freq) * (N), N)
    eeg_data['eeg_seconds'] = t

    # ---------------------------------------
    # ---2.Combines operations with data----
    # --------------------------------------
    times_data = pd.read_excel(times_file_name)
    times_data = times_data[times_data['time'].notna()]
    times_data['id'] = 1
    times_data['end_seconds'] = times_data.groupby('id').seconds.shift(-1)
    times_data['end_seconds'] = times_data['end_seconds'].fillna(np.inf)
    times_data = times_data.rename(columns={"seconds": "start_seconds"})

    a = eeg_data.eeg_seconds.values
    bh = times_data.end_seconds.values
    bl = times_data.start_seconds.values

    i, j = np.where((a[:, None] >= bl) & (a[:, None] <= bh))

    df = pd.DataFrame(
        np.column_stack([eeg_data.values[i], times_data.values[j]]),
        columns=eeg_data.columns.append(times_data.columns)
    )

    # ---------------------------------------
    # --------3. Respiratory cycles---------
    # --------------------------------------
    challenge_test_data = pd.read_excel(challenge_test_file_name)
    df = df.merge(challenge_test_data, on='level')


    # ---------------------------------------
    # --------4. Respiratory cycles---------
    # --------------------------------------

    df['Respiratory cycle'] = np.where(df['Respiration'] >= 0, 1, -1)
    df['NUM'] = (df['Respiratory cycle'] != df['Respiratory cycle'].shift()).cumsum()
    df['Respiratory cycle'] = np.where(df['Respiration'] >= 0, df['NUM'], df['NUM'] - 1)
    df['Respiratory cycle'] = (df['Respiratory cycle'] / 2).astype(int) + 1
    df.loc[:, 'marker'] = (df.loc[:, 'Respiratory cycle'] == df.loc[:, 'Respiratory cycle'].shift(-1))

    df['Length Respiratory cycle'] = df.groupby('Respiratory cycle')['eeg_seconds'].transform('max') - \
                                     df.groupby('Respiratory cycle')['eeg_seconds'].transform('min')
    df['min_Respiratory_cycle'] = df.groupby('Respiratory cycle')['Respiration'].transform('min')
    df['max_Respiratory_cycle'] = df.groupby('Respiratory cycle')['Respiration'].transform('max')
    df['min max diff Respiratory cycle'] = df['max_Respiratory_cycle'] - df['min_Respiratory_cycle']

    df['eeg_seconds2']=df['eeg_seconds']
    Respiratory_cycle_df=df.groupby(['Respiratory cycle']).agg({ 'step': 'first',
                                            'level': 'first',
                                            'eeg_seconds':'min',
                                             'eeg_seconds2': 'max',
                                             'FEV1': 'first',
                                            'min max diff Respiratory cycle': 'min',
                                            'Length Respiratory cycle': 'min',
                                             'max_Respiratory_cycle': 'min',
                                           }).reset_index()
    Respiratory_cycle_df = Respiratory_cycle_df.rename(columns={'eeg_seconds': 'start_seconds', 'eeg_seconds2': 'end_seconds'})

    # ---------------------------------------
    # ---- 5.fourier_transformations --------
    # ---------------------------------------
    data_to_fft=[]
    for signal_model in ['model_FFT','model_STFT','row_data']:
        data_signal1 = df.groupby('Respiratory cycle')['EEG (.5 - 35 Hz)'].apply(list)
        data_to_fft_signal1 = fourier_transformations(EEG_df=data_signal1, signal_model=signal_model)

        data_signal2 = df.groupby('Respiratory cycle')['EEG (.5 - 35 Hz).1'].apply(list)
        data_to_fft_signal2 = fourier_transformations(EEG_df=data_signal2, signal_model=signal_model)

        data_to_fft.append(data_to_fft_signal1.merge(data_to_fft_signal2, on='Respiratory cycle'))

    return Respiratory_cycle_df,data_to_fft

def model_FFT(signal, freq=1000):
    import scipy.fftpack
    N = signal.shape[0]
    T = 1.0 / freq
    yf = abs(scipy.fftpack.fft(signal))

    return yf

def model_STFT(signal, freq=1000):
    f, t, Zxx = stft(signal, fs=freq, nperseg=256, noverlap=128)
    yf = np.abs(Zxx).flatten()
    return yf

def level_steps(df, bounders=[], Except_levels=[]):
    """
    df=Respiratory_cycle_df
     bounders=[-10, -20]
     Except_levels=remove_level

    """
    df=df[~df['level'].isin(Except_levels)]
    bounders.sort()
    upper_bound = -np.inf
    if len(bounders) > 0:
        for i in bounders:
            df['level'] = np.where((~df['level'].isin(Except_levels) & (df['FEV1'] < i) & (df['FEV1'] >= upper_bound)),
                                   'FEV1 [' + str(upper_bound) + ',' + str(i) + ')', df['level'])
            # x_start=df[df['level'].isin(['FEV1 ['+str(upper_bound)+','+str(i)+')'])]['statrs_seconds'].min()
            x_end = df[df['level'].isin(['FEV1 [' + str(upper_bound) + ',' + str(i) + ')'])]['end_seconds'].min()
            df['step'] = np.where(
                (df['level'].isin(['FEV1 [' + str(upper_bound) + ',' + str(i) + ')'])) & (df['end_seconds'] > x_end),
                'spirometry', df['step'])
            upper_bound = i
        df['level'] = np.where((~df['level'].isin(Except_levels) & (df['FEV1'] < np.inf) & (df['FEV1'] >= upper_bound)),
                               'FEV1 [' + str(upper_bound) + ',' + str(np.inf) + ')', df['level'])
    return df

def Respiratory_types(df,min_diff=2,max_diff=7,min_length=2,max_length=7):
    df['quiet_breath'] = np.where(
        (df['min max diff Respiratory cycle'] >= min_diff) & (df['min max diff Respiratory cycle'] <= max_diff)
        & (df['Length Respiratory cycle'] >= min_length) & (df['Length Respiratory cycle'] <= max_length)
        , '1', '0')

    df['force_breath'] = np.where(
        (df['min max diff Respiratory cycle'] >= 8) & (df['min max diff Respiratory cycle'] <= 50)
        & (df['Length Respiratory cycle'] >= 4) & (df['Length Respiratory cycle'] <= 10) & (
                    df['max_Respiratory_cycle'] >= 2.5)
        , '1', '0')
    return df

def fourier_transformations(EEG_df, signal_model):
    # Complements zeros by the max len
    max_length = max(len(x) for x in EEG_df)
    EEG_df = EEG_df.apply(lambda x: np.pad(x, (0, max_length - len(x)), 'constant'))
    if signal_model == 'model_FFT':
        EEG_df2 = EEG_df.apply(lambda x: model_FFT(x))
    elif signal_model == 'model_STFT':
        EEG_df2 = EEG_df.apply(lambda x: model_STFT(x))
    elif signal_model == 'row_data':
        EEG_df2 = EEG_df
    EEG_df2 = pd.DataFrame(EEG_df2.tolist(), index=EEG_df2.index)
    return EEG_df2

def Percentage_level(df,Percentage_of_next_level):
    """
    the base is 120 sec
    :param df:
    :param Percentage_of_next_level:
    :return:
    """
    level_list = df['level'].unique().tolist()
    df['level_index'] = df['level'].apply(lambda x: level_list.index(x))
    df['start_level'] = df.groupby('level')['start_seconds'].transform('min')
    df["level2"] = np.where((df["start_seconds"]>=df['start_level']+(120*Percentage_of_next_level)), df["level"],df['level_index'].apply(lambda x: level_list[x - 1]))
    df["level"]=np.where(df['level'] == level_list[0], df["level"],df["level2"])
    df.drop(['level_index', 'start_level','level2'], axis='columns', inplace=True)
    return df




