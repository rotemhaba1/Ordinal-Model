from prepare_data import *
from config.file_paths import *
from src.training.train_utils import load_data_experiment_affine
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from src.preprocessing.AffineTransform  import SimpleAffineTransform
from config.hyperparams import *
from src.preprocessing.save_processed_data  import *

def apply_plot_colored_signals_stacked():
    patient = 51
    Patient_NO = 'P_' + str(patient)
    rolling_flag = False
    file_location = get_patient_raw_path(str(patient))
    Respiratory_cycle_df, data = get_data(eeg_file_name=file_location + r'/EEG_' + Patient_NO + '.txt',
                                          times_file_name=file_location + r'/Measuring Time.xlsx',
                                          challenge_test_file_name=file_location + r'/challenge_test_report.xlsx',
                                          rolling_flag=rolling_flag)
    min_diff = 1.5
    max_diff = 9
    min_length = 1.5
    max_length = 8

    min_diff = 0.2
    max_diff = 9
    min_length = 0.2
    max_length = 8

    data = data[~data['Respiration'].isna()]
    data['quiet_breath'] = np.where(
        (data['min max diff Respiratory cycle'] >= min_diff) & (data['min max diff Respiratory cycle'] <= max_diff)
        & (data['Length Respiratory cycle'] >= min_length) & (data['Length Respiratory cycle'] <= max_length)
        , '1', '0')
    print(f"{Patient_NO} rolling_flag {rolling_flag}")
    print(data[['Respiratory cycle', 'quiet_breath']].drop_duplicates(keep='first')['quiet_breath'].value_counts())

    plot_colored_signals_stacked(data[(data['eeg_seconds'] > 1200) & (data['eeg_seconds'] < 1500)])
    print("")

def plot_colored_signals_stacked(df):
    signals = ['Respiration', 'EEG (.5 - 35 Hz)', 'EEG (.5 - 35 Hz).1']
    time = df['eeg_seconds'].values
    quiet = df['quiet_breath'].astype(str).values

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 8), sharex=True)

    for idx, signal in enumerate(signals):
        ax = axes[idx]
        y = df[signal].values

        points = np.array([time, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = np.where(quiet[1:] == '1', 'green', 'red')

        lc = LineCollection(segments, colors=colors, linewidths=1)
        ax.add_collection(lc)

        ax.set_xlim(time[0], time[-1])
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
        ax.set_ylabel(signal)
        ax.grid(True)

    axes[-1].set_xlabel('Time (seconds)')
    plt.tight_layout()
    plt.show()

def apply_affine_transform_data():
    affine_transform_data(params)
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



            model.plot_before_after(
                X_anchor=x_df[['lda_1', 'lda_2']].loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject_test_before=x_df[['lda_1', 'lda_2']].loc[test_indices],
                y_subject_test_before=Y['level'].loc[test_indices],
                X_subject_test_after=model.transform(x_df.loc[test_indices]),
                y_subject_test_after=Y['level'].loc[test_indices],
                X_subject_train_after=model.transform(x_df.loc[train_indices]),
                y_subject_train_after=Y['level'].loc[train_indices],
                title=f"Affine Transform | Patient {p_n}"
            )


def apply_plot_lda_outliers():
    X, Y, split_train_test = load_data_experiment_affine(params)
    x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X_after_lda = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_lda = Y.copy()
        X_after_lda.loc[:, 'lda_1'] = np.nan
        X_after_lda.loc[:, 'lda_2'] = np.nan
        for p_n in Y["Patient_NO"].unique().tolist():
            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO'] == p_n)].index
            p_indices = split_train_test[split_train_test['Patient_NO'] == p_n].index
            lda = LDA(n_components=2)
            lda.fit_transform(x_df.loc[train_indices], Y['level'].loc[train_indices])
            X_after_lda.loc[p_indices, ['lda_1', 'lda_2']] = lda.transform(x_df.loc[p_indices])
            outliers_idx = clean_lda(X_after_lda.loc[p_indices], threshold=3)
            #outliers_idx = clean_mahalanobis_outliers(X_after_lda.loc[p_indices], train_indices, threshold=3)
            plot_lda_outliers(X_after_lda,outliers_idx,title=p_n)

def plot_lda_outliers(df, outliers_idx,title=''):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))

    # Normal points
    plt.scatter(df['lda_1'], df['lda_2'], label='Normal Data', alpha=0.5)

    # Outliers
    plt.scatter(df.loc[outliers_idx, 'lda_1'], df.loc[outliers_idx, 'lda_2'],
                color='red', label='Outliers')

    plt.xlabel('lda_1')
    plt.ylabel('lda_2')
    plt.title(f'LDA Outlier Detection {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # apply_plot_colored_signals_stacked()
    apply_affine_transform_data()
    # apply_plot_lda_outliers()

