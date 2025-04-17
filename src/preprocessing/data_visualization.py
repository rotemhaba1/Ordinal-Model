from prepare_data import *
from config.file_paths import *
from src.training.train_utils import load_data_experiment_affine
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from src.preprocessing.AffineTransform  import SimpleAffineTransform
from src.preprocessing.AffineNetWrapper  import SimpleMLPTransform
from config.hyperparams import *
from src.preprocessing.save_processed_data  import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_decomposition import PLSRegression
import pandas as pd





def balanced_core_points(X, y):
    core_idx = []
    min_class_size = min((y == label).sum() for label in np.unique(y))

    for label in np.unique(y):
        X_label = X[y == label]
        nbrs = NearestNeighbors(n_neighbors=5).fit(X_label)
        distances, _ = nbrs.kneighbors()
        densities = distances.sum(axis=1)
        top_k_idx = np.argsort(densities)[:min_class_size]
        core_idx.extend(X_label.iloc[top_k_idx].index)

    return core_idx


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

            model = SimpleAffineTransform()
            model = SimpleMLPTransform(hidden_layer_sizes=(64,), max_iter=500)


            model.fit_transform(
                X_anchor=x_df.loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject=x_df.loc[train_indices],
                y_subject=Y['level'].loc[train_indices]
            )

            X_anchor = x_df[['lda_1', 'lda_2']].loc[anchor_indices]
            y_anchor = Y['level'].loc[anchor_indices]
            X_subject_test_before = x_df[['lda_1', 'lda_2']].loc[test_indices]
            y_subject_test_before = Y['level'].loc[test_indices]
            X_subject_test_after = model.transform(x_df.loc[test_indices])
            y_subject_test_after = Y['level'].loc[test_indices].reset_index(drop=True)
            X_subject_train_after = model.transform(x_df.loc[train_indices])
            y_subject_train_after = Y['level'].loc[train_indices]

            clf_before = LogisticRegression(max_iter=1000)
            clf_before.fit(X_subject_test_before, y_subject_test_before)
            acc_before = accuracy_score(y_subject_test_before, clf_before.predict(X_subject_test_before))

            clf_after = LogisticRegression(max_iter=1000)
            clf_after.fit(X_subject_test_after, y_subject_test_after)
            acc_after = accuracy_score(y_subject_test_after, clf_after.predict(X_subject_test_after))


            print(f"Accuracy before transform: {acc_before:.2f}")
            print(f"Accuracy after transform: {acc_after:.2f}")




            model.plot_before_after(
                X_anchor=x_df[['lda_1', 'lda_2']].loc[anchor_indices],
                y_anchor=Y['level'].loc[anchor_indices],
                X_subject_test_before=x_df[['lda_1', 'lda_2']].loc[test_indices],
                y_subject_test_before=Y['level'].loc[test_indices],
                X_subject_test_after=model.transform(x_df.loc[test_indices]),
                y_subject_test_after=Y['level'].loc[test_indices],
                X_subject_train_after=model.transform(x_df.loc[train_indices]),
                y_subject_train_after=Y['level'].loc[train_indices],
                title=f"Affine Transform | Patient {p_n}, Accuracy before: {acc_before:.2f}, Accuracy after: {acc_after:.2f} "
            )



            print("")



def evaluate_lda_success(X_after_lda_p, Y_p):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Drop non-LDA columns if exist
    X_lda = X_after_lda_p[['lda_1', 'lda_2']].values
    y = Y_p.values

    # Simple Logistic Regression on LDA components
    clf = LogisticRegression()
    acc = cross_val_score(clf, X_lda, y, cv=3, scoring='accuracy').mean()

    print(f"LDA Success (cross-validated accuracy): {acc:.3f}")

    return acc


def apply_plot_lda_outliers():
    X, Y, split_train_test = load_data_experiment_affine(params)
    level_mapping = {
        'FEV1 [-10,inf)': 1,
        'FEV1 [-20,-10)': 2,
        'FEV1 [-inf,-20)': 3
    }
    Y["level_int"] = Y["level"].map(level_mapping)
    x_df = X.drop(columns=[col for col in ["Patient_NO", 'Respiratory cycle'] if col in X.columns])
    for cv_i in ['cv_1', 'cv_2', 'cv_3', 'cv_4', 'cv_5']:
        X_after_lda = X[["Patient_NO", 'Respiratory cycle']]
        Y_after_lda = Y.copy()
        X_after_lda.loc[:, 'lda_1'] = np.nan
        X_after_lda.loc[:, 'lda_2'] = np.nan
        for p_n in Y["Patient_NO"].unique().tolist():
            train_indices = split_train_test[(split_train_test[f'{cv_i}'] == True) & (split_train_test['Patient_NO'] == p_n)].index
            p_indices = split_train_test[split_train_test['Patient_NO'] == p_n].index
            test_indices = split_train_test[(split_train_test[f'{cv_i}'] == False) & (split_train_test['Patient_NO'] == p_n)].index

            core_indices = balanced_core_points(x_df.loc[train_indices], Y['level'].loc[train_indices])
            print(Y['level'].loc[core_indices].value_counts())

            lda = LDA(n_components=2)
            lda = PLSRegression(n_components=10)
            lda.fit_transform(x_df.loc[train_indices], Y['level_int'].loc[train_indices])
            #lda.fit_transform(x_df.loc[core_indices], Y['level'].loc[core_indices])
            X_after_lda.loc[p_indices, ['lda_1', 'lda_2']] = lda.transform(x_df.loc[p_indices])
            outliers_idx = clean_lda(X_after_lda.loc[p_indices], threshold=3)

            acc_with_outliers=evaluate_lda_success(X_after_lda.loc[test_indices], Y['level'].loc[test_indices])
            test_indices_without_outliers =[idx for idx in test_indices if idx not in outliers_idx]
            acc_without_outliers=evaluate_lda_success(X_after_lda.loc[test_indices_without_outliers], Y['level'].loc[test_indices_without_outliers])

            plot_lda_outliers(X_after_lda.loc[p_indices],Y_after_lda.loc[p_indices]['level'],outliers_idx,title=f'{p_n} , Acc : '
                                                                            f'with outliers {round(acc_with_outliers * 100)}%'
                                                                            f'without outliers {round(acc_without_outliers * 100)}%')
            print("next")


def plot_lda_outliers(df, Y, outliers_idx, title=''):
    outliers_idx = np.array(outliers_idx).flatten()
    mask_non_outliers = ~df.index.isin(outliers_idx)
    level_mapping = {
        'FEV1 [-10,inf)': 1,
        'FEV1 [-20,-10)': 2,
        'FEV1 [-inf,-20)': 3
    }
    Y_non_outliers_encoded = Y[mask_non_outliers].map(level_mapping)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(df['lda_1'], df['lda_2'], label='Normal Data', alpha=0.5)
    axes[0].scatter(df.loc[outliers_idx, 'lda_1'], df.loc[outliers_idx, 'lda_2'], color='red', label='Outliers')
    axes[0].set_title(f'LDA Outlier Detection\n{title}')
    axes[0].set_xlabel('lda_1')
    axes[0].set_ylabel('lda_2')
    axes[0].legend()
    axes[0].grid(True)

    scatter = axes[1].scatter(
        df.loc[mask_non_outliers, 'lda_1'],
        df.loc[mask_non_outliers, 'lda_2'],
        c=Y_non_outliers_encoded,
        cmap='Set1',
        alpha=0.7
    )
    legend1 = axes[1].legend(*scatter.legend_elements(), title="Class")
    axes[1].add_artist(legend1)
    axes[1].set_title('Label Distribution (without outliers)')
    axes[1].set_xlabel('lda_1')
    axes[1].set_ylabel('lda_2')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()





if __name__ == '__main__':

    # apply_plot_colored_signals_stacked()
    # apply_affine_transform_data()
    apply_plot_lda_outliers()

