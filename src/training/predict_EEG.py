import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
try:
    from imblearn.over_sampling import SMOTE , ADASYN
except:
    ('failed from imblearn.over_sampling import SMOTE')
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
import ast
from imblearn.over_sampling import ADASYN
np.random.seed(42)

'''
Step 1

Step 2

'''

def predict_models_by_sequence(predict_model_sequence):
    if predict_model_sequence=='DecisionTrees':
        model = predict_models(predict_model=predict_model_sequence, class_weight='balanced', alpha=0, WIGR_power=0.2,criterion='entropy')
    elif predict_model_sequence=='XGBoost':
        model = predict_models(predict_model=predict_model_sequence)
    elif predict_model_sequence == 'RandomForest_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, WIGR_power='entropy',criterion='entropy')
    elif predict_model_sequence == 'AdaBoost_SAMME_R_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, WIGR_power=0.2,criterion='WIGR_EV')
    elif predict_model_sequence == 'AdaBoost_half_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, WIGR_power=0.2,criterion='WIGR_EV')
    elif predict_model_sequence == 'AdaBoost_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, WIGR_power='entropy',criterion='entropy')
    elif predict_model_sequence == 'AdaBoost_SAMME_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, criterion='entropy')
    elif predict_model_sequence == 'RandomForest':
        model = predict_models(predict_model=predict_model_sequence, class_weight='balanced')
    elif predict_model_sequence == 'DecisionTrees_Ordinal':
        model = predict_models(predict_model=predict_model_sequence, WIGR_power=0.2,criterion='WIGR_EV')
    elif predict_model_sequence == 'catboost':
        model = predict_models(predict_model=predict_model_sequence)
    elif predict_model_sequence == 'AdaBoost':
        model = predict_models(predict_model=predict_model_sequence,)
    return model


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def convert_int_to_true_fales(x):
    if (type(x) == float):
        if x == 1:
            x = True
        else:
            x = False
    return x

def predict_models(predict_model='DecisionTrees',alpha=0, WIGR_power={}, criterion={},class_weight=None):
    """
    models option
        DecisionTrees
        RandomForest
        LogisticRegression
        GaussianNB
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier
    try:
        from catboost import CatBoostClassifier
    except:
        pass


    if predict_model == 'DecisionTrees':
        from sklearn import tree
        clf_model = tree.DecisionTreeClassifier(class_weight=class_weight)
    elif predict_model == 'SVC':
        from sklearn import svm
        clf_model = svm.SVC(class_weight=class_weight)
    elif predict_model == 'RandomForest':
        clf_model = RandomForestClassifier()
    elif predict_model == 'LogisticRegression':
        from sklearn.linear_model import LogisticRegression
        clf_model = LogisticRegression(random_state=0,solver = 'lbfgs',max_iter=100 )
    elif predict_model == 'GaussianNB':
        from sklearn.naive_bayes import GaussianNB
        clf_model = GaussianNB()
    elif predict_model == 'AdaBoost':
        clf_model = AdaBoostClassifier()
    elif predict_model == 'catboost':
        clf_model = CatBoostClassifier(silent=True)
    elif predict_model == 'catboost_ordinal':
        clf_model = CatBoostClassifier(silent=True,
                                    eval_metric='AUC',
                                    custom_metric='AUC:type=Mu;misclass_cost_matrix=0/0.5/2/1/0/1/0/0.5/0',
                                    loss_function='MultiClass',
                                    train_dir='model_dir',
                                    random_seed=42)
    elif predict_model == 'XGBRegressor':
        import xgboost as xgb
        clf_model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
    elif predict_model == 'XGBoost':
        from xgboost import XGBClassifier
        clf_model = XGBClassifier()
    elif predict_model == 'KMeans':
        from sklearn.cluster import KMeans
        clf_model = KMeans( random_state=0)
    elif predict_model == 'LogisticIT_ordinal':
        from mord import LogisticIT
        clf_model = LogisticIT(alpha=alpha)
    elif predict_model == 'LogisticAT_ordinal':
        from mord import LogisticAT
        clf_model = LogisticAT(alpha=alpha)
    elif predict_model == 'LogisticSE_ordinal':
        from mord import LogisticSE
        clf_model = LogisticSE(alpha=alpha)
    elif predict_model == 'DecisionTrees_Ordinal':
        if criterion == 'entropy':
            clf_model = DecisionTreeClassifier(criterion=criterion,class_weight=class_weight)
        else:
            clf_model = DecisionTreeClassifier(criterion=criterion, WIGR_power=WIGR_power,class_weight=class_weight)
    elif predict_model == 'RandomForest_Ordinal':
        if criterion == 'entropy':
            clf_model = RandomForestClassifier(criterion=criterion,class_weight=class_weight)
        else:
            clf_model = RandomForestClassifier(criterion=criterion, WIGR_power=WIGR_power,class_weight=class_weight)
    elif predict_model == 'AdaBoost_SAMME_R_Ordinal':
        if criterion == 'entropy':
            clf_model = AdaBoostClassifier(DecisionTreeClassifier(criterion=criterion, max_depth=1,class_weight=class_weight))
        else:
            clf_model = AdaBoostClassifier(
                DecisionTreeClassifier(criterion=criterion, WIGR_power=WIGR_power, max_depth=1,class_weight=class_weight))
    elif predict_model == 'AdaBoost_SAMME_Ordinal':
        if criterion == 'entropy':
            clf_model = AdaBoostClassifier(DecisionTreeClassifier(criterion=criterion, max_depth=1,class_weight=class_weight), algorithm='SAMME')
        else:
            clf_model = AdaBoostClassifier(
                DecisionTreeClassifier(criterion=criterion, WIGR_power=WIGR_power, max_depth=1,class_weight=class_weight),
                algorithm='SAMME')
    elif predict_model == 'AdaBoost_Ordinal':
        if criterion == 'entropy':
            clf_model = AdaBoostClassifier(DecisionTreeClassifier(criterion=criterion, random_state=1, max_depth=1,class_weight=class_weight),
                                     Ordinal_problem=1)
        else:
            clf_model = AdaBoostClassifier(
                DecisionTreeClassifier(criterion=criterion, WIGR_power=WIGR_power, max_depth=1,class_weight=class_weight),
                Ordinal_problem=1)
    elif predict_model == 'AdaBoost_half_Ordinal':
        if criterion == 'entropy':
            clf_model = AdaBoostClassifier(DecisionTreeClassifier(criterion=criterion, random_state=1, max_depth=1,class_weight=class_weight),
                                     Ordinal_problem=2)
        else:
            clf_model = AdaBoostClassifier(
                DecisionTreeClassifier(criterion=criterion, WIGR_power=WIGR_power, max_depth=1,class_weight=class_weight),
                Ordinal_problem=2)
    return clf_model

def get_summary_tables(y_predict_proba,y_test,y_train,matrix_cost=[1,2,3]):
    classes_ = y_train.unique().tolist()

    y_pred = np.argmax(y_predict_proba, axis=1) + 1
    y_pred = np.where(y_pred == 1, matrix_cost[0], np.where(y_pred == 2, matrix_cost[1], matrix_cost[2]))
    Accuracy = metrics.accuracy_score(y_test, y_pred)
    ################################
    yy_test = label_binarize(y_test, classes=classes_)
    yy_train = label_binarize(y_train, classes=classes_)
    if len(classes_) == 2:  # binary
        yy_test = np.hstack((yy_test, 1 - yy_test))
        yy_train = np.hstack((yy_train, 1 - yy_train))
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(classes_)):
        fpr[i], tpr[i], _ = roc_curve(yy_test[:, i], y_predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    ###################
    try:
        auc_value=roc_auc_score(y_test, y_predict_proba, multi_class='ovo', average='weighted')
    except:
        auc_value=roc_auc_score(roc_auc_score(y_test, y_pred, average='weighted'))
    confusion_matrix_=confusion_matrix(y_test, y_pred)
    return Accuracy,roc_auc,auc_value,confusion_matrix_

def probability_matrix(df_test_sequence,last_period=3):
    df_test_sequence.columns = df_test_sequence.columns.map(str)
    df_test_sequence = df_test_sequence.rename(columns={1: 'Patient_NO', 3: 'target_int', 4: 'preb1', 5: 'preb2', 6: 'preb3',
                                                        '1': 'Patient_NO', '3': 'target_int', '4': 'preb1', '5': 'preb2', '6': 'preb3'}, inplace=False)


    for col_ in ['preb1', 'preb2', 'preb3']:
        for i in range(1, 1 + last_period):
            df_test_sequence[col_ + ' t-'+str(i)] = df_test_sequence.groupby('Patient_NO')[col_].shift(i)

    df_test_sequence = df_test_sequence.dropna()

    col_name=[col for col in df_test_sequence.columns if 'preb' in col]
    df_test_sequence_X = df_test_sequence[col_name]
    df_test_sequence_y = df_test_sequence['target_int']
    return df_test_sequence_X,df_test_sequence_y

def print_records_summary(data,target):
    try:
        print("------------------")
        print(data.columns)
    except:
        pass
    try:
        print("------------------")
        print(target.columns)
        print("------------------")

    except:
        target = pd.DataFrame(target).reset_index()
        print(target.columns)
    try:
        print(data.shape)
        print(target.shape)
    except:
        pass

    try:
        print(target['level'].value_counts())
        print(target['Patient_NO'].value_counts())
        print(target.groupby(["Patient_NO", "level"]).size())
    except:
        pass
    try:
        print(target['target_int'].value_counts())
        print(target['Patient_NO'].value_counts())
        print(target.groupby(["Patient_NO", "target_int"]).size())
    except:
        pass
    try:
        print(data['Patient_NO'].value_counts())
    except:
        pass
def predict_probability(data, target, predict_model,majority_algorithm
                        ,predict_model_sequence,smote_,adasyn_,class_weight
                        , group_all, type_level, downsampling,alpha, criterion
                        ,WIGR_power, matrix_cost,majority_top_algorithm_option,how,CV_splits,last_period):
    if how=='sequence':
        target_names = target['level'].unique().tolist()
        target_names.sort()
        target['target_int'] = target['level'].apply(lambda x: target_names.index(x) + 1)
        Patient_NO_test = target.Patient_NO.unique()

        df_test_sequence = pd.DataFrame()
        for P in Patient_NO_test:
            if type_level == 'level':
                P = [P]
            X_train, X_test = data[~data['Patient_NO'].isin(P)][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy(),data[data['Patient_NO'].isin(P)][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy()
            y_train, y_test = target[~target['Patient_NO'].isin(P)].copy(), target[target['Patient_NO'].isin(P)]['target_int'].copy()
            y_test_sequence = target[target['Patient_NO'].isin(P)].copy()
            ##
            X_train_m, X_test_m=X_train, X_test
            y_train_m, y_test_m=y_train, y_test
            if 'majority' in predict_model:
                y_predict_proba_list = []
                try:
                    majority_algorithm = ast.literal_eval(majority_algorithm)
                except:
                    pass
                for predict in majority_algorithm:
                    X_train, X_test = X_train_m, X_test_m
                    y_train, y_test = y_train_m, y_test_m
                    downsampling = convert_int_to_true_fales(majority_top_algorithm_option[predict]['downsampling'])
                    if downsampling:
                        train = y_train.reset_index()
                        min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                        try:
                            train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                        except:
                            train = train.groupby(['Patient_NO', 'level']).head(min_group)
                        y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                        X_train = X_train[X_train.index.isin(train['index'].unique())]
                    smote_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['smote_list'])
                    if smote_list:
                        X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
                    adasyn_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['adasyn_list'])
                    if adasyn_list:
                        ada = ADASYN()
                        X_train, y_train = ada.fit_resample(X_train, y_train['target_int'])

                    clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                         , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                         , majority_top_algorithm_option[predict]['criterion']
                                         , majority_top_algorithm_option[predict]['class_weight'])

                    clf = clf.fit(X_train, y_train)
                    y_predict_proba_list.append(pd.DataFrame(data=clf.predict_proba(X_test)))
                y_predict_proba = pd.concat(y_predict_proba_list)
                if predict_model == 'majority_min':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                elif predict_model == 'majority_max':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                elif predict_model == 'majority_avg_all':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                elif predict_model == 'majority_top_avg_all':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                elif predict_model == 'majority_avg_max_grp':
                    y_predict_proba['max_grp'] = y_predict_proba.idxmax(axis=1)

                    y_predict_proba['max_grp_common'] = y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(
                        lambda x: x.value_counts().index[0])
                    y_predict_proba = y_predict_proba[y_predict_proba['max_grp'] == y_predict_proba['max_grp_common']]
                    y_predict_proba = y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()

                y_predict_proba = y_predict_proba.to_numpy()
            else:
                if downsampling:
                    train = y_train.reset_index()
                    min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                    try:
                        train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                    except:
                        train = train.groupby(['Patient_NO', 'level']).head(min_group)
                    y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                    X_train = X_train[X_train.index.isin(train['index'].unique())]

                if smote_:
                    X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
                if adasyn_:
                    ada = ADASYN()
                    X_train, y_train = ada.fit_resample(X_train, y_train['target_int'])
                clf = predict_models(predict_model, alpha, float(WIGR_power), criterion,class_weight)
                y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)
            '''
            if downsampling:
                train = y_train.reset_index()
                min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                try:
                    train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                except:

                    train = train.groupby(['Patient_NO', 'level']).head(min_group)
                y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                X_train = X_train[X_train.index.isin(train['index'].unique())]
            if smote:
                X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
            if 'majority' in predict_model:
                y_predict_proba_list=[]
                try:
                    majority_algorithm = ast.literal_eval(majority_algorithm)
                except:
                    pass
                for predict in majority_algorithm:
                    clf = predict_models(predict, alpha, float(WIGR_power), criterion,class_weight)
                    df = pd.DataFrame(data=clf.fit(X_train, y_train['target_int']).predict_proba(X_test))
                    y_predict_proba_list.append(df)
                y_predict_proba = pd.concat(y_predict_proba_list)
                if predict_model=='majority_min':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                elif predict_model=='majority_max':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                elif predict_model == 'majority_avg_all':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                elif predict_model == 'majority_avg_max_grp':
                    y_predict_proba['max_grp']=y_predict_proba.idxmax(axis=1)
                    y_predict_proba['max_grp_common']=y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(lambda x:x.value_counts().index[0])
                    y_predict_proba=y_predict_proba[y_predict_proba['max_grp']==y_predict_proba['max_grp_common']]
                    y_predict_proba=y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                y_predict_proba=y_predict_proba.to_numpy()
            else:
                clf = predict_models(predict_model, alpha, float(WIGR_power), criterion,class_weight)
                y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)
            '''

            y_predict_proba1=pd.DataFrame(y_predict_proba.tolist(),columns=['preb1', 'preb2', 'preb3'])
            y_predict_proba1=y_predict_proba1.reset_index(drop=True)
            y_test_sequence = y_test_sequence.reset_index(drop=True)
            y_predict_proba1=pd.concat([y_test_sequence, y_predict_proba1], axis=1,ignore_index=True)
            df_test_sequence = df_test_sequence.append(y_predict_proba1,ignore_index=True)

        df_test_sequence = df_test_sequence.rename(columns={1: 'Patient_NO', 3: 'target_int'
            , 4: 'preb1', 5: 'preb2', 6: 'preb3'}, inplace=False)
        df_before_sequence=df_test_sequence.copy()
        for col_ in ['preb1', 'preb2', 'preb3']:
            df_test_sequence[col_ + ' t-1'] = df_test_sequence.groupby('Patient_NO')[col_].shift(1)
            df_test_sequence[col_ + ' t-2'] = df_test_sequence.groupby('Patient_NO')[col_].shift(2)
            df_test_sequence[col_ + ' t-3'] = df_test_sequence.groupby('Patient_NO')[col_].shift(3)

        df_test_sequence = df_test_sequence.dropna()
        df_test_sequence_X = df_test_sequence[['preb1', 'preb2', 'preb3', 'preb1 t-1', 'preb1 t-2', 'preb1 t-3',
                                               'preb2 t-1', 'preb2 t-2', 'preb2 t-3', 'preb3 t-1', 'preb3 t-2',
                                               'preb3 t-3']]
        df_test_sequence_y = df_test_sequence['target_int']

        Y_test_sheet,y_predict_proba_sheet=[],[]
        cv = StratifiedKFold(n_splits=CV_splits, random_state=42, shuffle=True)
        count_sheet=0
        X, Y=df_test_sequence_X.reset_index(drop=True), df_test_sequence_y.reset_index(drop=True)
        for train_idx, test_idx in cv.split(X, Y):
            df_test_sequence_X, df_test_sequence_y = X.loc[train_idx], Y.loc[train_idx]
            df_test_sequence_X_test, df_test_sequence_y_test = X.loc[test_idx], Y.loc[test_idx]

            clf = predict_models(predict_model=predict_model_sequence, class_weight='balanced', alpha=0, WIGR_power=0.2,
                                  criterion='entropy')
            clf.fit(df_test_sequence_X, df_test_sequence_y)
            y_predict_proba = clf.predict_proba(df_test_sequence_X_test)
            Y_test_sheet.append(df_test_sequence_y_test)
            y_predict_proba_sheet.append(y_predict_proba)

        return Y_test_sheet,y_predict_proba_sheet,df_before_sequence
    if how=='sequence_new':
        save_data_path=r'C:\Users\rotem\OneDrive - Bar-Ilan University - Students\EEG & methacholine test\Python\data_after_preprocessing\sequence'
        target_names = target['level'].unique().tolist()
        target_names.sort()
        target['target_int'] = target['level'].apply(lambda x: target_names.index(x) + 1)
        Patient_NO_test = target.Patient_NO.unique()
        ## data for sequence
        data_main,target_main=data.copy(),target.copy()
        df_test_sequence = pd.DataFrame()
        df_before_sequence = pd.DataFrame()
        Y_test_sheet, y_predict_proba_sheet = [], []
        for Pj in Patient_NO_test:
            data,target=data_main[~data_main['Patient_NO'].isin([Pj])].copy(),target_main[~target_main['Patient_NO'].isin([Pj])].copy()
            try:
                df_test_sequence=pd.read_csv(save_data_path+'\df_test_sequence' + predict_model + Pj + '.csv')
            except:
                df_test_sequence = pd.DataFrame()
                for Pi in Patient_NO_test:
                    if Pj==Pi:
                        continue
                    #if type_level == 'level':
                    #    Pi = [Pi]
                    X_train, X_test = data[~data['Patient_NO'].isin([Pi])][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy(),data[data['Patient_NO'].isin([Pi])][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy()
                    y_train, y_test = target[~target['Patient_NO'].isin([Pi])].copy(), target[target['Patient_NO'].isin([Pi])]['target_int'].copy()
                    y_test_sequence = target[target['Patient_NO'].isin([Pi])].copy()
                    ##
                    X_train_m, X_test_m=X_train, X_test
                    y_train_m, y_test_m=y_train, y_test
                    if 'majority' in predict_model:
                        y_predict_proba_list = []
                        try:
                            majority_algorithm = ast.literal_eval(majority_algorithm)
                        except:
                            pass
                        for predict in majority_algorithm:
                            X_train, X_test = X_train_m, X_test_m
                            y_train, y_test = y_train_m, y_test_m
                            downsampling = convert_int_to_true_fales(majority_top_algorithm_option[predict]['downsampling'])
                            if downsampling:
                                train = y_train.reset_index()
                                min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                                try:
                                    train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                                except:
                                    train = train.groupby(['Patient_NO', 'level']).head(min_group)
                                y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                                X_train = X_train[X_train.index.isin(train['index'].unique())]
                            smote_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['smote_list'])
                            if smote_list:
                                X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])

                            adasyn_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['adasyn_list'])
                            if adasyn_list:
                                ada = ADASYN()
                                X_train, y_train = ada.fit_resample(X_train, y_train['target_int'])


                            clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                                 , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                                 , majority_top_algorithm_option[predict]['criterion']
                                                 , majority_top_algorithm_option[predict]['class_weight'])

                            clf = clf.fit(X_train, y_train)
                            y_predict_proba_list.append(pd.DataFrame(data=clf.predict_proba(X_test)))
                        y_predict_proba = pd.concat(y_predict_proba_list)
                        if predict_model == 'majority_min':
                            y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                        elif predict_model == 'majority_max':
                            y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                        elif predict_model == 'majority_avg_all':
                            y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                        elif predict_model == 'majority_top_avg_all':
                            y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                        elif predict_model == 'majority_avg_max_grp':
                            y_predict_proba['max_grp'] = y_predict_proba.idxmax(axis=1)

                            y_predict_proba['max_grp_common'] = y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(
                                lambda x: x.value_counts().index[0])
                            y_predict_proba = y_predict_proba[y_predict_proba['max_grp'] == y_predict_proba['max_grp_common']]
                            y_predict_proba = y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                            y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()

                        y_predict_proba = y_predict_proba.to_numpy()
                    else:
                        if downsampling:
                            train = y_train.reset_index()
                            min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                            try:
                                train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                            except:
                                train = train.groupby(['Patient_NO', 'level']).head(min_group)
                            y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                            X_train = X_train[X_train.index.isin(train['index'].unique())]

                        if smote_:
                            X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
                        if adasyn_:
                            ada = ADASYN()
                            X_train, Y_train = ada.fit_resample(X_train, y_train['target_int'])
                        clf = predict_models(predict_model, alpha, float(WIGR_power), criterion,class_weight)
                        try:
                            y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)
                        except:
                            print("falied predict_model")
                            clf = predict_models('AdaBoost_Ordinal', 1, float('0'), 'entropy', '')
                            y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)


                    y_predict_proba1=pd.DataFrame(y_predict_proba.tolist(),columns=['preb1', 'preb2', 'preb3'])
                    y_predict_proba1=y_predict_proba1.reset_index(drop=True)
                    y_test_sequence = y_test_sequence.reset_index(drop=True)
                    y_predict_proba1=pd.concat([y_test_sequence, y_predict_proba1], axis=1,ignore_index=True)
                    df_test_sequence = df_test_sequence.append(y_predict_proba1,ignore_index=True)
                df_test_sequence.to_csv(save_data_path+'\df_test_sequence' + predict_model + Pj + '.csv', index=False)

            df_test_sequence_X,df_test_sequence_y=probability_matrix(df_test_sequence,last_period)
            X, Y = df_test_sequence_X.reset_index(drop=True), df_test_sequence_y.reset_index(drop=True)
            clf_dt =predict_models_by_sequence(predict_model_sequence)
            clf_dt.fit(X, Y)
            try:
                y_predict_proba = pd.read_csv(save_data_path + '\df_test_sequence_self' + predict_model + Pj + '.csv')
                y_test_sequence = pd.read_csv(save_data_path + '\y_test_sequence_self' + predict_model + Pj + '.csv')
            except:
                data, target = data_main.copy(), target_main.copy()
                #if type_level == 'level':
                #    Pi = [Pj]
                X_train, X_test = data[~data['Patient_NO'].isin([Pj])][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy(), \
                                  data[data['Patient_NO'].isin([Pj])][data.columns.difference(['Patient_NO', 'Respiratory cycle'])].copy()
                y_train, y_test = target[~target['Patient_NO'].isin([Pj])].copy(), \
                                  target[target['Patient_NO'].isin([Pj])]['target_int'].copy()
                y_test_sequence = target[target['Patient_NO'].isin([Pj])].copy()
                y_test_sequence.to_csv(save_data_path + '\y_test_sequence_self' + predict_model + Pj + '.csv',index=False)
                ##
                X_train_m, X_test_m = X_train, X_test
                y_train_m, y_test_m = y_train, y_test
                if 'majority' in predict_model:
                    y_predict_proba_list = []
                    try:
                        majority_algorithm = ast.literal_eval(majority_algorithm)
                    except:
                        pass
                    for predict in majority_algorithm:
                        X_train, X_test = X_train_m, X_test_m
                        y_train, y_test = y_train_m, y_test_m
                        downsampling = convert_int_to_true_fales(majority_top_algorithm_option[predict]['downsampling'])
                        if downsampling:
                            train = y_train.reset_index()
                            min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                            try:
                                train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                            except:
                                train = train.groupby(['Patient_NO', 'level']).head(min_group)
                            y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                            X_train = X_train[X_train.index.isin(train['index'].unique())]
                        smote_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['smote_list'])
                        if smote_list:
                            X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
                        adasyn_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['adasyn_list'])
                        if adasyn_list:
                            ada = ADASYN()
                            X_train, y_train = ada.fit_resample(X_train, y_train['target_int'])


                        clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                             , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                             , majority_top_algorithm_option[predict]['criterion']
                                             , majority_top_algorithm_option[predict]['class_weight'])

                        clf = clf.fit(X_train, y_train)
                        y_predict_proba_list.append(pd.DataFrame(data=clf.predict_proba(X_test)))
                    y_predict_proba = pd.concat(y_predict_proba_list)
                    if predict_model == 'majority_min':
                        y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                    elif predict_model == 'majority_max':
                        y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                    elif predict_model == 'majority_avg_all':
                        y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                    elif predict_model == 'majority_top_avg_all':
                        y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                    elif predict_model == 'majority_avg_max_grp':
                        y_predict_proba['max_grp'] = y_predict_proba.idxmax(axis=1)

                        y_predict_proba['max_grp_common'] = y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(
                            lambda x: x.value_counts().index[0])
                        y_predict_proba = y_predict_proba[y_predict_proba['max_grp'] == y_predict_proba['max_grp_common']]
                        y_predict_proba = y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                        y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()

                    y_predict_proba = y_predict_proba.to_numpy()
                else:
                    if downsampling:
                        train = y_train.reset_index()
                        min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                        try:
                            train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                        except:
                            train = train.groupby(['Patient_NO', 'level']).head(min_group)
                        y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                        X_train = X_train[X_train.index.isin(train['index'].unique())]

                    if smote_:
                        X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
                    if adasyn_:
                        ada = ADASYN()
                        X_train, Y_train = ada.fit_resample(X_train, y_train['target_int'])

                    clf = predict_models(predict_model, alpha, float(WIGR_power), criterion, class_weight)
                    y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)
                y_predict_proba=pd.DataFrame(y_predict_proba.tolist(), columns=['preb1', 'preb2', 'preb3'])
                y_predict_proba.to_csv(save_data_path + '\df_test_sequence_self' + predict_model + Pj + '.csv',index=False)

            y_predict_proba1 = y_predict_proba.copy()
            y_predict_proba1 = y_predict_proba1.reset_index(drop=True)
            y_test_sequence = y_test_sequence.reset_index(drop=True)
            df_test_sequence = pd.concat([y_test_sequence, y_predict_proba1], axis=1, ignore_index=True)
            #df_test_sequence = df_test_sequence.append(y_predict_proba1, ignore_index=True)
            df_before_sequence=df_before_sequence.append(df_test_sequence, ignore_index=True)
            df_test_sequence_X, df_test_sequence_y = probability_matrix(df_test_sequence,last_period)

            X, Y = df_test_sequence_X.reset_index(drop=True), df_test_sequence_y.reset_index(drop=True)
            y_predict_proba = clf_dt.predict_proba(X)
            Y_test_sheet.append(Y)
            y_predict_proba_sheet.append(y_predict_proba)
        try:
            df_before_sequence=df_before_sequence.rename(columns={1: "Patient_NO", 3: "target_int", 4: "preb1", 5: "preb2", 6: "preb3",
                                                                  '1': "Patient_NO", '3': "target_int", '4': "preb1", '5': "preb2", '6': "preb3"})
        except:
            pass

        return Y_test_sheet,y_predict_proba_sheet,df_before_sequence
    ##############################################################################################

    try:
        target.drop('Patient_NO', axis='columns', inplace=True)
        target.drop('Respiratory cycle', axis='columns', inplace=True)
        data.drop('Patient_NO', axis='columns', inplace=True)
        data.drop('Respiratory cycle', axis='columns', inplace=True)
    except:
        pass

    data = data.reset_index(drop=True)
    target = target.reset_index(drop=True)
    target = pd.DataFrame(target).reset_index()
    target = target.rename(columns={'index': 'target_int'})
    target_names = target['level'].unique().tolist()
    target_names.sort()
    target['target_int'] = target['level'].apply(lambda x: target_names.index(x) + 1)
    target['target_int'] = np.where(target['target_int'] == 1, matrix_cost[0],
                                    np.where(target['target_int'] == 2, matrix_cost[1], matrix_cost[2]))
    X = pd.DataFrame(data=data.astype('float'))
    Y = target['target_int']
    X = X.reset_index(drop=True)
    Y = Y.reset_index(drop=True)

    Y_test_sheet, y_predict_proba_sheet = [],[]
    cv = StratifiedKFold(n_splits=CV_splits, random_state=42, shuffle=True)
    for train_idx, test_idx in cv.split(X, Y):
        X_train, Y_train = X.loc[train_idx], Y.loc[train_idx]
        X_test, Y_test = X.loc[test_idx], Y.loc[test_idx]

        if 'majority' in predict_model:
            y_predict_proba_list=[]
            try:
                majority_algorithm = ast.literal_eval(majority_algorithm)
            except:
                pass
            for predict in majority_algorithm:
                print(predict)
                X_train, Y_train = X.loc[train_idx].copy(), Y.loc[train_idx].copy()
                downsampling=convert_int_to_true_fales(majority_top_algorithm_option[predict]['downsampling'])
                if downsampling:
                    train = Y_train.reset_index()
                    min_group = train.groupby(['target_int']).size().min()
                    try:
                        train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                    except:
                        train = train.groupby(['target_int']).head(min_group)
                    Y_train = Y_train[Y_train.index.isin(train['index'].unique())]
                    X_train = X_train[X_train.index.isin(train['index'].unique())]
                smote_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['smote_list'])
                if smote_list :
                    X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)
                adasyn_list = convert_int_to_true_fales(majority_top_algorithm_option[predict]['adasyn_list'])
                if adasyn_list:
                    ada=ADASYN()
                    X_train, Y_train = ada.fit_resample(X_train, Y_train)
                try:
                    clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                         , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                         , majority_top_algorithm_option[predict]['criterion']
                                         ,majority_top_algorithm_option[predict]['class_weight'])
                except:
                    clf = predict_models(predict, alpha, WIGR_power, criterion, class_weight)
                clf=clf.fit(X_train, Y_train)
                y_predict_proba_list.append(pd.DataFrame(data=clf.predict_proba(X_test)))
            y_predict_proba = pd.concat(y_predict_proba_list)
            if predict_model=='majority_min':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
            elif predict_model=='majority_max':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
            elif predict_model == 'majority_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            elif predict_model == 'majority_top_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            elif predict_model == 'majority_avg_max_grp':
                y_predict_proba['max_grp']=y_predict_proba.idxmax(axis=1)

                y_predict_proba['max_grp_common']=y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(lambda x:x.value_counts().index[0])
                y_predict_proba=y_predict_proba[y_predict_proba['max_grp']==y_predict_proba['max_grp_common']]
                y_predict_proba=y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()

            y_predict_proba=y_predict_proba.to_numpy()
        else:
            if downsampling:
                train = Y_train.reset_index()
                min_group = train.groupby(['target_int']).size().min()
                try:
                    train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                except:
                    train = train.groupby(['target_int']).head(min_group)
                Y_train = Y_train[Y_train.index.isin(train['index'].unique())]
                X_train = X_train[X_train.index.isin(train['index'].unique())]
            else:
                X_train, Y_train = X_train, Y_train
            if smote_:
                X_train, Y_train = SMOTE().fit_resample(X_train, Y_train)
            if adasyn_:
                try:
                    ada = ADASYN()
                    X_train, Y_train = ada.fit_resample(X_train, Y_train)
                except:
                    ada = ADASYN(sampling_strategy='minority')
                    X_train, Y_train = ada.fit_resample(X_train, Y_train)
            clf = predict_models(predict_model, alpha, float(WIGR_power), criterion,class_weight)
            clf.fit(X_train, Y_train)
            """
            try:
                clf.fit(X_train, Y_train)
            except:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                clf.fit(X_train, le.fit_transform(Y_train) )
            """
            y_predict_proba = clf.predict_proba(X_test)
        Y_test_sheet.append(Y_test)
        y_predict_proba_sheet.append(y_predict_proba)


    return  Y_test_sheet,y_predict_proba_sheet,pd.DataFrame()


def predict_EEG_df(data, target, predict_model,majority_algorithm, smote=True,CV=3,class_weight=None,group_all=False,type_level='level',downsampling =True,alpha=0, WIGR_power=1,criterion='entropy',majority_top_algorithm_option='',have_train_test=False,matrix_cost=[1,2,3],sequence_predict=True,predict_model_sequence='XGBoost'):
    """
    models option
    DecisionTrees
    RandomForest
    data=patient1["EEG_matrix"][0]
    target=patient1["EEG_matrix"][2]
    balanced='smote'/'downsampling'
    """
    if group_all:
        target_names = target['level'].unique().tolist()
        target_names.sort()
        target['target_int'] = target['level'].apply(lambda x: target_names.index(x) + 1)
        Accuracy_table = 0
        auc_table_CV_list = []
        confusion_matrix_list = []
        roc_auc_class_list = []
        if type_level == 'binary':
            train_df = target[['Patient_NO', 'level']].drop_duplicates()
            Patient_NO_test=np.array(np.meshgrid(train_df[train_df['level']==0]['Patient_NO'].unique(), train_df[train_df['level']==1]['Patient_NO'].unique()))
            Patient_NO_test = Patient_NO_test.T.reshape(-1, 2)
        else:
            Patient_NO_test=target.Patient_NO.unique()

        df_test_sequence=pd.DataFrame()
        for P in Patient_NO_test:
            if type_level == 'level':
                P=[P]
            try:
                X_train , X_test = data[~data['Patient_NO'].isin(P)][data.columns.difference(['Patient_NO','Respiratory cycle'])].copy(), data[data['Patient_NO'].isin(P)][data.columns.difference(['Patient_NO','Respiratory cycle'])].copy()
                y_train,y_test = target[~target['Patient_NO'].isin(P)].copy(), target[target['Patient_NO'].isin(P)]['target_int'].copy()
                y_test_sequence=target[target['Patient_NO'].isin(P)].copy()
            except:
                X_train , X_test = data[data['Patient_NO']!=P][data.columns.difference(['Patient_NO','Respiratory cycle'])].copy(), data[data['Patient_NO']==P][data.columns.difference(['Patient_NO','Respiratory cycle'])].copy()
                y_train,y_test = target[target['Patient_NO']!=P].copy(), target[target['Patient_NO']==P]['target_int'].copy()
                y_test_sequence = target[target['Patient_NO']==P].copy()

            if downsampling:
                train = y_train.reset_index()
                min_group = train.groupby(['Patient_NO', 'level'])['level'].count().min()
                try:
                    train = train.groupby(['Patient_NO', 'level']).sample(n=min_group, random_state=42)
                except:

                    train = train.groupby(['Patient_NO', 'level']).head(min_group)
                y_train = y_train[y_train.index.isin(train['index'].unique())][['target_int']]
                X_train = X_train[X_train.index.isin(train['index'].unique())]
            if smote:
                X_train, y_train = SMOTE().fit_resample(X_train, y_train['target_int'])
            if 'majority' in predict_model:
                y_predict_proba_list=[]
                for predict in majority_algorithm:
                    clf = predict_models(predict, alpha, WIGR_power, criterion,class_weight)
                    df = pd.DataFrame(data=clf.fit(X_train, y_train['target_int']).predict_proba(X_test))
                    y_predict_proba_list.append(df)
                y_predict_proba = pd.concat(y_predict_proba_list)
                if predict_model=='majority_min':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                elif predict_model=='majority_max':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                elif predict_model == 'majority_avg_all':
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                elif predict_model == 'majority_avg_max_grp':
                    y_predict_proba['max_grp']=y_predict_proba.idxmax(axis=1)
                    y_predict_proba['max_grp_common']=y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(lambda x:x.value_counts().index[0])
                    y_predict_proba=y_predict_proba[y_predict_proba['max_grp']==y_predict_proba['max_grp_common']]
                    y_predict_proba=y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                    y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                y_predict_proba=y_predict_proba.to_numpy()
            else:
                clf = predict_models(predict_model, alpha, WIGR_power, criterion,class_weight)
                y_predict_proba = clf.fit(X_train, y_train['target_int']).predict_proba(X_test)
            y_pred = np.argmax(y_predict_proba, axis=1) + 1

            y_predict_proba1=pd.DataFrame(y_predict_proba.tolist(),columns=['preb1', 'preb2', 'preb3'])
            y_predict_proba1=y_predict_proba1.reset_index(drop=True)
            y_test_sequence = y_test_sequence.reset_index(drop=True)
            y_predict_proba1=pd.concat([y_test_sequence, y_predict_proba1], axis=1,ignore_index=True)
            df_test_sequence = df_test_sequence.append(y_predict_proba1,ignore_index=True)


            if len(np.unique(y_pred)) > len(np.unique(y_test)):
                empty_class=list(set(y_pred) - set(y_test))
                y_test.loc[y_test.index.min()]=empty_class[0]
            Accuracy_table += metrics.accuracy_score(y_test, y_pred)
            ################################
            try:
                y_train_unique_tolist=y_train.unique().tolist()
            except:
                y_train_unique_tolist = y_train.target_int.unique().tolist()

            yy_test = label_binarize(y_test, classes=y_train_unique_tolist)
            if len(y_train_unique_tolist) == 2:  # binary
                yy_test = np.hstack((yy_test, 1 - yy_test))
                yy_train = np.hstack((yy_train, 1 - yy_train))
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(len(y_train_unique_tolist)):
                fpr[i], tpr[i], _ = roc_curve(yy_test[:, i], y_predict_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            ###################
            roc_auc_class_list.append(roc_auc)
            y_test-1
            try:
                auc_table_CV_list.append(roc_auc_score(y_test, y_predict_proba, multi_class='ovo', average='weighted'))
            except:
                auc_table_CV_list.append(roc_auc_score(y_test, y_pred, average='weighted'))


            # classification_report_list.append (classification_report(y_test, y_pred,target_names=target_names))
            confusion_matrix_list.append(confusion_matrix(y_test, y_pred))


        df_test_sequence = df_test_sequence.rename(columns={1: 'Patient_NO', 3: 'target_int'
            , 4: 'preb1', 5: 'preb2', 6: 'preb3'}, inplace=False)
        for col_ in ['preb1','preb2','preb3']:
            df_test_sequence[col_+' t-1'] = df_test_sequence.groupby('Patient_NO')[col_].shift(1)
            df_test_sequence[col_ + ' t-2'] = df_test_sequence.groupby('Patient_NO')[col_].shift(2)
            df_test_sequence[col_ + ' t-3'] = df_test_sequence.groupby('Patient_NO')[col_].shift(3)

        df_test_sequence = df_test_sequence.dropna()
        df_test_sequence_X=df_test_sequence[['preb1', 'preb2', 'preb3', 'preb1 t-1', 'preb1 t-2', 'preb1 t-3',
       'preb2 t-1', 'preb2 t-2', 'preb2 t-3', 'preb3 t-1', 'preb3 t-2',
       'preb3 t-3']]
        df_test_sequence_y=df_test_sequence['target_int']
        df_test_sequence_X, df_test_sequence_X_test, df_test_sequence_y, df_test_sequence_y_test = train_test_split(df_test_sequence_X, df_test_sequence_y, stratify=df_test_sequence_y, test_size=0.2, random_state=32)

        clf2 = predict_models(predict_model=predict_model_sequence,class_weight='balanced',alpha=0, WIGR_power=0.2, criterion='entropy')
        y_predict_proba2 = clf2.fit(df_test_sequence_X, df_test_sequence_y).predict_proba(df_test_sequence_X_test)
        Accuracy_sequence, roc_auc_sequence, auc_value_sequence, confusion_matrix_sequence = get_summary_tables(y_predict_proba2, df_test_sequence_y_test, df_test_sequence_y,
                                                                                matrix_cost)
        Accuracy_table = Accuracy_table / len(target.Patient_NO.unique())
        auc_table = sum(auc_table_CV_list) / len(auc_table_CV_list)
        auc_std_CV = np.std(auc_table_CV_list, axis=0)
        roc_auc_class_avg = pd.DataFrame(roc_auc_class_list).mean().to_list()
        try:
            value_counts=target[['level', 'Patient_NO']].value_counts(sort=False)
        except:
            value_counts=pd.Series(target.groupby(['level', 'Patient_NO']).size())
        return Accuracy_table, auc_table, auc_table_CV_list, auc_std_CV, value_counts, confusion_matrix_list, roc_auc_class_list, roc_auc_class_avg,Patient_NO_test,Accuracy_sequence, roc_auc_sequence, auc_value_sequence, confusion_matrix_sequence

    try:
        target.drop('Patient_NO',axis='columns', inplace=True)
        target.drop('Respiratory cycle', axis='columns', inplace=True)
        data.drop('Patient_NO', axis='columns', inplace=True)
        data.drop('Respiratory cycle', axis='columns', inplace=True)
    except:
        pass

    data = data.reset_index(drop=True)
    target = target.reset_index(drop=True)
    target = pd.DataFrame(target).reset_index()
    target = target.rename(columns={'index': 'target_int'})
    target_names = target['level'].unique().tolist()
    target_names.sort()
    target['target_int'] = target['level'].apply(lambda x: target_names.index(x) + 1)
    target['target_int']=np.where(target['target_int']==1,matrix_cost[0],np.where(target['target_int']==2,matrix_cost[1],matrix_cost[2]))
    X = pd.DataFrame(data=data.astype('float'))
    Y=target['target_int']
    X=X.reset_index(drop=True)
    Y=Y.reset_index(drop=True)

    if 'majority' in predict_model:
        y_predict_proba_list = []






    # cv
    Accuracy_table = 0
    cost_value=0
    auc_table_CV_list = []
    confusion_matrix_list=[]
    roc_auc_class_list=[]
    if have_train_test:
        Accuracy_table_test = 0
        cost_value_test=0
        auc_table_CV_list_test = []
        confusion_matrix_list_test = []
        roc_auc_class_list_test = []

        Accuracy_table_test_test_train = 0
        cost_value_test_test_train= 0
        auc_table_CV_list_test_test_train = []
        confusion_matrix_list_test_test_train = []
        roc_auc_class_list_test_test_train = []
        X_all=X.copy()
        Y_all = Y.copy()
        X, X_test, Y, Y_test = train_test_split(X_all, Y_all, stratify=Y_all, test_size=0.2,random_state=32)
        y_train_main=Y.copy()
        X_train_main=X.copy()
        X_test=X_test.copy()
        y_test=Y_test.copy()
        if 'majority' in predict_model:
            y_predict_proba_list=[]
            y_predict_proba_test_train_list=[]
            for predict in majority_algorithm:
                if majority_top_algorithm_option[predict]['downsampling']:
                    train = y_train_main.reset_index()
                    min_group = train.groupby(['target_int']).size().min()
                    try:
                        train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                    except:
                        train = train.groupby(['target_int']).head(min_group)
                    y_train = y_train_main[y_train_main.index.isin(train['index'].unique())]
                    X_train = X_train_main[X_train_main.index.isin(train['index'].unique())]
                else:
                    X_train,y_train=X_train_main, y_train_main
                if majority_top_algorithm_option[predict]['smote_list']:
                    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                if majority_top_algorithm_option[predict]['adasyn_list']:
                    ada = ADASYN()
                    X_train, y_train = ada.fit_resample(X_train, y_train)
                try:
                    clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                         , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                         , majority_top_algorithm_option[predict]['criterion']
                                         ,majority_top_algorithm_option[predict]['class_weight'])
                except:
                    clf = predict_models(predict, alpha, WIGR_power, criterion, class_weight)
                clf=clf.fit(X_train, y_train)
                y_predict_proba_list.append(pd.DataFrame(data=clf.predict_proba(X_test)))
                y_predict_proba_test_train_list.append(pd.DataFrame(data=clf.predict_proba(X_all)))
            y_predict_proba = pd.concat(y_predict_proba_list)
            y_predict_proba_test_train = pd.concat(y_predict_proba_test_train_list)
            if predict_model=='majority_min':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
                y_predict_proba_test_train = y_predict_proba_test_train.groupby(y_predict_proba_test_train.index).min()
            elif predict_model=='majority_max':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
                y_predict_proba_test_train = y_predict_proba_test_train.groupby(y_predict_proba_test_train.index).max()
            elif predict_model == 'majority_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                y_predict_proba_test_train = y_predict_proba_test_train.groupby(y_predict_proba_test_train.index).mean()
            elif predict_model == 'majority_top_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
                y_predict_proba_test_train = y_predict_proba_test_train.groupby(y_predict_proba_test_train.index).mean()
            elif predict_model == 'majority_avg_max_grp':
                y_predict_proba['max_grp']=y_predict_proba.idxmax(axis=1)

                y_predict_proba['max_grp_common']=y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(lambda x:x.value_counts().index[0])
                y_predict_proba=y_predict_proba[y_predict_proba['max_grp']==y_predict_proba['max_grp_common']]
                y_predict_proba=y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            y_predict_proba=y_predict_proba.to_numpy()
            y_predict_proba_test_train = y_predict_proba_test_train.to_numpy()
        else:
            if downsampling:
                train = y_train_main.reset_index()
                min_group = train.groupby(['target_int']).size().min()
                try:
                    train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                except:
                    train = train.groupby(['target_int']).head(min_group)
                y_train = y_train_main[y_train_main.index.isin(train['index'].unique())]
                X_train = X_train_main[X_train_main.index.isin(train['index'].unique())]
            else:
                X_train, y_train = X_train_main, y_train_main
            if smote:
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            clf = predict_models(predict_model, alpha, WIGR_power, criterion,class_weight)
            clf.fit(X_train, y_train)
            y_predict_proba = clf.predict_proba(X_test)
            y_predict_proba_test_train = clf.predict_proba(X_all)

        y_pred = np.argmax(y_predict_proba, axis=1)+1
        f1_score_df= pd.DataFrame(y_predict_proba)

        f1_score_df['f1_score_class_2_0.3']=np.where(f1_score_df[1]>0.3,1,np.where(f1_score_df[2]>f1_score_df[0],2,0))+1
        f1_score_df['f1_score_class_2_0.6']=np.where(f1_score_df[1]>0.6,1,np.where(f1_score_df[2]>f1_score_df[0],2,0))+1
        f1_score_df['f1_score_class_2_0.9']=np.where(f1_score_df[1]>0.9,1,np.where(f1_score_df[2]>f1_score_df[0],2,0))+1

        f1_score_df['f1_score_class_3_0.3']=np.where(f1_score_df[2]>0.3,2,np.where(f1_score_df[1]>f1_score_df[0],1,0))+1
        f1_score_df['f1_score_class_3_0.6']=np.where(f1_score_df[2]>0.6,2,np.where(f1_score_df[1]>f1_score_df[0],1,0))+1
        f1_score_df['f1_score_class_3_0.9']=np.where(f1_score_df[2]>0.9,2,np.where(f1_score_df[1]>f1_score_df[0],1,0))+1

        from sklearn.metrics import f1_score
        f1_score_string_class_2='f1_score_class_2_0.3 :'+str(f1_score(y_test, f1_score_df['f1_score_class_2_0.3'], average='macro'))[0:4]
        f1_score_string_class_2 += ' f1_score_class_2_0.6 :' + str(f1_score(y_test, f1_score_df['f1_score_class_2_0.6'], average='macro'))[0:4]
        f1_score_string_class_2 += ' f1_score_class_2_0.9 :' + str(f1_score(y_test, f1_score_df['f1_score_class_2_0.9'], average='macro'))[0:4]
        f1_score_string_class_3 = ' f1_score_class_3_0.3 :' + str(f1_score(y_test, f1_score_df['f1_score_class_3_0.3'], average='macro'))[0:4]
        f1_score_string_class_3 += ' f1_score_class_3_0.6 :' + str(f1_score(y_test, f1_score_df['f1_score_class_3_0.6'], average='macro'))[0:4]
        f1_score_string_class_3 += ' f1_score_class_3_0.9 :' + str(f1_score(y_test, f1_score_df['f1_score_class_3_0.9'], average='macro'))[0:4]


        Accuracy_,roc_auc_,auc_value_,confusion_matrix_=get_summary_tables(y_predict_proba,y_test,y_train,matrix_cost)
        cost_=np.zeros((len(matrix_cost), len(matrix_cost)))
        for row_ in range(0,len(matrix_cost)):
            for col_ in range(0, len(matrix_cost)):
                cost_[row_][col_] = abs(matrix_cost[row_] - matrix_cost[col_])
        cost_value_test+=(confusion_matrix_*cost_).sum()/confusion_matrix_.sum()
        Accuracy_table_test+=Accuracy_
        roc_auc_class_list_test.append(roc_auc_)
        auc_table_CV_list_test.append(auc_value_)
        confusion_matrix_list_test.append (confusion_matrix_)
        auc_table_test = sum(auc_table_CV_list_test) / len(auc_table_CV_list_test)
        roc_auc_class_avg_test = pd.DataFrame(roc_auc_class_list_test).mean().to_list()
        y_pred_df= pd.DataFrame(y_predict_proba)
        y_pred_df['y_pred_0.2'] = np.where((y_pred_df[1] < 0.2) & (y_pred_df[2] < 0.2), 0, np.where(y_pred_df[1] > y_pred_df[2], 1, 2)) + 1
        y_pred_df['y_pred_0.25'] = np.where((y_pred_df[1] < 0.25) & (y_pred_df[2] < 0.25), 0, np.where(y_pred_df[1] > y_pred_df[2], 1, 2)) + 1
        y_pred_df['y_pred_0.3'] = np.where((y_pred_df[1] < 0.3) & (y_pred_df[2] < 0.3), 0, np.where(y_pred_df[1] > y_pred_df[2], 1, 2)) + 1
        y_pred_df['y_pred_0.35'] = np.where((y_pred_df[1] < 0.35) & (y_pred_df[2] < 0.35), 0, np.where(y_pred_df[1] > y_pred_df[2], 1, 2)) + 1
        y_test_con=np.where(y_test==matrix_cost[0],1,np.where(y_test==matrix_cost[1],2,3))
        confusion_matrix_2 = confusion_matrix(y_test_con, y_pred_df['y_pred_0.2'])
        confusion_matrix_25 = confusion_matrix(y_test_con, y_pred_df['y_pred_0.25'])
        confusion_matrix_3 = confusion_matrix(y_test_con, y_pred_df['y_pred_0.3'])
        confusion_matrix_35 = confusion_matrix(y_test_con, y_pred_df['y_pred_0.35'])



        Accuracy_test_train,roc_auc_test_train,auc_value_test_train,confusion_matrix_test_train=get_summary_tables(y_predict_proba_test_train,Y_all,y_train,matrix_cost)
        cost_=np.zeros((len(matrix_cost), len(matrix_cost)))
        for row_ in range(0,len(matrix_cost)):
            for col_ in range(0, len(matrix_cost)):
                cost_[row_][col_] = abs(matrix_cost[row_] - matrix_cost[col_])
        cost_value_test_test_train+=(confusion_matrix_test_train*cost_).sum()/confusion_matrix_test_train.sum()
        Accuracy_table_test_test_train+=Accuracy_test_train
        roc_auc_class_list_test_test_train.append(roc_auc_test_train)
        auc_table_CV_list_test_test_train.append(auc_value_test_train)
        confusion_matrix_list_test_test_train.append (confusion_matrix_test_train)
        auc_table_test_test_train = sum(auc_table_CV_list_test_test_train) / len(auc_table_CV_list_test_test_train)
        roc_auc_class_avg_test_test_train = pd.DataFrame(roc_auc_class_list_test_test_train).mean().to_list()

        X = X.reset_index(drop=True)
        Y = Y.reset_index(drop=True)

    else:
        Accuracy_table_test=0
        cost_value_test=0
        auc_table_test=0
        roc_auc_class_avg_test=0
        confusion_matrix_list_test=0

        Accuracy_table_test_test_train=0
        cost_value_test_test_train=0
        auc_table_test_test_train=0
        roc_auc_class_avg_test_test_train=0
        confusion_matrix_list_test_test_train=0

        f1_score_string_class_2=''
        f1_score_string_class_3 = ''

        confusion_matrix_2=''
        confusion_matrix_25 = ''
        confusion_matrix_3 = ''
        confusion_matrix_35 =''


    cv = StratifiedKFold(n_splits=CV,random_state=42,shuffle=True)
    for train_idx, test_idx in cv.split(X, Y):
        X_train_main, y_train_main = X.loc[train_idx], Y.loc[train_idx]
        #X_train_main, y_train_main = X[train_idx], Y[train_idx]
        #X_test, y_test = X[test_idx], Y[test_idx]

        X_test, y_test = X.loc[test_idx], Y.loc[test_idx]


        if 'majority' in predict_model:
            y_predict_proba_list=[]
            for predict in majority_algorithm:
                if majority_top_algorithm_option[predict]['downsampling']:
                    train = y_train_main.reset_index()
                    min_group = train.groupby(['target_int']).size().min()
                    try:
                        train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                    except:
                        train = train.groupby(['target_int']).head(min_group)
                    y_train = y_train_main[y_train_main.index.isin(train['index'].unique())]
                    X_train = X_train_main[X_train_main.index.isin(train['index'].unique())]
                else:
                    X_train,y_train=X_train_main, y_train_main
                if majority_top_algorithm_option[predict]['smote_list']:
                    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
                if majority_top_algorithm_option[predict]['adasyn_list']:
                    ada = ADASYN()
                    X_train, y_train = ada.fit_resample(X_train, y_train)
                try:
                    clf = predict_models(majority_top_algorithm_option[predict]['predict_model']
                                         , alpha, majority_top_algorithm_option[predict]['WIGR_power']
                                         , majority_top_algorithm_option[predict]['criterion']
                                         ,majority_top_algorithm_option[predict]['class_weight'])
                except:
                    clf = predict_models(predict, alpha, WIGR_power, criterion, class_weight)
                df = pd.DataFrame(data=clf.fit(X_train, y_train).predict_proba(X_test))
                y_predict_proba_list.append(df)
            y_predict_proba = pd.concat(y_predict_proba_list)
            if predict_model=='majority_min':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).min()
            elif predict_model=='majority_max':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).max()
            elif predict_model == 'majority_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            elif predict_model == 'majority_top_avg_all':
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            elif predict_model == 'majority_avg_max_grp':
                y_predict_proba['max_grp']=y_predict_proba.idxmax(axis=1)

                y_predict_proba['max_grp_common']=y_predict_proba.groupby(y_predict_proba.index)['max_grp'].agg(lambda x:x.value_counts().index[0])
                y_predict_proba=y_predict_proba[y_predict_proba['max_grp']==y_predict_proba['max_grp_common']]
                y_predict_proba=y_predict_proba.drop(columns=['max_grp', 'max_grp_common'])
                y_predict_proba = y_predict_proba.groupby(y_predict_proba.index).mean()
            y_predict_proba=y_predict_proba.to_numpy()
        else:
            if downsampling:
                train = y_train_main.reset_index()
                min_group = train.groupby(['target_int']).size().min()
                try:
                    train = train.groupby(['target_int']).sample(n=min_group, random_state=42)
                except:
                    train = train.groupby(['target_int']).head(min_group)
                y_train = y_train_main[y_train_main.index.isin(train['index'].unique())]
                X_train = X_train_main[X_train_main.index.isin(train['index'].unique())]
            else:
                X_train, y_train = X_train_main, y_train_main
            if smote:
                X_train, y_train = SMOTE().fit_resample(X_train, y_train)
            clf = predict_models(predict_model, alpha, WIGR_power, criterion,class_weight)
            y_predict_proba = clf.fit(X_train, y_train).predict_proba(X_test)

        #y_predict_proba = clf.fit(X_train, y_train).predict_proba(X_test)
        y_pred = np.argmax(y_predict_proba, axis=1)+1
        y_pred = np.where(y_pred == 1, matrix_cost[0],np.where(y_pred== 2, matrix_cost[1], matrix_cost[2]))
        Accuracy_table+=metrics.accuracy_score(y_test, y_pred)
        ################################
        yy_test = label_binarize(y_test, classes=y_train.unique().tolist())
        yy_train = label_binarize(y_train, classes=y_train.unique().tolist())
        if len(y_train.unique().tolist()) ==2: #binary
            yy_test = np.hstack((yy_test, 1 - yy_test))
            yy_train = np.hstack((yy_train, 1 - yy_train))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(y_train.unique().tolist())):
            fpr[i], tpr[i], _ = roc_curve(yy_test[:, i], y_predict_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        ###################
        roc_auc_class_list.append(roc_auc)

        try:
            auc_table_CV_list.append(roc_auc_score(y_test, y_predict_proba, multi_class='ovo', average='weighted'))
        except:
            auc_table_CV_list.append(roc_auc_score(y_test, y_pred,  average='weighted'))


        #classification_report_list.append (classification_report(y_test, y_pred,target_names=target_names))
        confusion_matrix_=confusion_matrix(y_test, y_pred)
        cost_ = np.zeros((len(matrix_cost), len(matrix_cost)))
        for row_ in range(0, len(matrix_cost)):
            for col_ in range(0, len(matrix_cost)):
                cost_[row_][col_] = abs(matrix_cost[row_] - matrix_cost[col_])
        try:
            cost_value += (confusion_matrix_ * cost_).sum() / confusion_matrix_.sum()
        except:
            cost_value += 0

        confusion_matrix_list.append (confusion_matrix(y_test, y_pred))
    Accuracy_table = Accuracy_table / CV
    cost_value= cost_value/ CV
    auc_table=sum(auc_table_CV_list) / len(auc_table_CV_list)
    auc_std_CV=np.std(auc_table_CV_list, axis=0)
    roc_auc_class_avg=pd.DataFrame(roc_auc_class_list).mean().to_list()
    try:
        value_counts = target.value_counts()
    except:
        value_counts = pd.Series(target.groupby(['level']).size())






    return  Accuracy_table,cost_value ,auc_table,auc_table_CV_list,auc_std_CV,value_counts,confusion_matrix_list,roc_auc_class_list,roc_auc_class_avg,\
            Accuracy_table_test,cost_value_test,auc_table_test,roc_auc_class_avg_test,confusion_matrix_list_test,f1_score_string_class_2,f1_score_string_class_3,\
            Accuracy_table_test_test_train,cost_value_test_test_train,auc_table_test_test_train,roc_auc_class_avg_test_test_train,confusion_matrix_list_test_test_train\
        ,confusion_matrix_2,confusion_matrix_25,confusion_matrix_3,confusion_matrix_35



