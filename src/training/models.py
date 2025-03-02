def predict_models(algorithm_params):
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

    predict_model=algorithm_params['predict_model']
    alpha=1
    WIGR_power = algorithm_params['WIGR_power']
    criterion = algorithm_params['criterion']
    class_weight = algorithm_params['class_weight']


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
