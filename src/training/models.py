def predict_models(params):

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier


    predict_model = params.get('model', None)
    alpha = 1
    WIGR_power = params['combo'].get('WIGR_power', None)
    criterion = params['combo'].get('criterion', None)
    class_weight = params['combo'].get('class_weight', None)


    if predict_model == 'DecisionTrees':
        from sklearn import tree
        clf_model = tree.DecisionTreeClassifier(class_weight=class_weight)
    elif predict_model == 'RandomForest':
        from sklearn.ensemble import RandomForestClassifier
        clf_model = RandomForestClassifier()
    elif predict_model == 'AdaBoost':
        from sklearn.ensemble import AdaBoostClassifier
        clf_model = AdaBoostClassifier()
    elif predict_model == 'catboost':
        from catboost import CatBoostClassifier
        clf_model = CatBoostClassifier(task_type="GPU",silent=True)
    elif predict_model == 'XGBoost':
        from xgboost import XGBClassifier
        #clf_model = XGBClassifier()
        clf_model = XGBClassifier(tree_method="hist", device="cuda")
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

def get_model_param_grid():
    param_grids = {
        "XGBoost": {
        },
        "RandomForest": {
        },
        "DecisionTrees": {
            "class_weight": [None, 'balanced'],
        },
        "AdaBoost": {
        },
        "catboost": {
        }
    }

    return param_grids