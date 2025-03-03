param_grids = {
    "RandomForest_Ordinal":{'WIGR_power':[0.2],'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix']},
    "DecisionTrees_Ordinal": {'WIGR_power': [ 2],'criterion': ['WIGR_max']},
    "AdaBoost_SAMME_R_Ordinal":{'WIGR_power':[2],'criterion':['WIGR_EV']},
    "AdaBoost_half_Ordinal":{'WIGR_power':[0.2],'criterion':['WIGR_EV_fix','WIGR_max']},
    "AdaBoost_Ordinal":{'WIGR_power':[2],'criterion':['WIGR_max']},
    "AdaBoost_SAMME_Ordinal": {'WIGR_power': [1], 'criterion': ['WIGR_min']},
    "catboost": {},
"XGBoost": {},
    "DecisionTrees": {
        "class_weight": [ 'balanced',None], },
    "RandomForest": {},
    "AdaBoost": {},

}