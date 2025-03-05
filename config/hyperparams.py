param_grids = {
    "RandomForest": {},
    "RandomForest_Ordinal":{'WIGR_power':[0.2,1,2],'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max']},
    "DecisionTrees_Ordinal": {'WIGR_power':[0.2,1,2],'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max']},
"XGBoost": {},
    "DecisionTrees": {
        "class_weight": [ 'balanced',None], },
    "AdaBoost": {},
"catboost": {},

    "AdaBoost_Ordinal": {'WIGR_power': [0.2, 1, 2],
                         'criterion': ['WIGR_EV', 'WIGR_mode', 'WIGR_min', 'WIGR_EV_fix','WIGR_max'],
                         'algorithm': ['', 'SAMME_R', 'half', 'SAMME']},

}

param_grids = {
    "RandomForest_Ordinal":{'WIGR_power':[0.2,1,2],'criterion':['WIGR_max']},
    "DecisionTrees_Ordinal": {'WIGR_power':[0.2,1,2],'criterion':['WIGR_max']},
    "AdaBoost_Ordinal": {'WIGR_power': [0.2, 1, 2],
                         'criterion': ['WIGR_max', 'WIGR_EV_fix'],
                         'algorithm': ['', 'SAMME_R', 'half', 'SAMME']},

}


param_ensemble={
'1':{'model':"RandomForest_Ordinal",'combo':{'WIGR_power':0.2,'criterion':'WIGR_EV'}},
'2':{'model':"RandomForest_Ordinal",'combo':{'WIGR_power':0.2,'criterion':'WIGR_EV'}},
'3':{'model':"XGBoost",'combo':{}},
'4':{'model':"RandomForest",'combo':{}},
'5':{'model':"XGBoost",'combo':{}},
'6':{'model':"catboost",'combo':{}},
'7':{'model':"catboost",'combo':{}},
}

