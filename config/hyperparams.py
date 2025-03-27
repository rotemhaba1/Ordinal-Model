param_grids = {
    "RandomForest": {
        "class_weight": [ 'balanced',None], },

    "RandomForest_Ordinal":{
        'WIGR_power':[0.2,1],
        'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max'],
        "class_weight": [ 'balanced',None]},

    "DecisionTrees_Ordinal": {
        'WIGR_power':[0.2,1],
        'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max'],
        "class_weight": [ 'balanced',None]},

    "DecisionTrees": {
        "class_weight": [ 'balanced',None], },

    "AdaBoost": {},

    "catboost": {},

    "XGBoost": {},

    "AdaBoost_Ordinal": {
        'WIGR_power': [0.2,1],
        'criterion': ['WIGR_EV', 'WIGR_mode', 'WIGR_min', 'WIGR_EV_fix','WIGR_max'],
        'algorithm': ['', 'SAMME_R', 'half', 'SAMME'],
        "class_weight": [ 'balanced',None]},



            }

param_grids = {
    "RandomForest": {
        "class_weight": [ 'balanced',None], },

    "RandomForest_Ordinal":{
        'WIGR_power':[0.2,1],
        'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max'],
        "class_weight": [ 'balanced',None]},

    "DecisionTrees_Ordinal": {
        'WIGR_power':[0.2,1],
        'criterion':['WIGR_EV','WIGR_mode','WIGR_min','WIGR_EV_fix','WIGR_max'],
        "class_weight": [ 'balanced',None]},

    "DecisionTrees": {
        "class_weight": [ 'balanced',None], },

    "AdaBoost": {},

    "XGBoost": {},

    "AdaBoost_Ordinal": {
        'WIGR_power': [1],
        'criterion': ['WIGR_min'],
        'algorithm': ['SAMME'],
        "class_weight": [None]},



            }



param_ensemble={
'1':{'model':"RandomForest_Ordinal",'combo':{'WIGR_power':0.2,'criterion':'WIGR_EV','class_weight':None}},
'2':{'model':"RandomForest_Ordinal",'combo':{'WIGR_power':0.2,'criterion':'WIGR_EV','class_weight':None}},
'3':{'model':"XGBoost",'combo':{}},
'4':{'model':"RandomForest",'combo':{'class_weight':'balanced'}},
'5':{'model':"XGBoost",'combo':{}},
'6':{'model':"catboost",'combo':{}},
'7':{'model':"catboost",'combo':{}},
}

params = {
    "min_diff": 1.5,
    "max_diff": 9,
    "min_length": 1.5,
    "max_length": 8,
    "remove_level": ["Inhalation"],
    'downsampling': False,
    'smote': True,
}

