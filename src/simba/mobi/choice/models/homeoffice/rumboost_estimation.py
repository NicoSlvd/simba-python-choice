import sys
sys.path.append('../../../../../../../../rumboost-dev/')

from rumboost.rumboost import rum_train

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold
from rumboost.metrics import cross_entropy

df_train = pd.read_csv('input/data/2015/persons_cleaned_train.csv')
df_test = pd.read_csv('input/data/2015/persons_cleaned_test.csv')

features = [
    'age',
    'dl_car',
    'work_sched_fully_flex',
    'ptsub_regional',
    #'ptsub_parcours',
    'male',
    'self_employed_with_employees',
    'self_employed_without_employees',
    'employed_by_family',
    'employee_management',
    'employee_with_supervision',
    'secondary_education',
    'hh_size',
    'hh_income_4000_6000',
    'hh_income_6000_8000',
    'hh_income_8000_10000',
    'hh_income_10000_12000',
    'hh_income_12000_14000',
    'hh_income_14000_16000',
    'hh_income_16000_plus',
    'car_net_distance',
    'accsib_car'
]

train_set = lgb.Dataset(df_train[features], label=df_train['work_home_days'], free_raw_data=False)
test_set = lgb.Dataset(df_test[features], label=df_test['work_home_days'], free_raw_data=False)

monotone_constraints = [0] * len(features)
interaction_constraints = [[i] for i, _ in enumerate(features)]


#rum_structure = [
#    {
#        "utility": [0],
#        "variables": features,
#        "boosting_params": {
#            "monotone_constraints_method": "advanced",
#            "max_depth": 1,
#            "n_jobs": -1,
#            "learning_rate": 0.1,
#            "monotone_constraints": monotone_constraints,
#            "interaction_constraints": interaction_constraints,
#        },
#        "shared": False,
#    }
#]

rum_structure1 = [
    {
        "utility": [0],
        "variables": [f],
        "boosting_params": {
            "monotone_constraints_method": "advanced",
            "max_depth": 1,
            "n_jobs": -1,
            "learning_rate": 0.005,
            "verbose": -1,
        },
        "shared": False,
    } for f in features
]
rum_structure2 = [
    {
        "utility": [0],
        "variables": [f],
        "boosting_params": {
            "monotone_constraints_method": "advanced",
            "max_depth": 1,
            "n_jobs": -1,
            "learning_rate": 0.01,
            "verbose": -1,
        },
        "shared": False,
    } for f in features
]
rum_structure = rum_structure1 + rum_structure2
general_params = {
    "n_jobs": -1,
    "num_classes": 6,  # important
    "verbosity": 1,  # specific RUMBoost parameter
    "verbose_interval": 1,
    "num_iterations": 50,
    "early_stopping_round": None,
    "boost_from_parameter_space": [True] * len(rum_structure1) + [False] * len(rum_structure2),
    "max_booster_to_update": len(rum_structure)
}

model_specification = {
    "general_params": general_params,
    "rum_structure": rum_structure,
    "ordinal_logit": {
        "model": "proportional_odds",
        "optim_interval": 1,
    }
}

torch_tensors = {
    "device": "cuda"
}
#kfold = 5

#kf = KFold(n_splits=kfold, shuffle=True, random_state=42)

#num_trees = 0
#for i, (train_index, val_index) in enumerate(kf.split(df_train)):
    #train_set_cv = lgb.Dataset(df_train.loc[train_index, features], label=df_train.loc[train_index, 'work_home_days'], free_raw_data=False)
    #val_set = lgb.Dataset(df_train.loc[val_index, features], label=df_train.loc[val_index, 'work_home_days'], free_raw_data=False)

    #model = rum_train(train_set_cv, model_specification, valid_sets=[val_set])

    #num_trees += model.best_iteration

    #print(f'Fold {i+1}/{kfold}: best ce = {model.best_score} with {model.best_iteration} trees')

#num_trees = int(num_trees / kfold)

#print(f'Average number of trees: {num_trees}')

#general_params['num_iterations'] = num_trees
#general_params['early_stopping_round'] = None

final_model = rum_train(train_set, model_specification, torch_tensors=torch_tensors)

y_pred = final_model.predict(test_set)

cel = cross_entropy(y_pred, df_test['work_home_days'].values)

if general_params['boost_from_parameter_space']:
    ASC = final_model.asc.sum()
else:
    df_zeros = pd.DataFrame({f: [0] for f in features})

    ASC = final_model.predict(lgb.Dataset(df_zeros, free_raw_data=False), utilities=True)[0]

print(f'Cross-entropy loss on train set: {final_model.best_score_train}')
print(f'Cross-entropy loss on test set: {cel}')
print(f'ASC: {ASC}')
print(f'Thresholds: {final_model.thresholds}')
print(f'Thresholds without ASC: {final_model.thresholds-ASC}')

#final_model.save_model('output/data/rumboost_model.json')

from rumboost.utility_plotting import plot_parameters


#plot_parameters(final_model, df_train, {"0": "Ordinal "}, save_file='output/figures/rumboost')
utility_names = {str(i): "ordinal" for i, f in enumerate(features * 2)}

boost_from_p_plot = {str(i): {} for i, f in enumerate(features * 2)}
for j, struct in enumerate(rum_structure):
    boost_from_p_plot[str(j)][struct["variables"][0]] = general_params["boost_from_parameter_space"][j]

plot_parameters(final_model, df_train, utility_names, boost_from_parameter_space=boost_from_p_plot, group_feature={'age': [0, len(rum_structure1)]}, only_tt=False)
#plot_parameters(final_model, df_train, utility_names, boost_from_parameter_space=False)