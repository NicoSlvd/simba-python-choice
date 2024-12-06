from simba.mobi.rumboost.rumboost import rum_train

import lightgbm as lgb
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold

from rumboost_definition import get_rumboost_model_spec, lightgbm_data, define_variables
from descriptive_stats import calculate_metrics

try:
    import torch

    TORCH_INSTALLED = True
    print("Torch is installed, attempting to run on GPU")
except ImportError:
    TORCH_INSTALLED = False


def train_rumboost_homeoffice(
    df_zp, output_directory, intensity_cutoff=None, df_zp_test=None, year: int = None
):
    year_name = f"{year}" if year else "2015_2020_2021"
    output_directory = output_directory / f"models/estimation/{year_name}/"
    output_directory.mkdir(parents=True, exist_ok=True)
    train_rumb(df_zp, output_directory, df_zp_test, intensity_cutoff)


def train_rumb(df_zp_train, output_directory, df_zp_test=None, intensity_cutoff=None):

    torch_tensors = {"device": "cuda"} if TORCH_INSTALLED else None

    choice_situation = "intensity" if intensity_cutoff else "possibility"
    print("Training RUMBoost model for predicting telecommuting " + choice_situation)

    choice = "telecommuting_intensity" if intensity_cutoff else "telecommuting"
    new_df_train = define_variables(df_zp_train, choice)
    new_df_test = (
        define_variables(df_zp_test, choice) if df_zp_test is not None else None
    )

    model_specification = get_rumboost_model_spec(new_df_train, intensity_cutoff)

    features = new_df_train.columns.tolist()

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    num_trees = 0
    metrics_df = pd.DataFrame({})

    # kfold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(new_df_train)):
        print(f"Fold {i+1}/{k}")
        train_data = lightgbm_data(
            new_df_train.loc[train_index, features],
            new_df_train.loc[train_index, choice],
        )
        val_data = lightgbm_data(
            new_df_train.loc[test_index, features], new_df_train.loc[test_index, choice]
        )

        try:
            model = rum_train(train_data, model_specification, valid_sets=[val_data], torch_tensors=torch_tensors)
        except:
            model = rum_train(train_data, model_specification, valid_sets=[val_data])

        print(
            f"Best cross-entropy: {model.best_score} with {model.best_iteration} trees"
        )

        # storing number of trees
        num_trees += model.best_iteration

        # cv metrics
        y_pred_train = model.predict(train_data)
        y_pred_val = model.predict(val_data)
        metrics = calculate_metrics(
            new_df_train.loc[test_index, choice], y_pred_train, intensity_cutoff
        )
        metrics_val = calculate_metrics(
            new_df_train.loc[test_index, choice], y_pred_val, intensity_cutoff
        )
        metrics_df = metrics_df.append(metrics, ignore_index=True)
        metrics_df = metrics_df.append(metrics_val, ignore_index=True)

    # average number of trees over the k folds
    avg_trees = int(num_trees / k)

    model_specification["general_params"]["num_iterations"] = avg_trees
    model_specification["general_params"]["early_stopping_round"] = None

    # final training on the test set
    train_data = lightgbm_data(new_df_train[features], new_df_train[choice])

    try:
        model = rum_train(train_data, model_specification, torch_tensors=torch_tensors)
    except:
        model = rum_train(train_data, model_specification)

    # training metrics
    y_pred = model.predict(train_data)

    metrics = calculate_metrics(new_df_train[choice], y_pred, intensity_cutoff)
    metrics_df = metrics_df.append(metrics, ignore_index=True)

    if df_zp_test is not None:
        test_data = lightgbm_data(new_df_test[features], new_df_test[choice])
        predictions = model.predict(test_data)
        metrics_test = calculate_metrics(
            new_df_test[choice], predictions, intensity_cutoff
        )
        metrics_df = metrics_df.append(metrics_test, ignore_index=True)

    # save model and metrics
    str_model = "intensity" if intensity_cutoff else "possibility"
    model.save_model(
        output_directory / f"rumboost_model_wfh_{str_model}_"
        + datetime.now().strftime("%Y_%m_%d-%H_%M")
        + ".json"
    )
    file_name = (
        f"rumboost_metrics_wfh_{str_model}_"
        + datetime.now().strftime("%Y_%m_%d-%H_%M")
        + ".csv"
    )
    metrics_df.to_csv(output_directory / file_name, index=False)

    # figures
    # plot_parameters(final_model, df_train, {"0": "Binary"}, save_file='output/figures/rumboost_with_intensity_alldata')
