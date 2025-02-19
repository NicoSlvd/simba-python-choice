from src.simba.mobi.rumboost.rumboost import rum_train
from src.simba.mobi.rumboost.utility_plotting import plot_parameters

import lightgbm as lgb
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold

from src.simba.mobi.choice.models.homeoffice.rumboost_definition import (
    get_rumboost_model_spec,
    lightgbm_data,
    define_variables,
)
from src.simba.mobi.choice.models.homeoffice.descriptive_stats import calculate_metrics

try:
    import torch

    TORCH_INSTALLED = True
except ImportError:
    TORCH_INSTALLED = False
# TORCH_INSTALLED = False


def train_rumboost_telecommuting(
    df_zp,
    output_directory,
    intensity_cutoff=None,
    df_zp_test=None,
    year: int = None,
    seed: int = None,
):
    year_name = f"{year}" if year else "2015_2020_2021"
    output_directory = output_directory / f"models/estimation/{year_name}/"
    output_directory.mkdir(parents=True, exist_ok=True)
    train_rumb(df_zp, output_directory, df_zp_test, intensity_cutoff, seed)


def train_rumb(
    df_zp_train, output_directory, df_zp_test=None, intensity_cutoff=None, seed=None
):

    if TORCH_INSTALLED:
        print("Torch is installed. Attempting to use GPU")
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
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    num_trees = 0
    metrics_df = pd.DataFrame({})

    # kfold cross-validation
    for i, (train_index, test_index) in enumerate(kf.split(new_df_train)):
        print(f"Fold {i+1}/{k}")
        new_df_train_t = new_df_train.iloc[train_index]
        new_df_train_v = new_df_train.iloc[test_index]
        train_data = lightgbm_data(new_df_train_t[features], new_df_train_t[choice])
        val_data = lightgbm_data(new_df_train_v[features], new_df_train_v[choice])

        try:
            model = rum_train(
                train_data,
                model_specification,
                valid_sets=[val_data],
                torch_tensors=torch_tensors,
            )
            torch_run = True
        except:
            model = rum_train(train_data, model_specification, valid_sets=[val_data])
            torch_run = False

        if not TORCH_INSTALLED:
            torch_run = False

        print(
            f"Best cross-entropy: {model.best_score} with {model.best_iteration} trees"
        )

        # storing number of trees
        num_trees += model.best_iteration

        # cv metrics
        y_pred_train = model.predict(train_data)
        if torch_run:
            y_pred_train = y_pred_train.cpu().numpy()
        y_pred_val = model.predict(val_data)
        if torch_run:
            y_pred_val = y_pred_val.cpu().numpy()
        if intensity_cutoff:
            tau_1 = model.thresholds[0]
        else:
            tau_1 = None
        metrics, intercept_mae, intercept_mse = calculate_metrics(
            new_df_train_t[choice].values.astype(int),
            y_pred_train,
            intensity_cutoff,
            tau_1,
        )
        metrics_val, _, _ = calculate_metrics(
            new_df_train_v[choice].values.astype(int),
            y_pred_val,
            intensity_cutoff,
            tau_1,
            intercept_mae,
            intercept_mse,
        )
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame(metrics, index=[f"train_{i}"])]
        )
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame(metrics_val, index=[f"val_{i}"])]
        )

    # average number of trees over the k folds
    avg_trees = int(num_trees / k)

    model_specification["general_params"]["num_iterations"] = avg_trees
    model_specification["general_params"]["early_stopping_round"] = None

    # final training on the test set
    train_data = lightgbm_data(new_df_train[features], new_df_train[choice])

    try:
        model = rum_train(train_data, model_specification, torch_tensors=torch_tensors)
        torch_run = True
    except:
        model = rum_train(train_data, model_specification)
        torch_run = False

    if not TORCH_INSTALLED:
        torch_run = False

    str_model = (
        f"intensity{intensity_cutoff}_all_vars_seed{seed}"
        if intensity_cutoff
        else f"possibility_seed{seed}"
    )
    model_name = (
        f"rumboost_model_wfh_{str_model}_"
        + datetime.now().strftime("%Y_%m_%d-%H_%M")
        + ".json"
    )
    model.save_model(
        output_directory / model_name,
    )
    # training metrics
    y_pred = model.predict(train_data)

    if torch_run:
        y_pred = y_pred.cpu().numpy()

    if intensity_cutoff:
        tau_1 = model.thresholds[0]
    else:
        tau_1 = None

    metrics, intercept_mae, intercept_mse = calculate_metrics(
        new_df_train[choice].values.astype(int), y_pred, intensity_cutoff, tau_1
    )
    metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=["train_final"])])

    if df_zp_test is not None:
        test_data = lightgbm_data(new_df_test[features], new_df_test[choice])
        predictions = model.predict(test_data)
        if torch_run:
            predictions = predictions.cpu().numpy()
        metrics_test, _, _ = calculate_metrics(
            new_df_test[choice].values.astype(int),
            predictions,
            intensity_cutoff,
            tau_1,
            intercept_mae,
            intercept_mse,
        )
        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics_test, index=["test"])])

    # save model and metrics

    file_name = (
        f"rumboost_metrics_wfh_{str_model}_"
        # + datetime.now().strftime("%Y_%m_%d-%H_%M")
        + ".csv"
    )
    metrics_df.to_csv(output_directory / file_name)

    # weird bug with plot_parameters, FileNotFoundError: [Errno 2] No such file or directory for work_time_flexibility_not_NA_fixed. path too long? plot empty?
    # plot_parameters(
    #     model,
    #     new_df_train,
    #     {"0": "Utility"},
    #     save_file=output_directory / "figures" / f"{str_model}",
    #     no_output=True,
    # )

