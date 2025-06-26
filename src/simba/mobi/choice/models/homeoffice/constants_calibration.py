import pandas as pd
import numpy as np
import torch
import lightgbm as lgb
import biogeme.biogeme as bio
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy.special import expit

from src.simba.mobi.choice.models.homeoffice.data_loader import get_data
from src.simba.mobi.choice.models.homeoffice.model_definition import (
    define_telecommuting_intensity_variable,
    return_model,
)
from src.simba.mobi.choice.models.homeoffice.rumboost_definition import (
    define_variables,
)
from src.simba.mobi.choice.models.homeoffice.descriptive_stats import (
    calculate_metrics,
)

from src.simba.mobi.rumboost.rumboost import RUMBoost
from src.simba.mobi.rumboost.utility_plotting import weights_to_plot_v2


def calibrate_constants(
    model_type="dcm",
    intensity_cutoff=0,
    seed=0,
    tol=0.01,
    results_directory=os.getcwd() + "/homeoffice/models/estimation/2021/",
    save_results=True,
):
    """
    Calibrate the constants (ASCs or thresholds) of the model to reproduce market shares.
    This is intended to be used with a model that has already been estimated, and to make
    the model suitable for forecasting.

    Parameters
    ----------
    model_type: str
        The type of model to be calibrated. It can be either "dcm" or "rumboost".
    intensity_cutoff: int
        The percentage cutoff to be used for defining telecommuting intensity variable.
        By example, if intensity_cutoff=20, the telecommuting intensity variable will have 6 levels.
        0 for no telecommuting, 1 for 1-20% of telecommuting, 2 for 21-40%, etc.
    seed: int
        The seed to be used for splitting the dataset into training and testing.
    tol: float
        The tolerance for the calibration of the constants.
    results_directory: str
        The directory where the results of the estimation are stored.
    all_vars: str
        The string to be used for the results file name if the models were estimated with all variables for ordinal models.
    save_results: bool
        Whether to save the results of the calibration.

    Returns
    -------
    betas: dict
        The updated betas of the model.
    model: object
        The model with the updated constants.
    """
    # load data and split into training and testing
    if "all_data" in model_type:
        test_size = 0
    else:
        test_size = 0.2
    data_train, data_test = load_and_split_dataset(
        intensity_cutoff=intensity_cutoff, year=2021, test_size=test_size, seed=seed
    )

    observed_ms = observed_market_shares(data_train, intensity_cutoff)

    # add correct directory to results_directory
    modelling_task = "intensity" if intensity_cutoff else "possibility"
    results_directory = results_directory + f"{model_type}/{modelling_task}/"

    # load model and return the betas as dict and df (for dcm) and dataset with new variables (for rumboost)
    model, betas, betas_to_update, data_train, data_test, df_dcm = load_model_and_betas(
        results_directory,
        model_type,
        intensity_cutoff,
        seed,
        data_train,
        data_test,
    )

    # initial market shares prediction
    predicted_ms = predicted_market_shares(
        data_train, model, intensity_cutoff, betas=betas
    )

    # if thresholds, need to update one after the other
    for i, b in enumerate(betas_to_update):
        tries = 0
        t = tol
        lr = 0.1 if intensity_cutoff else 1  # learning rate
        # while the difference between observed and predicted market shares for the current constant is greater than the tolerance
        while np.abs(observed_ms[i] - predicted_ms[i]) > t:
            new_b = update_constant(
                b, observed_ms[i], predicted_ms[i], lr=lr
            )  # update constant
            b = new_b
            if "dcm" in model_type:  # dcm
                if intensity_cutoff:
                    betas_to_update[i] = new_b  # update the threshold
                    if i == 0:
                        betas["tau_1"] = new_b  # update the first threshold
                    else:
                        betas[f"tau_1_diff_{i}"] = (
                            new_b - betas_to_update[i - 1]
                        )  # update the difference between thresholds
                else:
                    betas["alternative_specific_constant"] = new_b  # update the ASC
            else:  # rumboost
                if intensity_cutoff:
                    model.thresholds[i] = new_b  # update the threshold
                else:
                    model.asc = new_b  # update the ASC

            # recompute the predicted market shares
            predicted_ms = predicted_market_shares(
                data_train, model, intensity_cutoff, betas=betas
            )
            
            tries += 1
            if tries % 100 == 0:  # increase tolerance if too many tries
                t *= 2
                print(f"Increasing tolerance to {t}")
        if intensity_cutoff:
            print(f"threshold {i} calibrated for model {model_type} {all_vars} and run {seed}")
        else:
            print(f"ASC calibrated for model {model_type} {all_vars} and run {seed}")

    if save_results:  # save the results
        # recompute metrics
        metrics_df = recompute_metrics(
            data_train,
            model,
            betas,
            intensity_cutoff,
            data_test,
        )

        # save metrics
        metrics_df.to_csv(results_directory + f"metrics_seed{seed}_calibrated.csv")
        if "dcm" in model_type:
            #save parameters
            save_file = results_directory + f"parameters_seed{seed}_calibrated.csv"
            df_dcm.loc[:, "Value"] = list(betas.values())
            df_dcm.to_csv(save_file)
        else:  # rumboost
            save_file = results_directory + f"model_seed{seed}_calibrated.json"
            model.save_model(save_file)

    return model, betas


def update_constant(beta, observed_ms, predicted_ms, lr):
    """
    update the alternative specific constant using the heuristic method of Train (2003)

    Parameters
    ----------
    beta: float
        the current value of the alternative specific constant or threshold to be updated
    observed_ms: float
        the observed market shares of the dependent variable
        (intensity of telecommuting if ordinal model, possibility of telecommuting if binary model)
    predicted_ms: float
        the predicted market shares of the dependent variable
        (intensity of telecommuting if ordinal model, possibility of telecommuting if binary model)
    lr: float
        the learning rate to be used for the update

    Returns
    -------
    betas: list
        the updated betas
    """
    # Update the value of the alternative specific constant using the heuristic method of Train (2003)
    # Adding a learning rate because otherwise can fall into infinite loop
    beta = beta + lr * np.log(observed_ms / predicted_ms)

    return beta


def predicted_market_shares(data_train, model, intensity_cutoff, betas=None):
    """Compute the predicted market shares of the training dataset"""
    if isinstance(model, RUMBoost):
        if intensity_cutoff:
            train_set = lgb.Dataset(
                data_train.drop(columns=["telecommuting_intensity"]),
                label=data_train["telecommuting_intensity"],
                free_raw_data=False,
            )
            preds = model.predict(train_set)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            predicted_ms = (preds * data_train["WP"].values[:, None]).sum(
                axis=0
            ) / data_train["WP"].sum()
        else:
            train_set = lgb.Dataset(
                data_train.drop(columns=["telecommuting"]),
                label=data_train["telecommuting"],
                free_raw_data=False,
            )
            preds = expit(model.predict(train_set, utilities=True) + model.asc)
            if isinstance(preds, torch.Tensor):
                preds = preds.cpu().numpy()
            predicted_ms = (preds * data_train["WP"].values[:, None]).sum(
                axis=0
            ) / data_train["WP"].sum()
    else:
        if intensity_cutoff:
            results = model.simulate(betas)
            predicted_ms = (results.values * data_train["WP"].values[:, None]).sum(
                axis=0
            ) / data_train["WP"].sum()
        else:
            # the biogeme objects only simulate for chosen alternatives, but we need probabilities for class 0 only
            results = model.simulate(betas)
            results_0 = np.where(
                (data_train["telecommuting"].values == 1)[:, None],
                np.exp(results.values),
                1 - np.exp(results.values),
            )
            predicted_ms = (results_0 * data_train["WP"].values[:, None]).sum(
                axis=0
            ) / data_train["WP"].sum()

    return predicted_ms


def observed_market_shares(data_train, intensity_cutoff, weights="WP"):
    """Compute the observed market shares of the training dataset"""
    if intensity_cutoff:
        observed_ms = (
            data_train.groupby("telecommuting_intensity")[weights].sum()
            / data_train[weights].sum()
        ).values
    else:
        observed_ms = (
            data_train.groupby("telecommuting")[weights].sum()
            / data_train[weights].sum()
        ).loc[1].reshape(1)
    return observed_ms


def load_and_split_dataset(intensity_cutoff=0, year=2021, test_size=0.2, seed=0):
    """Load the dataset and split it into training and testing"""
    input_directory = Path(
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("input")
        .joinpath("homeoffice")
    )
    input_directory.mkdir(parents=True, exist_ok=True)
    df_zp = get_data(input_directory, intensity_cutoff=intensity_cutoff)
    df_zp = df_zp[df_zp["year"] == year] if year else df_zp
    if intensity_cutoff:
        df_zp = df_zp[df_zp["telecommuting"] > 0]
    # need to recompute teleworking intensity variable to a different level
    # if the pre-processed dataset persons.csv is not regenerated from scratch
    if intensity_cutoff and len(df_zp["telecommuting_intensity"].unique()) != (
        100 // intensity_cutoff + 1
    ):
        df_zp["telecommuting_intensity"] = df_zp.apply(
            define_telecommuting_intensity_variable,
            axis=1,
            intensity_cutoff=intensity_cutoff,
        )
    # split the data into training and testing
    df_zp_train, df_zp_test = (
        train_test_split(df_zp, test_size=test_size, random_state=seed)
        if test_size
        else (df_zp, None)
    )

    return df_zp_train, df_zp_test


def load_model_and_betas(
    results_directory,
    model_type,
    intensity_cutoff,
    seed,
    data_train,
    data_test=None,
    df_dcm=None,
):
    """Load the model and the betas from the estimation results"""
    if "dcm" in model_type:
        results_file = f"{results_directory}/parameters_seed{seed}.csv"
        with open(results_file, "r") as f:
            df_dcm = pd.read_csv(f, index_col=0)
        betas = df_dcm.loc[:, "Value"].to_dict()  # all betas
        if intensity_cutoff:
            betas_to_update = [betas["tau_1"]]
            for i in range(1, 100 // intensity_cutoff):
                betas_to_update.append(betas_to_update[-1] + betas[f"tau_1_diff_{i}"])
        else:
            betas_to_update = [betas["alternative_specific_constant"]]
        model = return_model(data_train, intensity_cutoff)

    elif "rumboost" in model_type:
        choice = "telecommuting_intensity" if intensity_cutoff else "telecommuting"
        data_train = define_variables(data_train, choice)
        data_test = define_variables(data_test, choice) if data_test is not None else None
        results_file = f"{results_directory}/model_seed{seed}.json"
        model = RUMBoost(model_file=results_file)
        betas = None

        # compute the implicit asc for binary model
        if intensity_cutoff:
            betas_to_update = model.thresholds.tolist()
        else:
            betas_to_update = [model.asc]

    return model, betas, betas_to_update, data_train, data_test, df_dcm


def recompute_metrics(
    data_train, model, betas, intensity_cutoff, data_test=None
):
    """Recompute the metrics of the model after calibration"""
    if isinstance(model, RUMBoost):  # rumboost
        y_true_train = (
            data_train["telecommuting_intensity"].values.astype(int)
            if intensity_cutoff
            else data_train["telecommuting"].values.astype(int)
        )
        train_set = (
            lgb.Dataset(
                data_train.drop(columns=["telecommuting_intensity"]),
                label=data_train["telecommuting_intensity"],
                free_raw_data=False,
            )
            if intensity_cutoff
            else lgb.Dataset(
                data_train.drop(columns=["telecommuting"]),
                label=data_train["telecommuting"],
                free_raw_data=False,
            )
        )
        y_pred_train = model.predict(train_set)
        if isinstance(y_pred_train, torch.Tensor):
            y_pred_train = y_pred_train.cpu().numpy()
        tau_1 = model.thresholds[0] if intensity_cutoff else None
        metrics_train, intercept_mae, intercept_mse = calculate_metrics(
            y_true_train, y_pred_train, intensity_cutoff, tau_1
        )
        if data_test is not None:  # no test set, return only train metrics
            y_true_test = (
                data_test["telecommuting_intensity"].values.astype(int)
                if intensity_cutoff
                else data_test["telecommuting"].values.astype(int)
            )
            test_set = (
                lgb.Dataset(
                    data_test.drop(columns=["telecommuting_intensity"]),
                    label=data_test["telecommuting_intensity"],
                    free_raw_data=False,
                )
                if intensity_cutoff
                else lgb.Dataset(
                    data_test.drop(columns=["telecommuting"]),
                    label=data_test["telecommuting"],
                    free_raw_data=False,
                )
            )
            y_pred_test = model.predict(test_set)
            if isinstance(y_pred_test, torch.Tensor):
                y_pred_test = y_pred_test.cpu().numpy()
            metrics_test, _, _ = calculate_metrics(
                y_true_test,
                y_pred_test,
                intensity_cutoff,
                tau_1,
                intercept_mae,
                intercept_mse,
            )
            metrics_df = pd.DataFrame(
                [metrics_train, metrics_test], index=["train", "test"]
            )
        else:
            metrics_df = pd.DataFrame([metrics_train], index=["train"])
    else:
        y_true_train = (
            data_train["telecommuting_intensity"].values.astype(int)
            if intensity_cutoff
            else data_train["telecommuting"].values.astype(int)
        )
        y_pred_train = model.simulate(betas).values
        tau_1 = betas["tau_1"] if intensity_cutoff else None
        metrics_train, intercept_mae, intercept_mse = calculate_metrics(
            y_true_train, y_pred_train, intensity_cutoff, tau_1
        )
        if data_test is not None:  
            model_pred = return_model(
                data_train, intensity_cutoff, data_test
            )
            y_pred_test = model_pred.simulate(betas).values
            y_true_test = (
                data_test["telecommuting_intensity"].values.astype(int)
                if intensity_cutoff
                else data_test["telecommuting"].values.astype(int)
            )
            metrics_test, _, _ = calculate_metrics(
                y_true_test,
                y_pred_test,
                intensity_cutoff,
                tau_1,
                intercept_mae,
                intercept_mse,
            )
            metrics_df = pd.DataFrame(
                [metrics_train, metrics_test], index=["train", "test"]
            )
        else:
            metrics_df = pd.DataFrame([metrics_train], index=["train"])

    return metrics_df
