from pathlib import Path

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
from biogeme.expressions import bioMax
from biogeme.expressions import bioMin
from biogeme.expressions import log
from biogeme.expressions import Elem
import numpy as np
from matplotlib import pyplot as plt


def descriptive_statistics(output_directory: Path) -> None:
    visualize_work_percentage(output_directory)
    visualize_accessibility(output_directory)


def visualize_accessibility(output_directory: Path) -> None:
    def f15(x_axis):
        return models.piecewiseFunction(x_axis, [0, 5, 10, 24], [0, -0.0419, 0.0847])

    def f21(x_axis):
        return models.piecewiseFunction(x_axis, [0, 5, 10, 24], [0, 0.0442, 0.0])

    x = np.arange(5, 24, 1)

    y21 = []
    for i in range(len(x)):
        y21.append(f21(x[i]))

    y15 = []
    for i in range(len(x)):
        y15.append(f15(x[i]))

    plt.plot(x, y21, c="red", ls="", ms=5, marker=".", label="2021")
    plt.plot(x, y15, c="green", ls="", ms=5, marker="*", label="2015+2020")
    plt.legend(loc="lower right")
    ax = plt.gca()
    plt.xlabel("Accessibility (* 100'000)")
    plt.ylabel("Nutzen")
    # ax.set_ylim([-1, 2])

    file_name = "effect_accessibility" + ".png"
    plt.savefig(output_directory / file_name)


def visualize_work_percentage(output_directory: Path) -> None:
    def f15(x_axis):
        return models.piecewiseFunction(x_axis, [0, 90, 101], [-0.0124, 0.0782])

    def f20(x_axis):
        return models.piecewiseFunction(x_axis, [0, 90, 101], [-0.00765, 0.0222])

    def f21(x_axis):
        return models.piecewiseFunction(x_axis, [0, 90, 101], [0.0, 0.0222])

    x = np.arange(0, 100, 1)

    y21 = []
    for i in range(len(x)):
        y21.append(f21(x[i]))

    y20 = []
    for i in range(len(x)):
        y20.append(f20(x[i]))

    y15 = []
    for i in range(len(x)):
        y15.append(f15(x[i]))

    plt.plot(x, y21, c="red", ls="", ms=5, marker=".", label="2021")
    plt.plot(x, y20, c="blue", ls="", ms=5, marker="*", label="2020")
    plt.plot(x, y15, c="green", ls="", ms=5, marker="*", label="2015")
    plt.legend(loc="lower right")
    ax = plt.gca()
    plt.xlabel("Alter")
    plt.ylabel("Nutzen")
    # ax.set_ylim([-1, 2])

    file_name = "effect_work_percentage" + ".png"
    plt.savefig(output_directory / file_name)


def mae(y_true, y_pred, tau_1, intercept=None):
    if tau_1 is not None:
        raw_value = tau_1 - np.log(y_pred[:, 0] / (1 - y_pred[:, 0]))
        # need to re-estimate the intercept
        # V is treated as fixed therefore the intercept is the median of the residuals with MAE
        intercept = np.median(y_true - raw_value) if not intercept else intercept
        y_pred = raw_value + intercept
    return np.mean(np.abs(y_true - y_pred)), intercept


def mse(y_true, y_pred, tau_1, intercept=None):
    if tau_1 is not None:
        raw_value = tau_1 - np.log(y_pred[:, 0] / (1 - y_pred[:, 0]))
        # need to re-estimate the intercept
        # V is treated as fixed therefore the intercept is the mean of the residuals with MSE
        intercept = (y_true - raw_value).mean() if not intercept else intercept
        y_pred = raw_value + intercept
    return np.mean((y_true - y_pred) ** 2), intercept


def emae(y_true, y_pred, intensity_cutoff):
    distance_abs = np.array(
        [[np.abs(i - y) for i in range(100 // intensity_cutoff + 1)] for y in y_true]
    )
    return np.mean(np.sum(distance_abs * y_pred, axis=1))


def emse(y_true, y_pred, intensity_cutoff):
    distance_squared = np.array(
        [[(i - y) ** 2 for i in range(100 // intensity_cutoff + 1)] for y in y_true]
    )
    return np.mean(np.sum(distance_squared * y_pred, axis=1))


def cel(y_true, y_pred):
    index = range(y_pred.shape[0])
    return -np.mean(np.log(y_pred[index, y_true]))


def bin_cel(y_true, y_pred):
    if (y_pred > 0).all():
        return -np.mean(
            np.where(
                y_true == 1, np.log(y_pred.reshape(-1)), np.log(1 - y_pred.reshape(-1))
            )
        )
    return -np.mean(y_pred)


def calculate_metrics(y_true, y_pred, intensity_cutoff=None, tau_1=None, intercept_mae=None, intercept_mse=None):
    metrics = {}

    if intensity_cutoff:
        cel_value = cel(y_true, y_pred)
        mae_value, intercept_mae = mae(y_true, y_pred, tau_1, intercept_mae)
        mse_value, intercept_mse = mse(y_true, y_pred, tau_1, intercept_mse)
        emae_value = emae(y_true, y_pred, intensity_cutoff)
        emse_value = emse(y_true, y_pred, intensity_cutoff)
        metrics["cel"] = cel_value
        metrics["mae"] = mae_value
        metrics["mse"] = mse_value
        metrics["emae"] = emae_value
        metrics["emse"] = emse_value
    else:
        cel_value = bin_cel(y_true, y_pred)
        metrics["cel"] = cel_value

    return metrics, intercept_mae, intercept_mse
