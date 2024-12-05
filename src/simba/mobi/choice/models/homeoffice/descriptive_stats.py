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

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

def emae(y_true, y_pred, intensity_cutoff):
    distance_abs = np.array([[np.abs(i - y) for i in range(100 // intensity_cutoff + 1)] for y in y_true])
    return np.mean(np.sum(distance_abs * y_pred, axis=1))

def emse(y_true, y_pred, intensity_cutoff):
    distance_squared = np.array([[(i - y)**2 for i in range(100 // intensity_cutoff + 1)] for y in y_true])
    return np.mean(np.sum(distance_squared * y_pred, axis=1))
    
def analyse_preds(df_zp_test, intensity_cutoff, results, V, tau_1):

    database_test = db.Database("persons_test", df_zp_test)

    define_variables(database_test)

    if intensity_cutoff:
        the_proba = models.ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[0, 1, 2, 3, 4, 5],
            tau_parameter=tau_1,
        )

        the_chosen_proba = Elem(the_proba, telecommuting_intensity)

        beta_values = results.getBetaValues()

        biogeme_obj = bio.BIOGEME(database_test, the_chosen_proba)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False

        biogeme_obj.modelName = "intensity_teleworking_ordinal_logit_test"
        results_ = biogeme_obj.simulate(beta_values)

        print(np.log(results_).mean())

        the_proba = ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[0, 1, 2, 3, 4, 5],
            tau_parameter=tau_1,
        )
        # Generate individual expressions for each probability
        proba_0 = Elem(the_proba, 0)
        proba_1 = Elem(the_proba, 1)
        proba_2 = Elem(the_proba, 2)
        proba_3 = Elem(the_proba, 3)
        proba_4 = Elem(the_proba, 4)
        proba_5 = Elem(the_proba, 5)

        all_probs = {"prob_0": proba_0, "prob_1": proba_1, "prob_2": proba_2, "prob_3": proba_3, "prob_4": proba_4, "prob_5": proba_5}

        # Load beta values from training results
        results = bioResults(pickle_file="output/data/intensity_teleworking_ordinal_logit_train_all_sign_5.pickle")
        beta_values = results.getBetaValues()


        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database_test, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = "intensity_teleworking_ordinal_logit_test"

        # Simulate probabilities for each class
        results_ = biogeme_obj.simulate(beta_values)

        print(np.abs(df['work_home_days'] - np.argmax(results_, axis=1)).mean())
        print(np.mean((df['work_home_days'] - np.argmax(results_, axis=1))**2))

        distance_squared = np.array([[(i - choice)**2 for i in range(6)] for choice in df['work_home_days']])
        distance_abs = np.array([[np.abs(i - choice) for i in range(6)] for choice in df['work_home_days']])

        print(np.mean(np.sum(distance_squared * results_.values, axis=1)))
        print(np.mean(np.sum(distance_abs * results_.values, axis=1)))
        
