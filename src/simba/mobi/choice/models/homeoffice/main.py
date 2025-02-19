import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..", ".."))

from pathlib import Path
from argparse import ArgumentParser, BooleanOptionalAction

from sklearn.model_selection import train_test_split

from src.simba.mobi.choice.models.homeoffice.data_loader import get_data
from src.simba.mobi.choice.models.homeoffice.descriptive_stats import (
    descriptive_statistics,
)
from src.simba.mobi.choice.models.homeoffice.model_definition import (
    define_telecommuting_intensity_variable
)
from src.simba.mobi.choice.models.homeoffice.model_estimation import (
    estimate_choice_model_telecommuting,
)
from src.simba.mobi.choice.models.homeoffice.rumboost_estimation import (
    train_rumboost_telecommuting,
)
from src.simba.mobi.choice.models.homeoffice.linearised_model_estimation import (
    estimate_linearised_model_telecommuting,
)
import gc
import torch


def run_home_office_in_microcensus(
    year, estimator, intensity_cutoff, data_intensity_only, test_size, seed,
) -> None:
    """Generate 2015 data"""
    input_directory = Path(
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("input")
        .joinpath("homeoffice")
    )
    input_directory.mkdir(parents=True, exist_ok=True)
    df_zp = get_data(input_directory, intensity_cutoff=intensity_cutoff)
    df_zp = df_zp[df_zp["year"] == year] if year else df_zp
    if data_intensity_only or intensity_cutoff:
        #only keep the data where telecommuting is available
        df_zp = df_zp[df_zp["telecommuting"] > 0]
        #remove missing observations
        df_zp = df_zp[df_zp["percentage_telecommuting"] >= 0]
    if data_intensity_only:
        # new dependant variable for comparing binary logit and ordinal logit
        # on the data where telecommuting is available
        df_zp["telecommuting"] = df_zp["telecommuting_intensity"].apply(
            lambda x: 1 if x > 0 else 0
        )
    # need to recompute teleworking intensity variable to a different level

    # split the data into training and testing
    df_zp_train, df_zp_test = (
        train_test_split(df_zp, test_size=test_size, random_state=seed)
        if test_size
        else (df_zp, None)
    )
    """Estimation results"""
    output_directory = (
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("output")
        .joinpath("homeoffice")
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    #estimate choice model
    if estimator == "dcm":
        estimate_choice_model_telecommuting(
            df_zp_train, output_directory, intensity_cutoff, df_zp_test, year, seed
        )
        # descriptive_statistics(output_directory)
    #estimate rumboost
    elif estimator == "rumboost":
        train_rumboost_telecommuting(
            df_zp_train, output_directory, intensity_cutoff, df_zp_test, year, seed
        )
    elif estimator == "linearised_model":
        estimate_linearised_model_telecommuting(
            df_zp_train, output_directory, intensity_cutoff, df_zp_test, year, seed
        )


if __name__ == "__main__":
    # models = ["dcm", "rumboost"]
    models = ["rumboost"]
    # models = ["linearised_model"]
    # intensity_cutoffs = [20, 10, 0]
    intensity_cutoffs = [20]

    for model in models:
        for cutoff in intensity_cutoffs:
            for seed in range(10): # 10 runs to mitigate randomness
                if model == "dcm":
                    path = Path(__file__).parent.parent.parent.joinpath("data").joinpath(
                        "output"
                    ).joinpath("homeoffice").joinpath("models").joinpath("estimation").joinpath("2021")
                    path = path / f"metrics_wfh_intensity{cutoff}_all_vars_seed{seed}_.csv" if cutoff else path / f"metrics_wfh_possibility_seed{seed}_.csv"
                elif model == "rumboost":
                    path = Path(__file__).parent.parent.parent.joinpath("data").joinpath(
                        "output"
                    ).joinpath("homeoffice").joinpath("models").joinpath("estimation").joinpath("2021")
                    path = path / f"rumboost_metrics_wfh_intensity{cutoff}_all_vars_seed{seed}_.csv" if cutoff else path / f"rumboost_metrics_wfh_possibility_seed{seed}_.csv"
                elif model == "linearised_model":
                    path = Path(__file__).parent.parent.parent.joinpath("data").joinpath(
                        "output"
                    ).joinpath("homeoffice").joinpath("models").joinpath("estimation").joinpath("2021")
                    path = path / f"rumboost_metrics_wfh_intensity{cutoff}_seed{seed}_.csv" if cutoff else path / f"rumboost_metrics_wfh_possibility_seed{seed}_.csv"

                if path.exists():
                    print(f"File already exists, skipping {model} with intensity_cutoff={cutoff}, seed={seed}")
                    continue
                else:
                    print(f"Running main.py with model={model} and intensity_cutoff={cutoff}, seed={seed}")
                # subprocess.run(["python", MAIN_FILE, "--model", model, "--intensity_cutoff", str(cutoff)])
                argparser = ArgumentParser()
                argparser.add_argument(
                    "-y",
                    "--year",
                    type=int,
                    choices=[2010, 2015, 2020, 2021],
                    default=2021,
                    help="Year of the data. If not provided, all years will be used",
                )
                argparser.add_argument(
                    "-m",
                    "--model",
                    type=str,
                    default="dcm",
                    choices=["dcm", "rumboost"],
                    help="Underlying model. Either 'dcm' or 'rumboost'",
                )
                argparser.add_argument(
                    "-i",
                    "--intensity_cutoff",
                    type=int,
                    default=20,
                    help="Cutoff for defining the intensity of telecommuting variable, by default 20 percent",
                )
                argparser.add_argument(
                    "-d",
                    "--data_intensity_only",
                    action=BooleanOptionalAction,
                )
                argparser.add_argument(
                    "-t",
                    "--test_size",
                    type=float,
                    default=0.2,
                    help="Fraction of the data used for testing",
                )

                args = argparser.parse_args()

                if cutoff and 100 % cutoff != 0:
                    raise ValueError("Intensity cutoff must be a divisor of 100")

                run_home_office_in_microcensus(
                    int(args.year),
                    model,
                    cutoff,
                    args.data_intensity_only,
                    float(args.test_size),
                    seed,
                )

                gc.collect()
                torch.cuda.empty_cache()
