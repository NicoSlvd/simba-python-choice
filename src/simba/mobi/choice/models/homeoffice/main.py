from pathlib import Path
from argparse import ArgumentParser

from sklearn.model_selection import train_test_split

from simba.mobi.choice.models.homeoffice.data_loader import get_data
from simba.mobi.choice.models.homeoffice.descriptive_stats import descriptive_statistics
from simba.mobi.choice.models.homeoffice.model_estimation import (
    estimate_choice_model_telecommuting
)
from simba.mobi.choice.models.homeoffice.rumboost_estimation import (
    train_rumboost_telecommuting
)
  
#) pour demain:
# - ajouter les requirements
# - shell script pour lancer les commandes
# - tester avec fake data?


def run_home_office_in_microcensus(
    year, choice_situation, estimator, intensity_cutoff, data_intensity_only, test_size
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
        df_zp = df_zp[df_zp["telecommuting"] > 0]
    if data_intensity_only:
        # new dependant variable for comparing binary logit and ordinal logit
        # on the data where telecommuting is available
        df_zp["telecommuting"] = df_zp["telecommuting_intensity"].apply(
            lambda x: 1 if x > 0 else 0
        )
    df_zp_train, df_zp_test = (
        train_test_split(df_zp, test_size=test_size, random_state=42)
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
    if estimator == "dcm":
        estimate_choice_model_telecommuting(df_zp_train, output_directory, intensity_cutoff, df_zp_test, year)
        # descriptive_statistics(output_directory)
    elif estimator == "rumboost":
        train_rumboost_telecommuting(df_zp_train, output_directory, df_zp_test, intensity_cutoff)


if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument(
        "-y",
        "--year",
        type=int,
        choices=[2010, 2015, 2020, 2021],
        default=None,
        help="Year of the data. If not provided, all years will be used",
    )
    argparser.add_argument(
        "-c",
        "--choice_situation",
        type=str,
        default="intensity",
        choices=["intensity", "possibility"],
        help="Choice situation. The model will either estimate the intensity or the possibility of telecommuting",
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
        type=bool,
        default=False,
        help="If True, only the data where telecommuting is available will be used",
    )
    argparser.add_argument(
        "-t",
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of the data used for testing",
    )

    args = argparser.parse_args()

    run_home_office_in_microcensus(
        args.year,
        args.choice_situation,
        args.model,
        args.intensity_cutoff,
        args.data_intensity_only,
        args.test_size,
    )
