from pathlib import Path

from simba.mobi.choice.models.homeoffice.homeoffice_possibility.data_loader_sbb import get_data
from simba.mobi.choice.models.homeoffice.homeoffice_possibility.descriptive_stats_sbb import descriptive_statistics
from simba.mobi.choice.models.homeoffice.homeoffice_possibility.model_estimation_sbb import (
    estimate_choice_model_telecommuting,
)


def run_home_office_in_microcensus() -> None:
    """Generate 2015 data"""
    input_directory = Path(
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("input")
        .joinpath("homeoffice_possibility")
    )
    input_directory.mkdir(parents=True, exist_ok=True)
    df_zp = get_data(input_directory)
    """Estimation results"""
    output_directory = (
        Path(__file__)
        .parent.parent.parent.joinpath("data")
        .joinpath("output")
        .joinpath("homeoffice_possibility")
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    estimate_choice_model_telecommuting(df_zp, output_directory)
    descriptive_statistics(output_directory)


if __name__ == "__main__":
    run_home_office_in_microcensus()
