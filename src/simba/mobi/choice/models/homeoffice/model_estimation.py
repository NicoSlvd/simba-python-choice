from datetime import datetime
from pathlib import Path

import biogeme.biogeme as bio
import biogeme.database as db
import biogeme.models as models
import pandas as pd
from biogeme.expressions import bioMax
from biogeme.expressions import bioMin
from biogeme.expressions import log
from biogeme.expressions import Elem

from src.simba.mobi.choice.models.homeoffice.model_definition import define_variables
from src.simba.mobi.choice.models.homeoffice.model_definition import get_dict_betas
from src.simba.mobi.choice.models.homeoffice.descriptive_stats import calculate_metrics
from src.simba.mobi.choice.utils.biogeme import estimate_in_directory


def estimate_choice_model_telecommuting(
    df_zp: pd.DataFrame, output_directory: Path, intensity_cutoff: int = None, df_zp_test: pd.DataFrame = None, year: int = None
) -> None:
    year_name = f"{year}" if year else "2015_2020_2021"
    output_directory = output_directory / f"models/estimation/{year_name}/"
    output_directory.mkdir(parents=True, exist_ok=True)
    run_estimation(df_zp, output_directory, intensity_cutoff, df_zp_test)


def run_estimation(
    df_zp: pd.DataFrame,
    output_directory: Path,
    intensity_cutoff: int = None,
    df_zp_test: pd.DataFrame = None,
) -> None:
    """
    :author: Antonin Danalet, based on the example '01logit.py' by Michel Bierlaire, EPFL, on biogeme.epfl.ch

    A binary logit model or ordinal logit model if intensity_cutoff is specified on the ability to work from home.
    intensity_cutoff is a percentage that represents the boundaries of the telecommuting intensity categories."""
    database = db.Database("persons", df_zp)

    define_variables(database)

    dict_betas = get_dict_betas(intensity_cutoff)

    # The following statement allows you to use the names of the variable as Python variable.
    globals().update(database.variables)

    U = (
        dict_betas["alternative_specific_constant"]
        + models.piecewiseFormula(age_1520, [18, 35])
        + models.piecewiseFormula(work_percentage_15, [0, 95, 101])
        + dict_betas["b_executives_1520"] * executives_1520
        + dict_betas["b_german_speaking"] * german_speaking
        + dict_betas["b_no_post_school_education"] * no_post_school_educ
        + dict_betas["b_secondary_education"] * secondary_education
        + dict_betas["b_tertiary_education"] * tertiary_education
        + dict_betas["b_rural_work_1520"] * rural_work_1520
        + dict_betas["b_hh_income_na"] * hh_income_na
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_less_than_2000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_2000_to_4000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_4001_to_6000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_6001_to_8000
        + dict_betas["b_number_of_children"] * number_of_children_not_NA
        + dict_betas["b_number_of_children_NA"] * number_of_children_NA
        + dict_betas["b_single_household"] * single_household
        + dict_betas["b_general_abo_halbtax"] * general_abo_halbtax
        + models.piecewiseFormula(home_work_distance_car, [0, 0.15])
        + dict_betas["b_home_work_distance_car_NA"] * home_work_distance_car_NA
        + dict_betas["b_is_agriculture_1_15"] * business_sector_agriculture_15
        + dict_betas["b_is_production_1520"] * business_sector_production_1520
        + dict_betas["b_is_wohlesale_1520"] * business_sector_wholesale_1520
        + dict_betas["b_is_falc_id_6to9_1520"] * is_falc_id_6to9_1520
        + dict_betas["b_falc_id_NA"] * falc_id_NA
        + dict_betas["beta_accsib_home_not_NA_5_10_1520"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 5.0), 5.0))
        + dict_betas["beta_accsib_home_not_NA_10_24_1520"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 10.0), 14.0))
        + dict_betas["beta_work_percentage_0_95_20"]
        * bioMax(0.0, bioMin((work_percentage_20 - 0.0), 95.0))
        + dict_betas["beta_work_percentage_95_101_20_21"]
        * bioMax(0.0, bioMin((work_percentage_20_21 - 95.0), 6.0))
        + models.piecewiseFormula(age_21, [18, 35])
        + dict_betas["beta_work_percentage_0_95_21"]
        * bioMax(0.0, bioMin((work_percentage_21 - 0.0), 95.0))
        + dict_betas["b_executives_21"] * executives_21
        + dict_betas["b_is_agriculture_1_21"] * business_sector_agriculture_21
        + dict_betas["b_is_production_1_21"] * business_sector_production_21
        + dict_betas["b_is_wohlesale_1_21"] * business_sector_wholesale_21
        + dict_betas["b_is_falc_id_6to9_1_21"] * is_falc_id_6to9_21
        + dict_betas["beta_accsib_home_not_NA_5_10_21"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_21 - 5.0), 5.0))
        + dict_betas["beta_accsib_home_not_NA_10_24_21"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_21 - 10.0), 14.0))
    )
    U_no_telecommuting = 0

    # Scale associated with 2020 is estimated
    scale = (
        (year == 2015)
        + (year == 2020) * dict_betas["scale_2020"]
        + (year == 2021) * dict_betas["scale_2021"]
    )

    # Thresholds if ordinal logit
    if intensity_cutoff:
        tau_1 = dict_betas["tau_1"]
        for i in range(1, 100 // intensity_cutoff):
            # add the difference between the thresholds
            globals()[f"tau_{i+1}"] = dict_betas[f"tau_{i}"] + dict_betas[f"diff_{i}{i+1}"]

    # Associate utility functions with the numbering of alternatives
    V = {1: scale * U, 0: U_no_telecommuting}  # 1: Yes or sometimes, 2: No

    # All alternatives are supposed to be always available
    av = {1: 1, 0: 1}

    # Definition of the model. This is the contribution of each observation to the log likelihood function.
    # Choice variable: "telecommuting"
    if intensity_cutoff:
        model_name = "wfh_intensity_model_sbb"

        the_proba = models.ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[i for i in range(100 // intensity_cutoff + 1)],
            tau_parameter=tau_1,
        )

        the_chosen_proba = Elem(the_proba, telecommuting_intensity)

        logprob = log(the_chosen_proba)

    else:
        model_name = "wfh_possibility_model_sbb"

        logprob = models.loglogit(V, av, telecommuting)

    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.modelName = model_name

    # Calculate the null log likelihood for reporting.
    the_biogeme.calculateNullLoglikelihood(av)

    results = estimate_in_directory(the_biogeme, output_directory)

    df_parameters = results.getEstimatedParameters()
    str_model = "intensity" if intensity_cutoff else "possibility"
    file_name = (
        f"parameters_dcm_wfh_{str_model}_" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".csv"
    )
    df_parameters.to_csv(output_directory / file_name)

    if intensity_cutoff:
        # Generate individual expressions for each probability
        for i in range(1, 100 // intensity_cutoff + 1):
            globals()[f"proba_{i}"] = Elem(the_proba, i)

        all_probs = {
            "prob_0": proba_0,
            "prob_1": proba_1,
            "prob_2": proba_2,
            "prob_3": proba_3,
            "prob_4": proba_4,
            "prob_5": proba_5,
        }

        beta_values = results.getBetaValues()

        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = "wfh_intensity_model_sbb_train"

        # Simulate probabilities for each class
        results_ = biogeme_obj.simulate(beta_values)

        # calculate metrics
        metrics = calculate_metrics(df_zp["telecommuting_intensity"], results_.values, intensity_cutoff)

        metrics_df = pd.DataFrame(metrics, index=['train'])
    
    else:
        beta_values = results.getBetaValues()

        results_ = the_biogeme.simulate(beta_values)

        metrics = calculate_metrics(df_zp["telecommuting"], results_.values)

        metrics_df = pd.DataFrame(metrics, index=['train'])

    #test the model on the test set
    if df_zp_test is not None and intensity_cutoff:

        database_test = db.Database('persons_test', df_zp_test)

        globals().update(database_test.variables)

        the_proba = ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[0, 1, 2, 3, 4, 5],
            tau_parameter=tau_1,
        )
        # Generate individual expressions for each probability
        for i in range(1, 100 // intensity_cutoff + 1):
            globals()[f"proba_{i}"] = Elem(the_proba, i)

        all_probs = {"prob_0": proba_0, "prob_1": proba_1, "prob_2": proba_2, "prob_3": proba_3, "prob_4": proba_4, "prob_5": proba_5}

        beta_values = results.getBetaValues()


        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database_test, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = "wfh_intensity_model_sbb_test"

        # Simulate probabilities for each class
        results_ = biogeme_obj.simulate(beta_values)

        # calculate metrics
        metrics = calculate_metrics(df_zp_test['telecommuting_intensity'], results_.values, intensity_cutoff)

        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=['test'])])

    elif df_zp_test is not None:
        
        database_test = db.Database('persons_test', df_zp_test)

        globals().update(database_test.variables)

        biogeme_obj = bio.BIOGEME(database_test, logprob)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False

        biogeme_obj.modelName = "wfh_possibility_model_sbb_test"
        results_ = biogeme_obj.simulate(beta_values)

        metrics = calculate_metrics(df_zp_test["telecommuting"], results_.values)

        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=['test'])])

    file_name = (
        f"metrics_wfh_{str_model}_" + datetime.now().strftime("%Y_%m_%d-%H_%M") + ".csv"
    )
    metrics_df.to_csv(output_directory / file_name)


