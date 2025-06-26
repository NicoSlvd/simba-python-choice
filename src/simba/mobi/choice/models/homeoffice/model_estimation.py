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
    df_zp: pd.DataFrame,
    output_directory: Path,
    intensity_cutoff: int = None,
    df_zp_test: pd.DataFrame = None,
    year: int = None,
    seed: int = None,
) -> None:
    run_estimation(df_zp, output_directory, intensity_cutoff, df_zp_test, seed)


def run_estimation(
    df_zp: pd.DataFrame,
    output_directory: Path,
    intensity_cutoff: int = None,
    df_zp_test: pd.DataFrame = None,
    seed: int = None,
) -> None:
    """
    :author: Antonin Danalet, based on the example '01logit.py' by Michel Bierlaire, EPFL, on biogeme.epfl.ch

    A binary logit model or ordinal logit model if intensity_cutoff is specified on the ability to work from home.
    intensity_cutoff is a percentage that represents the boundaries of the telecommuting intensity categories.
    """
    database = db.Database("persons", df_zp)

    define_variables(database)

    dict_betas = get_dict_betas(intensity_cutoff)

    # The following statement allows you to use the names of the variable as Python variable.
    globals().update(database.variables)

    U = (
        dict_betas["alternative_specific_constant"]
        # + models.piecewiseFormula(age_1520, [18, 35])
        # + models.piecewiseFormula(work_percentage_15, [0, 95, 101])
        # + dict_betas["b_executives_1520"] * executives_1520
        + dict_betas["b_german_speaking"] * german_speaking
        + dict_betas["b_no_post_school_education"] * no_post_school_educ
        + dict_betas["b_secondary_education"] * secondary_education
        + dict_betas["b_tertiary_education"] * tertiary_education
        # + dict_betas["b_rural_work_1520"] * rural_work_1520
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
        # + dict_betas["b_is_agriculture_1_15"] * business_sector_agriculture_15
        # + dict_betas["b_is_production_1520"] * business_sector_production_1520
        # + dict_betas["b_is_wohlesale_1520"] * business_sector_wholesale_1520
        # + dict_betas["b_is_falc_id_6to9_1520"] * is_falc_id_6to9_1520
        + dict_betas["b_falc_id_NA"] * falc_id_NA
        # + dict_betas["beta_accsib_home_not_NA_5_10_1520"]
        # * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 5.0), 5.0))
        # + dict_betas["beta_accsib_home_not_NA_10_24_1520"]
        # * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 10.0), 14.0))
        # + dict_betas["beta_work_percentage_0_95_20"]
        # * bioMax(0.0, bioMin((work_percentage_20 - 0.0), 95.0))
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
        if not intensity_cutoff
        else
        dict_betas["alternative_specific_constant"]
        # + models.piecewiseFormula(age_1520, [18, 35])
        # + models.piecewiseFormula(work_percentage_15, [0, 95, 101])
        # + dict_betas["b_executives_1520"] * executives_1520
        + dict_betas["b_german_speaking"] * german_speaking
        + dict_betas["b_no_post_school_education"] * no_post_school_educ #P-value > 0.8
        + dict_betas["b_secondary_education"] * secondary_education
        + dict_betas["b_tertiary_education"] * tertiary_education
        # + dict_betas["b_rural_work_1520"] * rural_work_1520
        + dict_betas["b_hh_income_na"] * hh_income_na
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_less_than_2000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_2000_to_4000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_4001_to_6000
        + dict_betas["b_hh_income_8000_or_less"] * hh_income_6001_to_8000
        + dict_betas["b_number_of_children"] * number_of_children_not_NA
        # + dict_betas["b_number_of_children_NA"] * number_of_children_NA #P-value > 0.8
        + dict_betas["b_single_household"] * single_household
        + dict_betas["b_general_abo_halbtax"] * general_abo_halbtax #P-value > 0.2
        # + models.piecewiseFormula(home_work_distance_car, [0, 0.15]) # correlated with travel times
        # + dict_betas["b_home_work_distance_car_NA"] * home_work_distance_car_NA #correlated with travel times NA
        # + dict_betas["b_is_agriculture_1_15"] * business_sector_agriculture_15
        # + dict_betas["b_is_production_1520"] * business_sector_production_1520
        # + dict_betas["b_is_wohlesale_1520"] * business_sector_wholesale_1520
        # + dict_betas["b_is_falc_id_6to9_1520"] * is_falc_id_6to9_1520
        + dict_betas["b_falc_id_NA"] * falc_id_NA
        # + dict_betas["beta_accsib_home_not_NA_5_10_1520"]
        # * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 5.0), 5.0))
        # + dict_betas["beta_accsib_home_not_NA_10_24_1520"]
        # * bioMax(0.0, bioMin((accsib_home_not_NA_1520 - 10.0), 14.0))
        # + dict_betas["beta_work_percentage_0_95_20"]
        # * bioMax(0.0, bioMin((work_percentage_20 - 0.0), 95.0))
        + dict_betas["beta_work_percentage_95_101_20_21"]
        * bioMax(0.0, bioMin((work_percentage_20_21 - 95.0), 6.0))
        + models.piecewiseFormula(age_21, [18, 35])
        + dict_betas["beta_work_percentage_0_95_21"]
        * bioMax(0.0, bioMin((work_percentage_21 - 0.0), 95.0))
        + dict_betas["b_executives_21"] * executives_21
        # + dict_betas["b_is_agriculture_1_21"] * business_sector_agriculture_21 #values are all 0
        + dict_betas["b_is_production_1_21"] * business_sector_production_21
        + dict_betas["b_is_wohlesale_1_21"] * business_sector_wholesale_21
        + dict_betas["b_is_falc_id_6to9_1_21"] * is_falc_id_6to9_21
        + dict_betas["beta_accsib_home_not_NA_5_10_21"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_21 - 5.0), 5.0))
        + dict_betas["beta_accsib_home_not_NA_10_24_21"]
        * bioMax(0.0, bioMin((accsib_home_not_NA_21 - 10.0), 14.0))
        # + dict_betas["b_hh_size"] * hh_size # removed because corellated with number of children
        # + dict_betas["b_identified_as_male"] * identified_as_male # removed because not used in SBB models
        # + dict_betas["b_nb_of_cars_NA"] * nb_of_cars_NA #P-value > 0.2
        + dict_betas["b_nb_of_cars_not_NA"] * nb_of_cars_not_NA
        # + dict_betas["b_car_avail_NA"] * car_avail_NA
        + dict_betas["b_car_avail_not_NA_always"] * car_avail_not_NA_always
        + dict_betas["b_car_avail_not_NA_on_demand"] * car_avail_not_NA_on_demand
        # + dict_betas["b_has_driving_licence_NA"] * has_driving_licence_NA #P-value > 0.2
        # + dict_betas["b_has_driving_licence_not_NA"] * has_driving_licence_not_NA #P-value > 0.8
        + dict_betas["b_work_time_flexibility_NA"] * work_time_flexibility_NA
        + dict_betas["b_work_time_flexibility_not_NA_fixed"] * work_time_flexibility_not_NA_fixed
        # + dict_betas["b_work_parking_NA"] * work_parking_NA
        + dict_betas["b_work_parking_not_NA_free"] * work_parking_not_NA_free
        + dict_betas["b_is_swiss"] * is_swiss
        # + dict_betas["b_typology_work_NA"] * typology_work_NA
        + dict_betas["b_typology_work_not_NA_urban"] * typology_work_not_NA_urban
        # + dict_betas["b_typology_work_not_NA_rural"] * typology_work_not_NA_rural #removed because corellated with urban typology
        # + dict_betas["b_typology_home_urban"] * typology_home_urban #P-value > 0.8
        # + dict_betas["b_typology_home_rural"] * typology_home_rural  #removed because corellated with urban typology
        + dict_betas["b_pt_travel_times_not_NA"] * pt_travel_times_not_NA
        # + dict_betas["b_pt_travel_times_NA"] * pt_travel_times_NA # grouped because corellated with other tt NA variables
        + dict_betas["b_pt_access_times_not_NA"] * pt_access_times_not_NA #P-value > 0.2
        # + dict_betas["b_pt_access_times_NA"] * pt_access_times_NA # grouped because corellated with other tt NA variables
        + dict_betas["b_pt_egress_times_not_NA"] * pt_egress_times_not_NA
        # + dict_betas["b_pt_egress_times_NA"] * pt_egress_times_NA # grouped because corellated with other tt NA variables
        + dict_betas["b_n_transfers_not_NA"] * n_transfers_not_NA #P-value > 0.2
        # + dict_betas["b_n_transfers_NA"] * n_transfers_NA # grouped because corellated with other tt NA variables
        + dict_betas["b_pt_tt_or_transfers_NA"] * pt_tt_or_transfers_NA
    )
    U_no_telecommuting = 0

    # Scale associated with 2020 is estimated
    # scale = (
    #     (year == 2015)
    #     + (year == 2020) * dict_betas["scale_2020"]
    #     + (year == 2021) * dict_betas["scale_2021"]
    # )

    # Thresholds if ordinal logit
    if intensity_cutoff:
        tau_1 = dict_betas["tau_1"]
        for i in range(1, 100 // intensity_cutoff):
            globals()[f"tau_{i+1}"] = dict_betas[f"tau_{i+1}"]

    # Associate utility functions with the numbering of alternatives
    if intensity_cutoff:
        V = U
    else:
        V = {1: U, 0: U_no_telecommuting}  # 1: Yes or sometimes, 2: No

    # All alternatives are supposed to be always available
    av = {1: 1, 0: 1}

    # Definition of the model. This is the contribution of each observation to the log likelihood function.
    # Choice variable: "telecommuting"
    if intensity_cutoff:
        model_name = f"wfh_intensity_seed{seed}"

        the_proba = models.ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[i for i in range(100 // intensity_cutoff + 1)],
            tau_parameter=tau_1,
        )

        the_chosen_proba = Elem(the_proba, telecommuting_intensity)

        logprob = log(the_chosen_proba)

    else:
        model_name = f"wfh_possibility_seed{seed}"

        logprob = models.loglogit(V, av, telecommuting)

    # Create the Biogeme object
    the_biogeme = bio.BIOGEME(database, logprob)
    the_biogeme.modelName = model_name

    # Calculate the null log likelihood for reporting.
    the_biogeme.calculateNullLoglikelihood(av)

    results = estimate_in_directory(the_biogeme, output_directory)

    df_parameters = results.getEstimatedParameters()
    str_model = f"seed{seed}"
    file_name = (
        f"parameters_{str_model}"
        + ".csv"
    )
    df_parameters.to_csv(output_directory / file_name)

    if intensity_cutoff:
        # Generate individual expressions for each probability
        for i in range(0, 100 // intensity_cutoff + 1):
            globals()[f"proba_{i}"] = Elem(the_proba, i)

        all_probs = {
            f"prob_{i}": globals()[f"proba_{i}"]
            for i in range(100 // intensity_cutoff + 1)
        }

        beta_values = results.getBetaValues()
        tau_1_res = beta_values["tau_1"]

        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = "wfh_intensity_model_all_vars_and_tt_sbb_train"

        # Simulate probabilities for each class
        results_ = biogeme_obj.simulate(beta_values)

        # calculate metrics
        metrics, intercept_mae, intercept_mse = calculate_metrics(
            df_zp["telecommuting_intensity"].values.astype(int),
            results_.values,
            intensity_cutoff,
            tau_1_res,
        )

        metrics_df = pd.DataFrame(metrics, index=["train"])

    else:
        beta_values = results.getBetaValues()

        results_ = the_biogeme.simulate(beta_values)

        metrics, _, _ = calculate_metrics(
            df_zp["telecommuting"].values.astype(int), results_.values
        )

        metrics_df = pd.DataFrame(metrics, index=["train"])

    # test the model on the test set
    if df_zp_test is not None and intensity_cutoff:

        database_test = db.Database("persons_test", df_zp_test)

        define_variables(database_test)

        globals().update(database_test.variables)

        the_proba = models.ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[i for i in range(100 // intensity_cutoff + 1)],
            tau_parameter=tau_1,
        )
        # Generate individual expressions for each probability
        for i in range(1, 100 // intensity_cutoff + 1):
            globals()[f"proba_{i}"] = Elem(the_proba, i)

        all_probs = {
            f"prob_{i}": globals()[f"proba_{i}"]
            for i in range(100 // intensity_cutoff + 1)
        }

        beta_values = results.getBetaValues()

        tau_1_res = beta_values["tau_1"]

        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database_test, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = "wfh_intensity_model_all_vars_and_tt_sbb_test"

        # Simulate probabilities for each class
        results_ = biogeme_obj.simulate(beta_values)

        # calculate metrics
        metrics, _, _ = calculate_metrics(
            df_zp_test["telecommuting_intensity"].values.astype(int),
            results_.values,
            intensity_cutoff,
            tau_1_res,
            intercept_mae,
            intercept_mse,
        )

        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=["test"])])

    elif df_zp_test is not None:

        database_test = db.Database("persons_test", df_zp_test)

        define_variables(database_test)

        globals().update(database_test.variables)

        biogeme_obj = bio.BIOGEME(database_test, logprob)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False

        biogeme_obj.modelName = "wfh_possibility_model_sbb_test"
        results_ = biogeme_obj.simulate(beta_values)

        metrics, _, _ = calculate_metrics(
            df_zp_test["telecommuting"].values.astype(int), results_.values
        )

        metrics_df = pd.concat([metrics_df, pd.DataFrame(metrics, index=["test"])])

    file_name = (
        f"metrics_{str_model}"
        + ".csv"
    )
    metrics_df.to_csv(output_directory / file_name)
