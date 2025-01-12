import pandas as pd
import numpy as np
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.models as models
from biogeme.expressions import bioMax
from biogeme.expressions import bioMin
from biogeme.expressions import Beta
from biogeme.expressions import Elem
from biogeme.expressions import log


def define_telecommuting_variable(row):
    """Defines a choice variable with value 1 if the person is allowed to telecommute
    (answer "yes" - 1 - or answer "sometimes" - 2)"""
    telecommuting = 0
    if (row["telecommuting_is_possible"] == 1) or (
        row["telecommuting_is_possible"] == 2
    ):
        telecommuting = 1
    return telecommuting


def define_telecommuting_intensity_variable(row, intensity_cutoff):
    """Defines an ordinal choice variable which represents the intensity of telecommuting.
    The variable is defined as i if telecommuting_intensity is between i*intensity_cutoff and (i+1)*intensity_cutoff.
    for i up to 100/intensity_cutoff. No telecommuting is represented by 0."""
    if row["percentage_telecommuting"] > 0:
        telecommuting_intensity = np.ceil(
            np.minimum(100, row["percentage_telecommuting"]) / intensity_cutoff
        )
    else:
        telecommuting_intensity = 0
    return telecommuting_intensity


def define_variables(database: pd.DataFrame) -> None:
    globals().update(database.variables)

    #  Utility
    single_parent_with_children = database.DefineVariable(
        "single_parent_with_children", hh_type == 230
    )
    # Definition of new variables
    executives_1520 = database.DefineVariable(
        "executives_1520", (work_position == 1) * ((year == 2020) + (year == 2015))
    )
    german_speaking = database.DefineVariable("german_speaking", language == 1)

    # household attributes
    single_household = database.DefineVariable("single_household", hh_type == 10)
    number_of_children_NA = database.DefineVariable(
        "number_of_children_NA", number_of_children_less_than_21 == -999
    )
    number_of_children_not_NA = database.DefineVariable(
        "number_of_children_not_NA",
        (number_of_children_less_than_21 != -999) * number_of_children_less_than_21,
    )

    hh_income_na = database.DefineVariable("hh_income_na", hh_income < 0)
    hh_income_less_than_2000 = database.DefineVariable(
        "hh_income_less_than_2000", hh_income == 1
    )
    hh_income_2000_to_4000 = database.DefineVariable(
        "hh_income_2000_to_4000", hh_income == 2
    )
    hh_income_4001_to_6000 = database.DefineVariable(
        "hh_income_4001_to_6000", hh_income == 3
    )
    hh_income_6001_to_8000 = database.DefineVariable(
        "hh_income_6001_to_8000", hh_income == 4
    )

    # mobility tools
    general_abo_halbtax = database.DefineVariable(
        "general_abo_halbtax", (has_ga == 1) | (has_hta == 1)
    )

    is_falc_id_6to9_1520 = database.DefineVariable(
        "is_falc_id_6to9_1520",
        (
            business_sector_finance
            + business_sector_services_fc
            + business_sector_other_services
            + business_sector_non_movers
        )
        * ((year == 2020) + (year == 2015)),
    )
    # spatial attributes
    rural_work_1520 = database.DefineVariable(
        "rural_work_1520",
        (urban_typology_work == 3) * ((year == 2020) + (year == 2015)),
    )

    home_work_distance_car = database.DefineVariable(
        "home_work_distance_car",
        car_network_distance * (car_network_distance >= 0.0) / 1000.0,
    )
    home_work_distance_car_NA = database.DefineVariable(
        "home_work_distance_car_NA", car_network_distance < 0.0
    )

    falc_id_NA = database.DefineVariable(
        "falc_id_NA",
        (
            business_sector_agriculture
            + business_sector_retail
            + business_sector_gastronomy
            + business_sector_finance
            + business_sector_production
            + business_sector_wholesale
            + business_sector_services_fc
            + business_sector_other_services
            + business_sector_others
            + business_sector_non_movers
        )
        == 0,
    )
    accsib_home_not_NA_1520 = database.DefineVariable(
        "accsib_home_not_NA_1520",
        (accsib_mul_home * (accsib_mul_home >= 0) / 100000.0)
        * ((year == 2020) + (year == 2015)),
    )

    ### 2021 ###
    executives_21 = database.DefineVariable(
        "executives_21", (work_position == 1) * (year == 2021)
    )
    is_falc_id_6to9_21 = database.DefineVariable(
        "is_falc_id_6to9_21",
        (
            business_sector_finance
            + business_sector_services_fc
            + business_sector_other_services
            + business_sector_non_movers
        )
        * (year == 2021),
    )
    accsib_home_not_NA_21 = database.DefineVariable(
        "accsib_home_not_NA_21",
        (accsib_mul_home * (accsib_mul_home >= 0) / 100000.0) * (year == 2021),
    )

    work_percentage_15 = database.DefineVariable(
        "work_percentage_15", work_percentage * (year == 2015)
    )
    business_sector_agriculture_15 = database.DefineVariable(
        "business_sector_agriculture_15", business_sector_agriculture * (year == 2015)
    )

    work_percentage_20 = database.DefineVariable(
        "work_percentage_20", work_percentage * (year == 2020)
    )

    business_sector_agriculture_21 = database.DefineVariable(
        "business_sector_agriculture_21", business_sector_agriculture * (year == 2021)
    )
    business_sector_production_21 = database.DefineVariable(
        "business_sector_production_21", business_sector_production * (year == 2021)
    )
    business_sector_wholesale_21 = database.DefineVariable(
        "business_sector_wholesale_21", business_sector_wholesale * (year == 2021)
    )
    work_percentage_21 = database.DefineVariable(
        "work_percentage_21", work_percentage * (year == 2021)
    )
    work_percentage_20_21 = database.DefineVariable(
        "work_percentage_20_21", work_percentage * ((year == 2021) + (year == 2020))
    )
    age_21 = database.DefineVariable("age_21", age * (year == 2021))

    age_1520 = database.DefineVariable(
        "age_1520", age * ((year == 2015) + (year == 2020))
    )
    business_sector_production_1520 = database.DefineVariable(
        "business_sector_production_1520",
        business_sector_production * ((year == 2015) + (year == 2020)),
    )
    business_sector_wholesale_1520 = database.DefineVariable(
        "business_sector_wholesale_1520",
        business_sector_wholesale * ((year == 2015) + (year == 2020)),
    )
    # new variables from linearised model
    home_work_distance_car_short = database.DefineVariable(
        "home_work_distance_car_short",
        home_work_distance_car * (home_work_distance_car <= 0.19),
    )
    home_work_distance_car_long = database.DefineVariable(
        "home_work_distance_car_long",
        log(home_work_distance_car) * (home_work_distance_car > 0.19),
    )


def get_dict_betas(intensity_cutoff: int = None) -> dict:
    # Parameters to be estimated (global)
    dict_betas = {
        "alternative_specific_constant": Beta(
            "alternative_specific_constant", 0, None, None, 0
        ),
        "scale_2020": Beta("scale_2020", 1, 0.001, None, 1),  # not estimated
        "scale_2021": Beta("scale_2021", 1, 0.001, None, 1),  # not estimated
        # person attributes
        "b_executives_1520": Beta("b_executives_1520", 0, None, None, 0),
        "b_german_speaking": Beta("b_german_speaking", 0, None, None, 0),
        "b_no_post_school_education": Beta(
            "b_no_post_school_education", 0, None, None, 0
        ),
        "b_secondary_education": Beta("b_secondary_education", 0, None, None, 0),
        "b_tertiary_education": Beta("b_tertiary_education", 0, None, None, 0),
        # household attributes
        "b_number_of_children": Beta("b_number_of_children", 0, None, None, 0),
        "b_number_of_children_NA": Beta("b_number_of_children_NA", 0, None, None, 0),
        "b_single_household": Beta("b_single_houshold", 0, None, None, 0),
        "b_hh_income_na": Beta("b_hh_income_na", 0, None, None, 0),
        "b_hh_income_8000_or_less": Beta("b_hh_income_8000_or_less", 0, None, None, 0),
        # mobility tools
        "b_general_abo_halbtax": Beta("b_general_abo_halbtax", 0, None, None, 0),
        # work-place attributes
        "b_is_agriculture_1_15": Beta("b_is_agriculture_1_15", 0, None, None, 0),
        "b_is_production_1520": Beta("b_is_production_1520", 0, None, None, 0),
        "b_is_wohlesale_1520": Beta("b_is_wohlesale_1520", 0, None, None, 0),
        "b_is_falc_id_6to9_1520": Beta("b_is_falc_id_6to9_1520", 0, None, None, 0),
        "b_falc_id_NA": Beta("b_falc_id_NA", 0, None, None, 0),
        # spatial attributes
        "b_rural_work_1520": Beta("b_rural_work_1520", 0, None, None, 0),
        "b_home_work_distance_car_NA": Beta(
            "b_home_work_distance_car_NA", 0, None, None, 0
        ),
        "b_executives_21": Beta("b_executives_21", 0, None, None, 0),
        # work-place attributes
        "b_is_agriculture_1_21": Beta("b_is_agriculture_1_21", 0, None, None, 1),
        "b_is_production_1_21": Beta("b_is_production_1_21", 0, None, None, 0),
        "b_is_wohlesale_1_21": Beta("b_is_wohlesale_1_21", 0, None, None, 0),
        "b_is_falc_id_6to9_1_21": Beta("b_is_falc_id_6to9_1_21", 0, None, None, 0),
        "beta_work_percentage_0_95_21": Beta(
            "beta_work_percentage_0_95_21", 0, None, None, 1
        ),
        "beta_work_percentage_95_101_20_21": Beta(
            "beta_work_percentage_95_101_20_21", 0, None, None, 0
        ),
        "beta_work_percentage_0_95_20": Beta(
            "beta_work_percentage_0_95_20", 0, None, None, 0
        ),
        "beta_accsib_home_not_NA_5_10_1520": Beta(
            "beta_accsib_home_not_NA_5_10_1520", 0, None, None, 0
        ),
        "beta_accsib_home_not_NA_10_24_1520": Beta(
            "beta_accsib_home_not_NA_10_24_1520", 0, None, None, 0
        ),
        "beta_accsib_home_not_NA_5_10_21": Beta(
            "beta_accsib_home_not_NA_5_10_21", 0, None, None, 0
        ),
        "beta_accsib_home_not_NA_10_24_21": Beta(
            "beta_accsib_home_not_NA_10_24_21", 0, None, None, 1
        ),
        # new betas from linearisation of rumboost
        "b_home_work_distance_car_long": Beta(
            "b_home_work_distance_car_long", 0, None, None, 0
        ),
    }

    # thresholds
    if intensity_cutoff:
        dict_betas["tau_1"] = Beta(
            "tau_1", -2, None, 0, 0
        )  # threshold 1 <= 0 according to biogeme ordinal logit model
        for i in range(1, 100 // intensity_cutoff):
            dict_betas[f"tau_{i}"] = Beta(f"tau_{i}", -2, None, 0, 0)
            dict_betas[f"diff_{i}{i+1}"] = Beta(
                f"diff_{i}{i+1}", 1, 0, None, 0
            )  # difference between subsequent thresholds
            dict_betas[f"tau_{i+1}"] = (
                dict_betas[f"diff_{i}{i+1}"] + dict_betas[f"tau_{i}"]
            )

        dict_betas["alternative_specific_constant"] = Beta(
            "alternative_specific_constant", 0, None, None, 1
        )  # not estimated, asc is implicitly estimated within the thresholds

    return dict_betas


def return_model(df_zp, intensity_cutoff, df_zp_test=None, linearised=False):

    if df_zp_test is not None:
        database = db.Database("persons", df_zp_test)

        define_variables(database)

        dict_betas = get_dict_betas(intensity_cutoff)

        # The following statement allows you to use the names of the variable as Python variable.
        globals().update(database.variables)
    else:
        database = db.Database("persons", df_zp)

        define_variables(database)

        dict_betas = get_dict_betas(intensity_cutoff)

        # The following statement allows you to use the names of the variable as Python variable.
        globals().update(database.variables)

    U = (
        (
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
            + models.piecewiseFormula(
                home_work_distance_car_short, [0.0001, 0.005, 0.19]
            )  # linearised from rumboost
            + dict_betas["b_home_work_distance_car_long"] * home_work_distance_car_long
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
            # + dict_betas["beta_work_percentage_95_101_20_21"]
            # * bioMax(0.0, bioMin((work_percentage_20_21 - 95.0), 6.0))
            + models.piecewiseFormula(
                age_21, [17, 21, 35, 61, 83]
            )  # linearised from rumboost
            + models.piecewiseFormula(work_percentage_21, [3.5, 4.5, 28, 31, 39, 84])
            + dict_betas["b_executives_21"] * executives_21
            + dict_betas["b_is_agriculture_1_21"] * business_sector_agriculture_21
            + dict_betas["b_is_production_1_21"] * business_sector_production_21
            + dict_betas["b_is_wohlesale_1_21"] * business_sector_wholesale_21
            + dict_betas["b_is_falc_id_6to9_1_21"] * is_falc_id_6to9_21
            + models.piecewiseFormula(
                accsib_home_not_NA_21, [0.2, 0.45, 0.6, 1.4, 9.1, 15, 95, 16.05]
            )  # linearised from rumboost
        )
        if linearised
        else (
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
        )
    )
    U_no_telecommuting = 0

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

        the_proba = models.ordered_logit(
            continuous_value=V,
            list_of_discrete_values=[i for i in range(100 // intensity_cutoff + 1)],
            tau_parameter=tau_1,
        )
        # Generate individual expressions for each probability
        for i in range(0, 100 // intensity_cutoff + 1):
            globals()[f"proba_{i}"] = Elem(the_proba, i)

        all_probs = {
            f"prob_{i}": globals()[f"proba_{i}"]
            for i in range(100 // intensity_cutoff + 1)
        }

        # Create the BIOGEME object, using all_probabilities for simulation
        biogeme_obj = bio.BIOGEME(database, all_probs)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = (
            ("wfh_intensity_model_sbb_test")
            if df_zp_test is not None
            else "wfh_intensity_model_sbb_train"
        )

    else:
        logprob = models.loglogit(V, av, telecommuting)

        # Create the Biogeme object
        biogeme_obj = bio.BIOGEME(database, logprob)
        biogeme_obj.generate_pickle = False
        biogeme_obj.generate_html = False
        biogeme_obj.modelName = (
            "wfh_possibility_model_sbb_test"
            if df_zp_test is not None
            else "wfh_possibility_model_sbb_train"
        )

    return biogeme_obj
