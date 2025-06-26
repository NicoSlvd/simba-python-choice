import pandas as pd
import lightgbm as lgb
import numpy as np


def define_variables(
    df_zp: pd.DataFrame,
    choice_var_name: str,
    remove_corr_vars: bool = False,
    for_binary_model: bool = False,
) -> pd.DataFrame:
    """Define variables for the RUMBoost model based on the provided DataFrame.
    Args:
        df_zp (pd.DataFrame): DataFrame containing the data.
        choice_var_name (str): Name of the choice variable.
        remove_corr_vars (bool): Whether to remove correlated variables.
        for_binary_model (bool): Whether the model is for binary choice (default is False).
    Returns:
        pd.DataFrame: DataFrame with the defined variables.
    """

    new_df = pd.DataFrame({})
    #  Utility
    new_df["single_parent_with_children"] = (df_zp["hh_type"] == 230).astype(int)
    new_df["executives_1520"] = (
        (df_zp["work_position"] == 1)
        * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))
    ).astype(int)
    new_df["german_speaking"] = (df_zp["language"] == 1).astype(int)

    # household attributes
    new_df["single_household"] = (df_zp["hh_type"] == 10).astype(int)
    new_df["number_of_children_NA"] = (
        df_zp["number_of_children_less_than_21"] == -999
    ).astype(int)
    new_df["number_of_children_not_NA"] = (
        df_zp["number_of_children_less_than_21"] != -999
    ) * df_zp["number_of_children_less_than_21"]
    new_df["hh_income_na"] = (df_zp["hh_income"] < 0).astype(int)
    new_df["hh_income_less_than_2000"] = (df_zp["hh_income"] == 1).astype(int)
    new_df["hh_income_2000_to_4000"] = (df_zp["hh_income"] == 2).astype(int)
    new_df["hh_income_4001_to_6000"] = (df_zp["hh_income"] == 3).astype(int)
    new_df["hh_income_6001_to_8000"] = (df_zp["hh_income"] == 4).astype(int)

    # mobility tools
    new_df["general_abo_halbtax"] = (
        (df_zp["has_ga"] == 1) | (df_zp["has_hta"] == 1)
    ).astype(int)

    new_df["is_falc_id_6to9_1520"] = (
        df_zp["business_sector_finance"]
        + df_zp["business_sector_services_fc"]
        + df_zp["business_sector_other_services"]
        + df_zp["business_sector_non_movers"]
    ) * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))

    # spatial attributes
    new_df["rural_work_1520"] = (
        (df_zp["urban_typology_work"] == 3)
        * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))
    ).astype(int)

    new_df["home_work_distance_car"] = (
        df_zp["car_network_distance"] * (df_zp["car_network_distance"] >= 0.0) / 1000.0
    )
    new_df["home_work_distance_car_NA"] = (df_zp["car_network_distance"] < 0.0).astype(
        int
    )

    new_df["falc_id_NA"] = (
        (
            df_zp["business_sector_agriculture"]
            + df_zp["business_sector_retail"]
            + df_zp["business_sector_gastronomy"]
            + df_zp["business_sector_finance"]
            + df_zp["business_sector_production"]
            + df_zp["business_sector_wholesale"]
            + df_zp["business_sector_services_fc"]
            + df_zp["business_sector_other_services"]
            + df_zp["business_sector_others"]
            + df_zp["business_sector_non_movers"]
        )
        == 0
    ).astype(int)
    new_df["accsib_home_not_NA_1520"] = (
        df_zp["accsib_mul_home"] * (df_zp["accsib_mul_home"] >= 0) / 100000.0
    ) * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))

    ### 2021 ###
    new_df["executives_21"] = (
        (df_zp["work_position"] == 1) * (df_zp["year"] == 2021)
    ).astype(int)
    new_df["is_falc_id_6to9_21"] = (
        df_zp["business_sector_finance"]
        + df_zp["business_sector_services_fc"]
        + df_zp["business_sector_other_services"]
        + df_zp["business_sector_non_movers"]
    ) * (df_zp["year"] == 2021)
    new_df["accsib_home_not_NA_21"] = (
        df_zp["accsib_mul_home"] * (df_zp["accsib_mul_home"] >= 0) / 100000.0
    ) * (df_zp["year"] == 2021)

    new_df["work_percentage_15"] = df_zp["work_percentage"] * (df_zp["year"] == 2015)
    new_df["business_sector_agriculture_15"] = df_zp["business_sector_agriculture"] * (
        df_zp["year"] == 2015
    )

    new_df["work_percentage_20"] = df_zp["work_percentage"] * (df_zp["year"] == 2020)

    new_df["business_sector_agriculture_21"] = df_zp["business_sector_agriculture"] * (
        df_zp["year"] == 2021
    )
    new_df["business_sector_production_21"] = df_zp["business_sector_production"] * (
        df_zp["year"] == 2021
    )
    new_df["business_sector_wholesale_21"] = df_zp["business_sector_wholesale"] * (
        df_zp["year"] == 2021
    )
    new_df["work_percentage_21"] = df_zp["work_percentage"] * (df_zp["year"] == 2021)
    new_df["work_percentage_20_21"] = df_zp["work_percentage"] * (
        (df_zp["year"] == 2021) + (df_zp["year"] == 2020)
    )
    new_df["age_21"] = df_zp["age"] * (df_zp["year"] == 2021)

    new_df["age_1520"] = df_zp["age"] * (
        (df_zp["year"] == 2015) + (df_zp["year"] == 2020)
    )

    new_df["business_sector_production_1520"] = df_zp["business_sector_production"] * (
        (df_zp["year"] == 2015) + (df_zp["year"] == 2020)
    )
    new_df["business_sector_wholesale_1520"] = df_zp["business_sector_wholesale"] * (
        (df_zp["year"] == 2015) + (df_zp["year"] == 2020)
    )

    # new vars for ordinal model
    new_df["identified_as_male"] = (df_zp["sex"] == 1).astype(int)
    new_df["nb_of_cars_NA"] = (df_zp["nb_of_cars"] < 0).astype(int)
    new_df["nb_of_cars_not_NA"] = (df_zp["nb_of_cars"] >= 0).astype(int) * df_zp[
        "nb_of_cars"
    ]
    new_df["car_avail_NA"] = (df_zp["car_avail"] < 0).astype(int)
    new_df["car_avail_not_NA_always"] = (df_zp["car_avail"] == 1).astype(int)
    new_df["car_avail_not_NA_on_demand"] = (df_zp["car_avail"] == 2).astype(int)
    new_df["has_driving_licence_NA"] = (df_zp["has_driving_licence"] < 0).astype(int)
    new_df["has_driving_licence_not_NA"] = (df_zp["has_driving_licence"] == 1).astype(
        int
    )
    new_df["work_time_flexibility_NA"] = (df_zp["work_time_flexibility"] < 0).astype(
        int
    )
    new_df["work_time_flexibility_not_NA_fixed"] = (
        df_zp["work_time_flexibility"] == 1
    ).astype(int)
    new_df["work_parking_NA"] = (df_zp["parking_place_at_work"] < 0).astype(int)
    new_df["work_parking_not_NA_free"] = (df_zp["parking_place_at_work"] == 1).astype(
        int
    )
    new_df["is_swiss"] = (df_zp["nation"] == 8100).astype(int)
    new_df["typology_work_NA"] = (df_zp["urban_typology_work"] < 0).astype(int)
    new_df["typology_work_not_NA_urban"] = (df_zp["urban_typology_work"] == 1).astype(
        int
    )
    new_df["typology_work_not_NA_rural"] = (df_zp["urban_typology_work"] == 2).astype(
        int
    )
    new_df["typology_home_urban"] = (df_zp["urban_typology_home"] == 1).astype(int)
    new_df["typology_home_rural"] = (df_zp["urban_typology_home"] == 2).astype(int)

    # pt travel times and number of transfers
    new_df["pt_travel_times_not_NA"] = (
        df_zp["pt_travel_times"] * (df_zp["pt_travel_times"] >= 0.0) / 60.0
    )
    new_df["pt_travel_times_NA"] = (df_zp["pt_travel_times"] < 0.0).astype(int)
    new_df["pt_access_times_not_NA"] = (
        df_zp["pt_access_times"] * (df_zp["pt_access_times"] >= 0.0) / 60.0
    )
    new_df["pt_access_times_NA"] = (df_zp["pt_access_times"] < 0.0).astype(int)
    new_df["pt_egress_times_not_NA"] = (
        df_zp["pt_egress_times"] * (df_zp["pt_egress_times"] >= 0.0) / 60.0
    )
    new_df["pt_egress_times_NA"] = (df_zp["pt_egress_times"] < 0.0).astype(int)
    new_df["n_transfers_not_NA"] = df_zp["n_transfers"] * (df_zp["n_transfers"] >= 0.0)
    new_df["n_transfers_NA"] = (df_zp["n_transfers"] < 0.0).astype(int)
    new_df["pt_tt_or_transfers_NA"] = (
        (df_zp["pt_travel_times"] < 0.0)
        + (df_zp["pt_access_times"] < 0.0)
        + (df_zp["pt_egress_times"] < 0.0)
        + (df_zp["n_transfers"] < 0.0)
        > 0
    ).astype(int)

    other_columns = [
        "no_post_school_educ",
        "secondary_education",
        "tertiary_education",
        "hh_size",
        "WP",
        choice_var_name,
    ]
    new_df.loc[:, other_columns] = df_zp.loc[:, other_columns]

    if for_binary_model:
        # for binary model, we only keep the variables that are not correlated
        new_df = new_df.drop(
            columns=[
                "hh_size",
                "identified_as_male",
                "nb_of_cars_NA",
                "nb_of_cars_not_NA",
                "car_avail_NA",
                "car_avail_not_NA_always",
                "car_avail_not_NA_on_demand",
                "has_driving_licence_NA",
                "has_driving_licence_not_NA",
                "work_time_flexibility_NA",
                "work_time_flexibility_not_NA_fixed",
                "work_parking_NA",
                "work_parking_not_NA_free",
                "is_swiss",
                "typology_work_NA",
                "typology_work_not_NA_urban",
                "typology_work_not_NA_rural",
                "typology_home_urban",
                "typology_home_rural",
                "pt_travel_times_not_NA",
                "pt_travel_times_NA",
                "pt_access_times_not_NA",
                "pt_access_times_NA",
                "pt_egress_times_not_NA",
                "pt_egress_times_NA",
                "n_transfers_not_NA",
                "n_transfers_NA",
                "pt_tt_or_transfers_NA",
                "business_sector_agriculture_21",
            ]
        )
    elif remove_corr_vars:
        # remove correlated variables
        new_df = new_df.drop(
            columns=[
                "home_work_distance_car",
                "home_work_distance_car_NA",
                "hh_size",
                "identified_as_male",
                "car_avail_NA",
                "work_parking_NA",
                "typology_work_NA",
                "typology_work_not_NA_rural",
                "typology_home_rural",
                "pt_travel_times_NA",
                "pt_access_times_NA",
                "pt_egress_times_NA",
                "n_transfers_NA",
                # "nb_of_cars_NA",
                "business_sector_agriculture_21",
                "has_driving_licence_NA",
                "has_driving_licence_not_NA",
                "number_of_children_NA",
                "typology_home_urban"
            ]
        )

    return new_df


def get_rumboost_rum_structure(new_df):

    variables = [
        c
        for c in new_df.columns
        if (
            c
            not in [
                "telecommuting_intensity",
                "telecommuting",
                "HHNR",
                "WP",
                "work_percentage_20_21",
                "work_percentage_20",
            ]
        )
        and ("1520" not in c)
        and ("_15" not in c)
    ]

    monotone_constraints = [0] * len(variables)
    interaction_constraints = [[i] for i, _ in enumerate(variables)]

    # rum structure
    rum_structure = [
        {
            "utility": [0],
            "variables": variables,
            "boosting_params": {
                "monotone_constraints_method": "advanced",
                "max_depth": 1,
                "n_jobs": -1,
                "learning_rate": 0.1,
                "verbose": -1,
                # "min_data_in_leaf": 1,
                # "min_sum_hessian_in_leaf": 0,
                "monotone_constraints": monotone_constraints,
                "interaction_constraints": interaction_constraints,
            },
            "shared": False,
        }
    ]

    boost_from_parameter_space = [False]

    return rum_structure, boost_from_parameter_space


def get_lin_rumboost_rum_structure(new_df):

    variables = [
        c
        for c in new_df.columns
        if (
            c
            not in [
                "telecommuting_intensity",
                "telecommuting",
                "HHNR",
                "WP",
                "work_percentage_20_21",
                "work_percentage_20",
            ]
        )
        and ("1520" not in c)
        and ("_15" not in c)
    ]

    lr = np.minimum(0.1, 1 / len(variables))  # learning rate for linear RUMBoost

    # rum structure
    rum_structure = [
        {
            "utility": [0],
            "variables": [v],
            "boosting_params": {
                "monotone_constraints_method": "advanced",
                "max_depth": 1,
                "n_jobs": -1,
                "learning_rate": lr,
                "verbose": -1,
                # "min_data_in_leaf": 1,
                # "min_sum_hessian_in_leaf": 0,
                "monotone_constraints": [0],
                "interaction_constraints": [0],
                "max_bin": 11,  # max 10 split points
                "min_data_in_leaf": 1,
            },
            "shared": False,
        }
        for v in variables
    ]

    boost_from_parameter_space = [
        True if new_df[v].nunique() > 2 else False for v in variables
    ]

    return rum_structure, boost_from_parameter_space


def get_rumboost_model_spec(new_df, intensity_cutoff=None, lin_rumboost=False) -> dict:

    if lin_rumboost:
        rum_structure, boost_from_parameter_space = get_lin_rumboost_rum_structure(
            new_df
        )
    else:
        rum_structure, boost_from_parameter_space = get_rumboost_rum_structure(new_df)

    general_params = {
        "n_jobs": -1,
        "num_classes": (
            100 // intensity_cutoff + 1 if intensity_cutoff else 2
        ),  # important
        "verbosity": 0,  # specific RUMBoost parameter
        "num_iterations": 2000,
        "early_stopping_round": 100,
        "boost_from_parameter_space": boost_from_parameter_space,
        "max_booster_to_update": len(boost_from_parameter_space),
    }

    model_specification = {
        "general_params": general_params,
        "rum_structure": rum_structure,
    }

    if intensity_cutoff:
        model_specification["ordinal_logit"] = {
            "model": "proportional_odds",
            "optim_interval": 20,
        }

    return model_specification


def lightgbm_data(features, choice):

    lgb_dataset = lgb.Dataset(features, label=choice, free_raw_data=False)

    return lgb_dataset
