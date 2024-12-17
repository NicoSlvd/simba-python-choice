import pandas as pd
import lightgbm as lgb

def define_variables(df_zp, choice_var_name) -> pd.DataFrame:

    new_df = pd.DataFrame({})
    #  Utility
    new_df["single_parent_with_children"] = (df_zp["hh_type"] == 230).astype(int)
    new_df["executives_1520"] = ((df_zp["work_position"] == 1) * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))).astype(int)
    new_df["german_speaking"] = (df_zp["language"] == 1).astype(int)

    # household attributes
    new_df["single_household"] = (df_zp["hh_type"] == 10).astype(int)
    new_df["number_of_children_NA"] = (df_zp["number_of_children_less_than_21"] == -999).astype(int)
    new_df["number_of_children_not_NA"] = (df_zp["number_of_children_less_than_21"] != -999) * df_zp["number_of_children_less_than_21"]
    new_df["hh_income_na"] = (df_zp["hh_income"] < 0).astype(int)
    new_df["hh_income_less_than_2000"] = (df_zp["hh_income"] == 1).astype(int)
    new_df["hh_income_2000_to_4000"] = (df_zp["hh_income"] == 2).astype(int)
    new_df["hh_income_4001_to_6000"] = (df_zp["hh_income"] == 3).astype(int)
    new_df["hh_income_6001_to_8000"] = (df_zp["hh_income"] == 4).astype(int)

    # mobility tools
    new_df["general_abo_halbtax"] = ((df_zp["has_ga"] == 1) | (df_zp["has_hta"] == 1)).astype(int)

    new_df["is_falc_id_6to9_1520"] = (
        (df_zp["business_sector_finance"]
        + df_zp["business_sector_services_fc"]
        + df_zp["business_sector_other_services"]
        + df_zp["business_sector_non_movers"])
        * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))
    )

    # spatial attributes
    new_df["rural_work_1520"] = (
        (df_zp["urban_typology_work"] == 3) * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))
    ).astype(int)

    new_df["home_work_distance_car"] = (
        df_zp["car_network_distance"] * (df_zp["car_network_distance"] >= 0.0) / 1000.0
    )
    new_df["home_work_distance_car_NA"] = (df_zp["car_network_distance"] < 0.0).astype(int)
    
    new_df["fal_id_NA"] = (
        (df_zp["business_sector_agriculture"]
        + df_zp["business_sector_retail"]
        + df_zp["business_sector_gastronomy"]
        + df_zp["business_sector_finance"]
        + df_zp["business_sector_production"]
        + df_zp["business_sector_wholesale"]
        + df_zp["business_sector_services_fc"]
        + df_zp["business_sector_other_services"]
        + df_zp["business_sector_others"]
        + df_zp["business_sector_non_movers"])
        == 0
    ).astype(int)
    new_df["accsib_home_not_NA_1520"] = (
        df_zp["accsib_mul_home"] * (df_zp["accsib_mul_home"] >= 0) / 100000.0
    ) * ((df_zp["year"] == 2020) + (df_zp["year"] == 2015))

    ### 2021 ###
    new_df["executives_21"] = ((df_zp["work_position"] == 1) * (df_zp["year"] == 2021)).astype(int)
    new_df["is_falc_id_6to9_21"] = (
        (df_zp["business_sector_finance"]
        + df_zp["business_sector_services_fc"]
        + df_zp["business_sector_other_services"]
        + df_zp["business_sector_non_movers"])
        * (df_zp["year"] == 2021)
    )
    new_df["accsib_home_not_NA_21"] = (
        df_zp["accsib_mul_home"] * (df_zp["accsib_mul_home"] >= 0) / 100000.0
    ) * (df_zp["year"] == 2021)

    new_df["work_percentage_15"] = df_zp["work_percentage"] * (df_zp["year"] == 2015)
    new_df["business_sector_agriculture_15"] = df_zp["business_sector_agriculture"] * (df_zp["year"] == 2015)

    new_df["work_percentage_20"] = df_zp["work_percentage"] * (df_zp["year"] == 2020)

    new_df["business_sector_agriculture_21"] = df_zp["business_sector_agriculture"] * (df_zp["year"] == 2021)
    new_df["business_sector_production_21"] = df_zp["business_sector_production"] * (df_zp["year"] == 2021)
    new_df["business_sector_wholesale_21"] = df_zp["business_sector_wholesale"] * (df_zp["year"] == 2021)
    new_df["work_percentage_21"] = df_zp["work_percentage"] * (df_zp["year"] == 2021)
    new_df["work_percentage_20_21"] = df_zp["work_percentage"] * ((df_zp["year"] == 2021) + (df_zp["year"] == 2020))
    new_df["age_21"] = df_zp["age"] * (df_zp["year"] == 2021)

    new_df["age_1520"] = df_zp["age"] * ((df_zp["year"] == 2015) + (df_zp["year"] == 2020))

    new_df["business_sector_production_1520"] = df_zp["business_sector_production"] * ((df_zp["year"] == 2015) + (df_zp["year"] == 2020))
    new_df["business_sector_wholesale_1520"] = df_zp["business_sector_wholesale"] * ((df_zp["year"] == 2015) + (df_zp["year"] == 2020))

    other_columns = [
        "no_post_school_educ",
        "secondary_education",
        "tertiary_education",
        choice_var_name,
    ]
    new_df.loc[:, other_columns] = df_zp.loc[:, other_columns]

    #remove columns with only zeros
    #new_df = new_df.loc[:, (new_df != 0).any(axis=0)]

    return new_df


def get_rumboost_rum_structure(new_df):
    
    variables = [c for c in new_df.columns if c not in ["telecommuting_intensity", "telecommuting", "HHNR"]]

    monotone_constraints = [0] * len(variables)
    interaction_constraints = [[i] for i, _ in enumerate(variables)]

    #rum structure
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

    return rum_structure

def get_rumboost_model_spec(new_df, intensity_cutoff = None) -> dict:

    rum_structure = get_rumboost_rum_structure(new_df)

    general_params = {
        "n_jobs": -1,
        "num_classes": 100 // intensity_cutoff + 1 if intensity_cutoff else 2,  # important
        "verbosity": 0,  # specific RUMBoost parameter
        "num_iterations": 2000,
        "early_stopping_round": 100,
    }

    model_specification = {
        "general_params": general_params,
        "rum_structure": rum_structure,
    }

    if intensity_cutoff:
        model_specification["ordinal_logit"] = {
            "model": "proportional_odds",
            "optim_interval": 1,
        }

    return model_specification

def lightgbm_data(features, choice):

    lgb_dataset = lgb.Dataset(features, label=choice, free_raw_data=False)

    return lgb_dataset