"""Utils package dealing with data of the Mobility and Transport Microcensus (MTMC)."""
from pathlib import Path
import os

import pandas as pd

def add_urban_typology(
    df: pd.DataFrame, year: int, field_bfs: str = "zone_id_home"
) -> pd.DataFrame:
    """Add an urban typology from the Federal Statistical Office (FSO) to the dataframe.
    This typology is called "Stadt/Land-Typologie" in German.
    More info: https://www.bfs.admin.ch/asset/de/2544676
    The typology defines three levels (urban, rural and "intermediate").
    Can be used e.g. with df_zp (with variable "zone_id_home" or "zone_id_work") or df_hh (with variable "zone_id_home" for 2015)."""
    if (year != 2015) & (year != 2020) & (year != 2021):
        raise ValueError("Spatial typology is only available for 2015, 2020 and 2021!")
    # CHANGED
    # path_to_typology = Path(r"\\wsbbrz0283\mobi\10_Daten\Raumgliederungen")
    # path_to_typology = path_to_typology / str(year) / "Raumgliederungen.xlsx"
    # urban_rural_typology = pd.read_excel(
    #     path_to_typology,
    #     sheet_name="Daten",
    #     skiprows=[
    #         0,
    #         2,
    #     ],  # Removes the 1st row, with information, and the 3rd, with links
    #     usecols="A,G",  # Selects only the BFS commune number and the column with the typology
    # )

    path_to_typology = Path(os.path.join(os.getcwd(), "..", "skims_zone_to_zone_2017", "mobi-zones-all-info.csv"))
    urban_rural_typology = pd.read_csv(
        path_to_typology,
        sep=";",
    )
    urban_rural_typology = urban_rural_typology.rename(columns={"ID_SL3": "Städtische / Ländliche Gebiete"})
    urban_rural_typology = urban_rural_typology[["ID","Städtische / Ländliche Gebiete"]]
    urban_rural_typology = urban_rural_typology.rename(
        columns={"Städtische / Ländliche Gebiete": "urban_typology"}
    )
    df = pd.merge(
        df,
        urban_rural_typology,
        how="left",
        left_on=field_bfs,
        right_on="ID",
    )
    df.drop("ID", axis=1, inplace=True)
    return df
