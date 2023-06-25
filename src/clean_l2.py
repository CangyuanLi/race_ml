from pathlib import Path

import polars as pl
import tqdm

from utils.paths import CW_PATH, L2_PATH
from utils.utils import coerce_to_ascii

# https://www.census.gov/topics/population/race/about.html

ASIAN_DESCRIPTIONS = {
    "bangladeshi",
    "bhutanese",
    "chinese",
    "filipino",
    "indian/hindu",
    "japanese",
    "korean",
    "laotian",
    "malay",
    "maldivian",
    "mongolian",
    "myanmar (burmese)",
    "nepalese",
    "thai",
    "tibetan",
    "unknown asian",
    "vietnamese",
}


def clean_data(file: Path):
    df = (
        pl.scan_parquet(file)
        .select(pl.all().map_alias(lambda col_name: col_name.lower()))
        .rename(
            {
                "voters_firstname": "first_name",
                "voters_middlename": "middle_name",
                "voters_lastname": "last_name",
                "voters_namesuffix": "name_suffix",
                "residence_addresses_state": "state_abbrev",
                "residence_addresses_zip": "zip",
                "residence_addresses_zipplus4": "zip_last4",
                "residence_addresses_censustract": "census_tract",
                "residence_addresses_censusblockgroup": "census_block_group",
                "residence_addresses_censusblock": "census_block",
                "ethnicgroups_ethnicgroup1desc": "ethnic_group",
                "countyethnic_lalethniccode": "ethnic_code",
                "countyethnic_description": "race_ethnicity",
                "county": "county_name",
                "voters_fips": "county_fips",
            }
        )
        .with_columns(pl.all().cast(str).str.to_lowercase())
        # remove rows where there is NO ethnicity information
        .filter(
            ~pl.all(
                pl.col(col).is_null()
                for col in [
                    "ethnic_group",
                    "ethnic_code",
                    "race_ethnicity",
                    "ethnic_description",
                ]
            )
        )
        .collect()
    )

    # Get state from filepath
    state = file.name[5:7]

    df.write_parquet(L2_PATH / f"clean/{state}.parquet")


def filter_data(file: Path) -> pl.DataFrame:
    # https://www.pewresearch.org/short-reads/2022/09/15/who-is-hispanic/
    race_ethnicity_mapper = {
        "east asian": "asian",
        "african or af-am self reported": "black",
        "white self reported": "white",
        "other undefined race": "other",
        "native american (self reported)": "other",
    }
    df = (
        pl.scan_parquet(file)
        .select(pl.exclude("ethnic_code"))
        .with_columns(pl.col("race_ethnicity").map_dict(race_ethnicity_mapper))
        .with_columns(
            race_ethnicity=pl.when(pl.col("ethnic_group") == "east and south asian")
            .then("asian")
            .otherwise(pl.col("race_ethnicity")),
        )
        .with_columns(
            race_ethnicity=pl.when(
                pl.col("ethnic_description").str.contains("|".join(ASIAN_DESCRIPTIONS))
            )
            .then("asian")
            .otherwise(pl.col("race_ethnicity")),
        )
        .with_columns(
            race_ethnicity=pl.when(pl.col("ethnic_description") == "hispanic")
            .then("hispanic")
            .otherwise(pl.col("race_ethnicity")),
        )
        .with_columns(
            first_name=pl.col("first_name").apply(coerce_to_ascii),
            last_name=pl.col("last_name").apply(coerce_to_ascii),
        )
        .filter(
            ~(
                (pl.col("race_ethnicity") == "other")
                | (pl.col("race_ethnicity").is_null())
            )
        )
        .select(
            "first_name",
            "last_name",
            "race_ethnicity",
            "state_abbrev",
            "zip",
            "census_tract",
            "county_fips",
        )
        .collect()
    )

    return df


def main():
    # Read files
    files = list((L2_PATH / "intermediate").glob("*.parquet"))
    for file in tqdm.tqdm(files):
        clean_data(file)

    # Subset to rows that have race
    files = list((L2_PATH / "clean").glob("*.parquet"))
    dfs = []
    for file in tqdm.tqdm(files):
        dfs.append(filter_data(file))
    df: pl.DataFrame = pl.concat(dfs)

    # Final cleaning

    sa_fips_cw = pl.read_parquet(CW_PATH / "state_abbrev_to_fips.parquet")
    df = df.join(sa_fips_cw, on="state_abbrev", how="left").with_columns(
        county_fips=pl.col("state_fips") + pl.col("county_fips"),
    )
    df.write_parquet(L2_PATH / "l2_clean.parquet")


if __name__ == "__main__":
    main()
