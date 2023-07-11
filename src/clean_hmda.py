import pandas as pd
import polars as pl

from utils.paths import HMDA_PATH


def main():
    df = (
        pl.scan_csv(HMDA_PATH / "hmda_raw.csv", null_values="NA")
        .select(
            "state_code",
            "county_code",
            "census_tract",
            "derived_race",
            "derived_ethnicity",
        )
        .with_columns(
            pl.when(pl.col("derived_ethnicity") == "Hispanic or Latino")
            .then(pl.col("derived_ethnicity"))
            .otherwise(pl.col("derived_race"))
            .alias("race_ethnicity")
        )
        .filter(
            ~pl.col("race_ethnicity").is_in(
                ["Race Not Available", "Free Form Text Only"]
            )
        )
        .with_columns(
            pl.col("race_ethnicity")
            .str.to_lowercase()
            .map_dict(
                {
                    "black or african american": "black",
                    "hispanic or latino": "hispanic",
                    "american indian or alaska native": "aian",
                    "native hawaiian or other pacific islander": "pi",
                    "joint": "other",
                    "2 or more minority races": "other",
                },
                default=pl.first(),
            )
        )
        .select(pl.exclude("derived_race", "derived_ethnicity"))
        .collect()
    )

    print(df)
    df.write_parquet(HMDA_PATH / "hmda_clean.parquet")


if __name__ == "__main__":
    main()
