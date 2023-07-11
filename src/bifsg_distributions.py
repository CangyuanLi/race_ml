from __future__ import annotations

import string
from collections.abc import Iterable

import polars as pl

from utils.paths import DIST_PATH, L2_PATH
from utils.utils import make_representative

UNWANTED_CHARACTERS = string.digits + string.punctuation + string.whitespace


def remove_chars(expr: pl.Expr, chars: Iterable[str]) -> pl.Expr:
    for char in chars:
        expr = expr.str.replace_all(char, "", literal=True)

    return expr


def initial_cleaning(overwrite: bool = False) -> pl.LazyFrame:
    filepath = L2_PATH / "l2_for_distributions.parquet"
    if not overwrite and filepath.is_file():
        return pl.scan_parquet(filepath)

    df = (
        pl.scan_parquet(L2_PATH / "l2_clean.parquet")
        .select("first_name", "last_name", "race_ethnicity", "is_self_reported")
        .filter(~(pl.col("race_ethnicity").is_in(["aian", "other"])))
        .collect()
    )

    # replicates https://github.com/theonaunheim/surgeo/blob/master/surgeo/models/base_model.py
    df = df.with_columns(
        pl.col("first_name", "last_name")
        .pipe(remove_chars, chars=UNWANTED_CHARACTERS)
        .str.to_uppercase()
        .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
    )
    df = make_representative(df)
    df.write_parquet(filepath)

    return df.lazy()


def create_prob_files(df: pl.LazyFrame, name_col: str) -> pl.DataFrame:
    """We do not consider names whose proportions are based on fewer than 30
    observations. The only exception is when the proportion is unity for a single
    category and zero for all other five categories, and it is based on 15-29
    observations.

    Args:
        df (pl.LazyFrame): _description_
        name_col (str): _description_

    Returns:
        pl.DataFrame: _description_
    """
    base = (
        df.select(name_col, "race_ethnicity")
        .filter(pl.col(name_col).is_not_null())
        .filter(pl.col(name_col) != "")
        .filter(pl.col(name_col).str.lengths() > 1)
        .groupby(name_col, "race_ethnicity")
        .agg(pl.count().alias("count"))
        .with_columns(pl.col("count").sum().over(name_col).alias("total_count"))
        .filter(
            (pl.col("total_count") >= 25)
            | (
                (pl.col("count").max().over(name_col) == pl.col("total_count"))
                & (pl.col("total_count") >= 15)
            )
        )
        .with_columns((pl.col("count") / pl.col("total_count")).alias("pct"))
        .collect()
    )

    prob_race_given_name = (
        base.pivot(
            index=name_col,
            columns="race_ethnicity",
            values=["count", "pct"],
            aggregate_function=None,
            sort_columns=True,
        )
        .select(
            pl.col("*").map_alias(
                lambda colname: colname.replace("_race_ethnicity", "")
            )
        )
        .fill_null(0)
        .sort(name_col)
    )
    prob_race_given_name.write_parquet(L2_PATH / f"prob_race_given_{name_col}.parquet")

    prob_name_given_race = (
        base.with_columns(
            pl.col("count").sum().over("race_ethnicity").alias("total_count")
        )
        .with_columns((pl.col("count") / pl.col("total_count")).alias("pct"))
        .pivot(
            index=name_col,
            columns="race_ethnicity",
            values=["count", "pct"],
            aggregate_function=None,
            sort_columns=True,
        )
        .select(
            pl.col("*").map_alias(
                lambda colname: colname.replace("_race_ethnicity", "")
            )
        )
        .fill_null(0)
        .sort(name_col)
    )
    prob_name_given_race.write_parquet(L2_PATH / f"prob_{name_col}_given_race.parquet")


def adjust_surgeo_files():
    pxr_files = [
        "prob_first_name_given_race_harvard.csv",
        "prob_tract_given_race_2010.csv",
        "prob_zcta_given_race_2010.csv",
    ]
    prx_files = [
        "prob_race_given_first_name_harvard.csv",
        "prob_race_given_surname_2010.csv",
        "prob_race_given_tract_2010.csv",
        "prob_race_given_zcta_2010.csv",
    ]

    for f in pxr_files:
        (
            pl.read_csv(DIST_PATH / f"original/{f}")
            .rename({"api": "asian"})
            .drop("native", "multiple")
        ).write_csv(DIST_PATH / f"{f}")

    for f in prx_files:
        df = (
            pl.read_csv(DIST_PATH / f"original/{f}")
            .rename({"api": "asian"})
            .drop("native", "multiple")
            .with_columns(pl.sum("asian", "black", "hispanic", "white").alias("sum"))
        )

        for race in ("asian", "black", "hispanic", "white"):
            df = df.with_columns(pl.col(race) / pl.col("sum"))

        df.select(pl.exclude("sum")).write_csv(DIST_PATH / f"{f}")


def create_combined_files():
    # race given last name
    surgeo = pl.read_csv(DIST_PATH / "prob_race_given_surname_2010.csv").select(
        "name", "asian", "black", "hispanic", "white"
    )
    l2 = (
        pl.read_parquet(L2_PATH / "prob_race_given_last_name.parquet")
        .select(pl.exclude("^^count_.*$"))
        .rename({"last_name": "name"})
        .select(pl.col("*").map_alias(lambda colname: colname.replace("pct_", "")))
        .filter(~(pl.col("name").is_in(surgeo.get_column("name"))))
    )
    pl.concat([surgeo, l2], how="vertical").write_csv(
        DIST_PATH / "prob_race_given_last_name.csv"
    )

    # race given first name
    l2 = (
        pl.read_parquet(L2_PATH / "prob_race_given_first_name.parquet")
        .select(pl.exclude("^^count_.*$"))
        .rename({"first_name": "name"})
        .select(pl.col("*").map_alias(lambda colname: colname.replace("pct_", "")))
    )
    surgeo = (
        pl.read_csv(DIST_PATH / "prob_race_given_first_name_harvard.csv")
        .filter(~(pl.col("name").is_in(l2.get_column("name"))))
        .select("name", "asian", "black", "hispanic", "white")
    )
    pl.concat([l2, surgeo], how="vertical").write_csv(
        DIST_PATH / "prob_race_given_first_name.csv"
    )

    # first name given race
    l2 = (
        pl.read_parquet(L2_PATH / "prob_first_name_given_race.parquet")
        .select(pl.exclude("^^count_.*$"))
        .rename({"first_name": "name"})
        .select(pl.col("*").map_alias(lambda colname: colname.replace("pct_", "")))
    )
    surgeo = (
        pl.read_csv(DIST_PATH / "prob_first_name_given_race_harvard.csv")
        .filter(~(pl.col("name").is_in(l2.get_column("name"))))
        .select("name", "asian", "black", "hispanic", "white")
    )
    pl.concat([l2, surgeo], how="vertical").write_csv(
        DIST_PATH / "prob_first_name_given_race.csv"
    )


def create_flz_lookup_table():
    lf = initial_cleaning(overwrite=False)
    lf = (
        lf.select("first_name", "last_name", "zcta", "race_ethnicity")
        .filter(pl.col("first_name").str.lengths() > 1)
        .filter(pl.col("last_name").str.lengths() > 1)
        .filter(pl.col("zcta").is_not_null())
        .filter(pl.col("zcta") != "")
        .groupby("first_name", "last_name", "zcta", "race_ethnicity")
        .agg(pl.count().alias("count"))
    )

    print(lf.collect())
    # lf = lf.select()

    # base = (
    #     df.select(name_col, "race_ethnicity")
    #     .filter(pl.col(name_col).is_not_null())
    #     .filter(pl.col(name_col) != "")
    #     .filter(pl.col(name_col).str.lengths() > 1)
    #     .groupby(name_col, "race_ethnicity")
    #     .agg(pl.count().alias("count"))
    #     .with_columns(pl.col("count").sum().over(name_col).alias("total_count"))
    #     .filter(
    #         (pl.col("total_count") >= 25)
    #         | (
    #             (pl.col("count").max().over(name_col) == pl.col("total_count"))
    #             & (pl.col("total_count") >= 15)
    #         )
    #     )
    #     .with_columns((pl.col("count") / pl.col("total_count")).alias("pct"))
    #     .collect()
    # )

    # prob_race_given_name = (
    #     base.pivot(
    #         index=name_col,
    #         columns="race_ethnicity",
    #         values=["count", "pct"],
    #         aggregate_function=None,
    #         sort_columns=True,
    #     )
    #     .select(
    #         pl.col("*").map_alias(
    #             lambda colname: colname.replace("_race_ethnicity", "")
    #         )
    #     )
    #     .fill_null(0)
    #     .sort(name_col)
    # )
    # prob_race_given_name.write_parquet(L2_PATH / f"prob_race_given_{name_col}.parquet")

    # prob_name_given_race = (
    #     base.with_columns(
    #         pl.col("count").sum().over("race_ethnicity").alias("total_count")
    #     )
    #     .with_columns((pl.col("count") / pl.col("total_count")).alias("pct"))
    #     .pivot(
    #         index=name_col,
    #         columns="race_ethnicity",
    #         values=["count", "pct"],
    #         aggregate_function=None,
    #         sort_columns=True,
    #     )
    #     .select(
    #         pl.col("*").map_alias(
    #             lambda colname: colname.replace("_race_ethnicity", "")
    #         )
    #     )
    #     .fill_null(0)
    #     .sort(name_col)
    # )
    # prob_name_given_race.write_parquet(L2_PATH / f"prob_{name_col}_given_race.parquet")


def create_fl_lookup_table():
    lf = initial_cleaning(overwrite=False)
    lf = (
        lf.select("first_name", "last_name", "race_ethnicity")
        .filter(pl.col("first_name").str.lengths() > 1)
        .filter(pl.col("last_name").str.lengths() > 1)
        .with_columns(
            (pl.col("first_name") + pl.lit(" ") + pl.col("last_name")).alias("name")
        )
        .select("name", "race_ethnicity")
    )

    create_prob_files(lf, name_col="name")

    pl.read_parquet(L2_PATH / "prob_race_given_name.parquet").write_csv(
        DIST_PATH / "prob_race_given_name.csv"
    )

    pl.read_parquet(L2_PATH / "prob_name_given_race.parquet").write_csv(
        DIST_PATH / "prob_name_given_race.csv"
    )


def main():
    df = initial_cleaning(overwrite=False)

    create_prob_files(df, name_col="first_name")
    create_prob_files(df, name_col="last_name")

    adjust_surgeo_files()

    create_combined_files()

    # create_flz_lookup_table()
    create_fl_lookup_table()


if __name__ == "__main__":
    main()
