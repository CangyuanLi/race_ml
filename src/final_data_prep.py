from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path

import polars as pl

from utils.constants import VALID_NAME_CHARS
from utils.paths import CENSUS_PATH, CW_PATH, DIST_PATH, FINAL_PATH, L2_PATH
from utils.utils import make_representative

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()
LOGGER.addHandler(logging.FileHandler("final_data_prep.log", mode="w"))


@functools.lru_cache()
def has_invalid_name(name: str) -> bool:
    return any(c not in VALID_NAME_CHARS for c in name)


@functools.lru_cache()
def remove_single_chars(name: str) -> str:
    return " ".join(part for part in name.split(" ") if len(part) > 1)


def create_baseline_file(outpath: Path, overwrite: bool = False) -> pl.LazyFrame:
    if not overwrite:
        return pl.scan_parquet(outpath)

    l2 = pl.scan_parquet(L2_PATH / "l2_clean.parquet")
    zip_zcta = pl.scan_parquet(CW_PATH / "zip_zcta_cw_final.parquet")
    race_pct = pl.scan_parquet(CENSUS_PATH / "race_by_zcta.parquet")
    final = (
        l2.filter(
            ~(
                (pl.col("race_ethnicity").is_null())
                | (pl.col("race_ethnicity").is_in(["null", "other", "aian"]))
            )
        )
        .with_columns(
            pl.col("first_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
            pl.col("last_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
        )
        .filter(~(pl.col("first_name") == "zz"))
        .with_columns(pl.col("first_name").apply(remove_single_chars))
        .with_columns(
            first_name_length=pl.col("first_name").str.lengths(),
            last_name_length=pl.col("last_name").str.lengths(),
        )
        .filter(
            ~((pl.col("first_name_length") == 1) | (pl.col("last_name_length") == 1))
        )
        .select(pl.exclude("first_name_length", "last_name_length"))
        # don't do any filtering on if the name is "too long". By visual inspection,
        # these seem to still be valid names
        .with_columns(pl.col("zip").str.replace(".0", "", literal=True))
        .join(zip_zcta, on="zip", how="left")
        .join(race_pct, on="zcta", how="left")
        # There are some names that are self-reported as a certain race but would be
        # very confusing to an algorithm. For example, Amy Kearney self-reports as
        # Asian, but there would be really no way to tell that she is Asian from just
        # name and zip code. So for the purposes for this model, I need to remove such
        # examples. Since it is unrealistic to do it manually, the rule I use is if
        # P(Race | Last Name) < threshold and P(Race | Last Name) < threshold, drop that
        # observation
        .join(
            pl.scan_csv(DIST_PATH / "prob_race_given_last_name.csv")
            .select(
                pl.col("*").map_alias(
                    lambda colname: f"prl_{colname}" if colname != "name" else colname
                )
            )
            .with_columns(pl.col("name").str.to_lowercase()),
            left_on="last_name",
            right_on="name",
            how="left",
        )
        .join(
            pl.scan_csv(DIST_PATH / "prob_race_given_first_name.csv")
            .select(
                pl.col("*").map_alias(
                    lambda colname: f"prf_{colname}" if colname != "name" else colname
                )
            )
            .with_columns(pl.col("name").str.to_lowercase()),
            left_on="first_name",
            right_on="name",
            how="left",
        )
        .with_columns(
            pl.when(pl.col("race_ethnicity") == "asian")
            .then(pl.col("prl_asian"))
            .when(pl.col("race_ethnicity") == "black")
            .then(pl.col("prl_black"))
            .when(pl.col("race_ethnicity") == "hispanic")
            .then(pl.col("prl_hispanic"))
            .when(pl.col("race_ethnicity") == "white")
            .then(pl.col("prl_white"))
            .otherwise(None)
            .alias("prl_race")
        )
        .with_columns(
            pl.when(pl.col("race_ethnicity") == "asian")
            .then(pl.col("prf_asian"))
            .when(pl.col("race_ethnicity") == "black")
            .then(pl.col("prf_black"))
            .when(pl.col("race_ethnicity") == "hispanic")
            .then(pl.col("prf_hispanic"))
            .when(pl.col("race_ethnicity") == "white")
            .then(pl.col("prf_white"))
            .otherwise(None)
            .alias("prf_race")
        )
        .with_columns(
            ((pl.col("prf_race") < 0.2) & (pl.col("prl_race") < 0.01)).alias(
                "is_misleading_name"
            )
        )
        .select(
            pl.exclude(
                [f"prf_{race}" for race in ("asian", "black", "hispanic", "white")]
            )
        )
        .select(
            pl.exclude(
                [f"prl_{race}" for race in ("asian", "black", "hispanic", "white")]
            )
        )
        .with_row_count(name="index")
    )

    final.collect().write_parquet(outpath)

    return final


def train_test_split(
    df: pl.DataFrame,
    sample: str,
    train_pct: float = 0.8,
    undersample: bool = True,
    overwrite: bool = False,
):
    # https://reader.elsevier.com/reader/sd/pii/S2352711021001874
    # https://datascience.stackexchange.com/questions/61858/oversampling-undersampling-only-train-set-only-or-both-train-and-validation-set
    """
    Undersample the training dataset based on the smallest group, as per Xie 2022.
    This should improve performance. However, we don't want to undersample the test
    dataset because our data will be imbalanced in practice.
    """
    train_path = FINAL_PATH / f"{sample}_train.parquet"
    test_path = FINAL_PATH / f"{sample}_test.parquet"

    if not overwrite and train_path.is_file() and test_path.is_file():
        return

    if not undersample:
        df = df.with_row_count(name="index")
        train_sample = df.sample(fraction=train_pct, seed=1)
        train_sample.write_parquet(train_path)
        df.join(train_sample, on="index", how="anti").write_parquet(test_path)

        return

    # Identify smallest group
    min_count = df.get_column("race_ethnicity").value_counts().min()
    train_sample_size = round(min_count[0, 1] * train_pct)

    df = df.with_columns(
        mask=pl.arange(0, pl.count()).shuffle(seed=1).over("race_ethnicity")
    )
    df.filter(pl.col("mask") < train_sample_size).drop("mask").write_parquet(train_path)
    df.filter(pl.col("mask") >= train_sample_size).drop("mask").write_parquet(test_path)


def normalize_pcts(
    df: pl.DataFrame, cols: list[str], outname: str, limit: float = 0.001
) -> pl.DataFrame:
    """Winsorize percentages and then scale them so that they sum to 1.

    Args:
        df (pl.DataFrame): data
        cols (list[str]): cols to winsorize and scale
        outname (str): file that holds each columns lower / upper
        limit (float, optional): level to winsorize at. Defaults to 0.001.

    Returns:
        pl.DataFrame: df with processed cols
    """
    # https://stats.stackexchange.com/questions/350487/is-winsorization-performed-on-test-data-as-well
    # we need to save the lower and upper since we want to apply the same winsorization
    # step to the test data

    out = {col: None for col in cols}
    for col in cols:
        df = df.with_columns(
            lower=pl.col(col).quantile(limit), upper=pl.col(col).quantile(1 - limit)
        )
        lower = df.get_column("lower").max()
        upper = df.get_column("upper").max()
        out[col] = [lower, upper]
        df = df.with_columns(pl.col(col).clip(lower, upper))
        df = df.with_columns(pl.col(col).clip(1e-10, 1 - 1e-10))

    with open(FINAL_PATH / outname, "w") as f:
        json.dump(out, f, sort_keys=True, indent=4)

    df = df.with_columns(sum=pl.sum(col for col in cols))

    for col in cols:
        df = df.with_columns(pl.col(col) / pl.col("sum"))

    df = df.select(pl.exclude("sum"))

    return df


def iterative_drop_nulls(expr: pl.Expr, subset: list[str]) -> pl.LazyFrame:
    for col in subset:
        expr = expr.filter(~pl.col(col).is_null())

    return expr


def iterative_drop_nans(expr: pl.Expr, subset: list[str]) -> pl.LazyFrame:
    for col in subset:
        expr = expr.filter(~pl.col(col).is_nan())

    return expr


def unique(df: pl.DataFrame, subset: list[str]) -> pl.DataFrame:
    return pl.from_pandas(df.to_pandas().drop_duplicates(subset=subset))
    # valid_indices = set(
    #     df.select(["index"] + subset)
    #     .unique(subset=subset)
    #     .get_column("index")
    #     .to_list()
    # )
    # print(len(valid_indices))
    # print(df.shape[0])
    # print(df.shape[0] - len(valid_indices))

    # return df.filter(pl.col("index").is_in(valid_indices))


def create_flz_file(df: pl.LazyFrame, overwrite: bool = False) -> pl.LazyFrame:
    outpath = FINAL_PATH / "has_name_zcta.parquet"

    if not overwrite and outpath.is_file():
        return pl.scan_parquet(outpath)

    has_name_zcta = (
        df.select(
            "index",
            "first_name",
            "last_name",
            "race_ethnicity",
            "pct_asian_zcta",
            "pct_black_zcta",
            "pct_hispanic_zcta",
            "pct_white_zcta",
            "zcta",
            "is_misleading_name",
            "is_self_reported",
            "prf_race",
            "prl_race",
        )
        .filter(~pl.col("is_misleading_name"))
        .pipe(
            iterative_drop_nulls,
            subset=[
                "first_name",
                "last_name",
                "zcta",
                "pct_asian_zcta",
                "pct_black_zcta",
                "pct_hispanic_zcta",
                "pct_white_zcta",
            ],
        )
        .pipe(
            iterative_drop_nans,
            subset=[
                "pct_asian_zcta",
                "pct_black_zcta",
                "pct_hispanic_zcta",
                "pct_white_zcta",
            ],
        )
        .filter(~pl.col("first_name").is_in(("", " ")))
        .filter(~pl.col("last_name").is_in(("", " ")))
    )
    has_name_zcta.collect().write_parquet(outpath)

    return has_name_zcta


def create_fl_file(df: pl.LazyFrame, overwrite: bool = False) -> pl.LazyFrame:
    outpath = FINAL_PATH / "has_name.parquet"

    if not overwrite and outpath.is_file():
        return pl.scan_parquet(outpath)

    has_name = (
        df.select(
            "index",
            "first_name",
            "last_name",
            "race_ethnicity",
            "is_misleading_name",
            "is_self_reported",
            "prf_race",
            "prl_race",
        )
        .filter(~pl.col("is_misleading_name"))
        .pipe(
            iterative_drop_nulls,
            subset=[
                "first_name",
                "last_name",
            ],
        )
        .filter(~pl.col("first_name").is_in(("", " ")))
        .filter(~pl.col("last_name").is_in(("", " ")))
    )
    has_name.collect().write_parquet(outpath)

    return has_name


def dedupe_fl(lf: pl.LazyFrame) -> pl.DataFrame:
    lf = (
        lf.select("first_name", "last_name", "race_ethnicity")
        .with_columns(
            pl.count().over("first_name", "last_name").alias("count"),
            pl.count()
            .over("first_name", "last_name", "race_ethnicity")
            .alias("count_race"),
        )
        .with_columns(
            (pl.col("count") == pl.col("count_race")).alias("single_race"),
            (pl.col("count") > 10).alias("to_dedupe"),
        )
    )

    dups = lf.filter(pl.col("to_dedupe"))

    one_race = (
        dups.filter(pl.col("single_race"))
        .unique(["first_name", "last_name"])
        .select("first_name", "last_name", "race_ethnicity")
    )

    races = ("asian", "black", "hispanic", "white")

    def _create_dummies(lf: pl.LazyFrame) -> pl.LazyFrame:
        for race in races:
            lf = lf.with_columns((pl.col("race_ethnicity") == race).alias(f"is_{race}"))

        return lf

    def _calc_percentages(lf: pl.LazyFrame) -> pl.LazyFrame:
        for race in races:
            lf = lf.with_columns(
                pl.when(pl.col(f"pct_{race}").is_null())
                .then(pl.col(f"{race}_count") / pl.col("count"))
                .otherwise(pl.col(f"pct_{race}"))
            )

        return lf

    def _calc_allowed(lf: pl.LazyFrame) -> pl.LazyFrame:
        for race in races:
            lf = lf.with_columns(
                (pl.col(f"{race}_count") // pl.col(f"pct_{race}")).alias(
                    f"allowed_{race}"
                )
            ).with_columns(
                pl.when(pl.col(f"allowed_{race}") == 0)
                .then(None)
                .otherwise(pl.col(f"allowed_{race}"))
                .keep_name()
            )

        return lf

    mult_race = (
        dups.filter(~pl.col("single_race"))
        .with_columns((pl.col("first_name") + " " + pl.col("last_name")).alias("name"))
        .join(
            pl.scan_csv(DIST_PATH / "prob_race_given_name.csv")
            .select("name", "pct_asian", "pct_black", "pct_hispanic", "pct_white")
            .with_columns(pl.col("name").str.to_lowercase()),
            on="name",
            how="left",
        )
        .select(pl.exclude("name"))
        .pipe(_create_dummies)
        .with_columns(
            pl.sum("pct_asian", "pct_black", "pct_hispanic", "pct_white").alias("sum"),
        )
        .with_columns(
            pl.col("pct_asian", "pct_black", "pct_hispanic", "pct_white")
            / pl.col("sum"),
            pl.col("is_asian", "is_black", "is_hispanic", "is_white")
            .sum()
            .over("first_name", "last_name", "race_ethnicity")
            .map_alias(lambda colname: f"{colname.replace('is_', '')}_count"),
            pl.count().over("first_name", "last_name").alias("total_count"),
        )
        .pipe(_calc_percentages)
        .pipe(_calc_allowed)
        .with_columns(
            pl.min(
                "allowed_asian", "allowed_black", "allowed_hispanic", "allowed_white"
            ).alias("allowed")
        )
        .with_columns(pl.col("allowed").clip(0, 10))
        .drop("is_asian", "is_black", "is_hispanic", "is_white")
        .with_columns(
            pl.when(pl.col("race_ethnicity") == "asian")
            .then(pl.col("pct_asian"))
            .when(pl.col("race_ethnicity") == "black")
            .then(pl.col("pct_black"))
            .when(pl.col("race_ethnicity") == "hispanic")
            .then(pl.col("pct_hispanic"))
            .when(pl.col("race_ethnicity") == "white")
            .then(pl.col("pct_white"))
            .otherwise(None)
            .alias("pct")
        )
        .drop("pct_asian", "pct_black", "pct_hispanic", "pct_white")
        .with_columns((pl.col("allowed") * pl.col("pct")).floor())
        .with_columns(
            pl.arange(0, pl.count())
            .over("first_name", "last_name", "race_ethnicity")
            .alias("mask")
        )
        .filter(pl.col("mask") < pl.col("allowed"))
        .select("first_name", "last_name", "race_ethnicity")
    )

    no_dedupe = lf.filter(~pl.col("to_dedupe"))
    pt1 = (
        no_dedupe.filter(pl.col("single_race"))
        .unique(["first_name", "last_name"])
        .select("first_name", "last_name", "race_ethnicity")
    )
    pt2 = no_dedupe.filter(~pl.col("single_race")).select(
        "first_name", "last_name", "race_ethnicity"
    )

    logging.info("final concat")
    return (
        pl.concat([pt1, pt2, mult_race, one_race], how="vertical")
        .select("first_name", "last_name", "race_ethnicity")
        .collect()
    )


def main():
    all_df = create_baseline_file(FINAL_PATH / "all.parquet", overwrite=True)

    # FLZ

    has_name_zcta = create_flz_file(all_df, overwrite=True)
    has_name_zcta_dups = has_name_zcta.select(
        pl.exclude(
            "is_misleading_name",
            "is_self_reported",
            "prf_race",
            "prl_race",
        )
    ).collect()
    train_test_split(
        has_name_zcta_dups, sample="flz_dups", train_pct=0.8, overwrite=True
    )

    # FL

    has_name = create_fl_file(all_df, overwrite=True)
    has_name_deduped = dedupe_fl(has_name)
    has_name_deduped.write_parquet(FINAL_PATH / "fl_imb_deduped_train.parquet")

    # print(
    #     pl.read_parquet(FINAL_PATH / "fl_imb_deduped_train.parquet")
    #     .sort("first_name", "last_name")
    #     .filter(pl.col("first_name").str.starts_with("a"))
    #     .write_csv("temp.csv")
    # )

    stop

    train_test_split(
        has_name.unique(["first_name", "last_name", "race_ethnicity"]).collect(),
        sample="fl_dups",
        train_pct=0.8,
        overwrite=True,
    )

    # For ensemble

    # races = ("asian", "black", "hispanic", "white")
    # for race in races:
    #     distribution = {}
    #     for r in races:
    #         if r == race:
    #             distribution[r] = 0.5
    #         else:
    #             distribution[r] = 0.166

    #     make_representative(has_name_zcta_dups, distribution=distribution).with_columns(
    #         (pl.col("race_ethnicity") == race).cast(pl.Int16).alias("is_race")
    #     ).write_parquet(FINAL_PATH / f"flz_{race}_train.parquet")

    # # Version where duplicates are kept (one input can have multiple correct answers)

    # has_name_zcta_dups = pl.from_pandas(
    #     has_name_zcta.collect()
    #     .to_pandas()
    #     .drop_duplicates(subset=["first_name", "last_name", "zcta", "race_ethnicity"])
    # )
    # train_test_split(has_name_zcta_dups, sample="flz_dups", train_pct=0.8)
    # print("done with has_name_zcta_dups")

    # # Imbalanced, keep duplicates

    # train_test_split(
    #     has_name_zcta_dups, sample="flz_imb_dups", train_pct=0.8, undersample=False
    # )
    # print("done with flz imbalanced dups")

    # Separate for each race

    # has_name = (
    #     all_df.drop_nulls(["first_name", "last_name"])
    #     .filter(~pl.col("first_name").is_in(("", " ")))
    #     .filter(~pl.col("last_name").is_in(("", " ")))
    #     .unique(subset=["first_name", "last_name"])
    #     .select("index", "first_name", "last_name", "race_ethnicity")
    #     .collect()
    # )
    # train, test = train_test_split(has_name)
    # train.write_parquet(FINAL_PATH / "fl_train.parquet")
    # test.write_parquet(FINAL_PATH / "fl_test.parquet")


if __name__ == "__main__":
    main()

    # pl.scan_parquet(FINAL_PATH / "fl_train.py")
