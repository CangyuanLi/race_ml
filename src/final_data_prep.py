import functools
import json
import math
from pathlib import Path

import dask.dataframe
import polars as pl

from utils.constants import VALID_NAME_CHARS
from utils.paths import CENSUS_PATH, CW_PATH, FINAL_PATH, L2_PATH


@functools.lru_cache()
def has_invalid_name(name: str) -> bool:
    return any(c not in VALID_NAME_CHARS for c in name)


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
                | (pl.col("race_ethnicity") == "null")
            )
        )
        .with_columns(
            pl.col("first_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
            pl.col("last_name").str.replace_all(f"[^{VALID_NAME_CHARS}]", ""),
        )
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
        .collect()
    )
    final = final.with_columns(pl.Series(name="index", values=range(final.shape[0])))
    final.write_parquet(outpath)

    return final.lazy()


def train_test_split(
    df: pl.DataFrame,
    sample: str,
    train_pct: float = 0.7,
    undersample: bool = True,
    overwrite: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
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


def main():
    all_df = create_baseline_file(FINAL_PATH / "all.parquet", overwrite=False)

    # Clean the base FLZ dataset
    # (
    #     all_df.select(
    #         "index",
    #         "first_name",
    #         "last_name",
    #         "race_ethnicity",
    #         "pct_asian_zcta",
    #         "pct_black_zcta",
    #         "pct_hispanic_zcta",
    #         "pct_white_zcta",
    #         "zcta",
    #     )
    #     .pipe(
    #         iterative_drop_nulls,
    #         subset=[
    #             "first_name",
    #             "last_name",
    #             "zcta",
    #             "pct_asian_zcta",
    #             "pct_black_zcta",
    #             "pct_hispanic_zcta",
    #             "pct_white_zcta",
    #         ],
    #     )
    #     .pipe(
    #         iterative_drop_nans,
    #         subset=[
    #             "pct_asian_zcta",
    #             "pct_black_zcta",
    #             "pct_hispanic_zcta",
    #             "pct_white_zcta",
    #         ],
    #     )
    #     .filter(~pl.col("first_name").is_in(("", " ")))
    #     .filter(~pl.col("last_name").is_in(("", " ")))
    #     .sink_parquet(FINAL_PATH / "has_name_zcta.parquet")
    # )

    has_name_zcta = pl.scan_parquet(FINAL_PATH / "has_name_zcta.parquet")

    # Version where duplicates are dropped (one input has one correct answer)

    has_name_zcta_dup_dropped = pl.from_pandas(
        has_name_zcta.collect()
        .to_pandas()
        .drop_duplicates(subset=["first_name", "last_name", "zcta"])
    )  # weirdness is due to polars .unique() using too much memory
    train_test_split(has_name_zcta_dup_dropped, sample="flz", train_pct=0.8)
    del has_name_zcta_dup_dropped
    print("done with has_name_zcta_dup_dropped")

    # Version where duplicates are kept (one input can have multiple correct answers)

    has_name_zcta_dups = pl.from_pandas(
        has_name_zcta.collect()
        .to_pandas()
        .drop_duplicates(subset=["first_name", "last_name", "zcta", "race_ethnicity"])
    )
    train_test_split(has_name_zcta_dups, sample="flz_dups", train_pct=0.8)
    print("done with has_name_zcta_dups")

    # Imbalanced, keep duplicates

    train_test_split(
        has_name_zcta_dups, sample="flz_imb_dups", train_pct=0.8, undersample=False
    )
    print("done with flz imbalanced dups")

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
