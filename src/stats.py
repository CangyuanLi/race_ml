import polars as pl

from utils.paths import FIG_PATH, FINAL_PATH, TBL_PATH

all_df = (
    pl.scan_parquet(FINAL_PATH / "all.parquet")
    .select("race_ethnicity", "is_self_reported")
    .collect()
)

# with pl.Config(tbl_rows=100):
#     print(all_df.get_column("state_abbrev").value_counts())
#     print(all_df.get_column("zcta").value_counts())


race = all_df.get_column("race_ethnicity").value_counts().rename({"counts": "Total"})
race_self_report = (
    all_df.filter(pl.col("is_self_reported"))
    .get_column("race_ethnicity")
    .value_counts()
    .rename({"counts": "Self-Reported"})
)
race_inferred = (
    all_df.filter(~pl.col("is_self_reported"))
    .get_column("race_ethnicity")
    .value_counts()
    .rename({"counts": "From Ethnicity"})
)
race_counts = (
    race.join(race_self_report, on="race_ethnicity", how="left")
    .join(race_inferred, on="race_ethnicity", how="left")
    .sort("race_ethnicity")
    .rename({"race_ethnicity": "Race"})
    .fill_null(0)
    .fill_nan(0)
    .with_columns(
        pl.col("Total", "Self-Reported", "From Ethnicity").apply(lambda x: f"{x:,}"),
        pl.col("Race").str.to_titlecase(),
    )
)
race_counts.to_pandas().to_latex(TBL_PATH / "race_counts.tex", index=False)
