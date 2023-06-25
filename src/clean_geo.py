import polars as pl

from utils.paths import DAT_PATH

# df = (
#     pl.scan_csv(DAT_PATH / "crosswalks/state_abbrev_to_fips.csv")
#     .filter(~pl.col("state_fips").is_null())
#     .with_columns(
#         pl.col("state_fips").cast(str).str.zfill(2),
#         pl.col("state_abbrev").str.to_lowercase(),
#     )
#     .collect()
# )

# df.write_parquet(DAT_PATH / "crosswalks/state_abbrev_to_fips.parquet")

zip_zcta_cw = (
    pl.scan_csv(DAT_PATH / "crosswalks/zip_zcta_cw.csv")
    .with_columns(pl.col("zip", "zcta").cast(str).str.zfill(5))
    .sort("year", descending=True)
    .unique(["zip", "zcta"], keep="first")
    .collect()
    .write_parquet(DAT_PATH / "crosswalks/zip_zcta_cw_final.parquet")
)
