import polars as pl

from utils.paths import FINAL_PATH

all_df = pl.read_parquet(FINAL_PATH / "all.parquet")

with pl.Config(tbl_rows=100):
    print(all_df.get_column("state_abbrev").value_counts())
    print(all_df.get_column("race_ethnicity").value_counts())
