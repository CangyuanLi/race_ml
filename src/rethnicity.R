library(arrow)
library(dplyr)
library(rethnicity)

FINAL_PATH <- file.path(getwd(), "src/data/final")
RETH_PATH <- file.path(FINAL_PATH, "rethnicity")

make_preds <- function(inpath, outpath) {
    preds <- arrow::read_parquet(
        file.path(FINAL_PATH, inpath),
        col_select = c("first_name", "last_name")
    )
    preds <- preds %>% dplyr::distinct(first_name, last_name, .keep_all = TRUE)

    reth_preds <- rethnicity::predict_ethnicity(
        firstnames = preds$first_name,
        lastnames = preds$last_name,
        threads = 2
    )

    reth_preds <- reth_preds %>%
        dplyr::rename(
            first_name = firstname,
            last_name = lastname,
            reth_asian = prob_asian,
            reth_black = prob_black,
            reth_hispanic = prob_hispanic,
            reth_white = prob_white,
            reth_race = race
        )

    arrow::write_parquet(reth_preds, file.path(RETH_PATH, outpath))
}

# make_preds("flz_test_sample.parquet", "flz_reth_preds.parquet")
make_preds("ppp_test_sample.parquet", "ppp_reth_preds.parquet")
# make_preds("lendio_ppp_sample.parquet", "lendio_ppp_reth_preds.parquet")
# make_preds("fl_test_sample.parquet", "fl_reth_preds.parquet")
