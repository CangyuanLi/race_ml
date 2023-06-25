library(arrow)
library(dplyr)
library(stringr)
library(tidycensus)
library(tidyr)

CENSUS_PATH <- file.path("/Users/cangyuanli/Documents/Projects/race_ml/src/data/census")

vars <- tidycensus::load_variables(year = 2010)
vars <- vars %>% dplyr::filter(stringr::str_detect(name, "P005"))

get_race_by_zcta <- function(year)
{
    race_by_zcta <- tidycensus::get_acs(geography = "zcta", table = "B03002", year = year, cache_table = TRUE) %>%
        tidyr::pivot_wider(id_cols = NAME, names_from = variable, values_from = estimate) %>%
        dplyr::rename(
            zcta = NAME,
            total = B03002_001,
            non_hispanic = B03002_002,
            non_hispanic_white = B03002_003,
            non_hispanic_black = B03002_004,
            non_hispanic_asian = B03002_006,
            hispanic = B03002_012,
        ) %>%
        dplyr::mutate(
            zcta = zcta %>%
                stringr::str_remove("ZCTA5") %>%
                stringr::str_trim(),
            pct_black_zcta = non_hispanic_black / total,
            pct_white_zcta = non_hispanic_white / total,
            pct_asian_zcta = non_hispanic_asian / total,
            pct_hispanic_zcta = hispanic / total,
            year = year
        ) %>%
        dplyr::select(
            zcta, year, pct_black_zcta, pct_white_zcta, pct_asian_zcta, pct_hispanic_zcta
        )

    return(race_by_zcta)
}

get_race_by_zcta_decennial <- function()
{
    dat <- tidycensus::get_decennial(geography = "zcta", year = 2010, table = "P005") %>%
        tidyr::pivot_wider(id_cols = NAME, names_from = variable, values_from = value) %>%
        dplyr::rename(
            zcta = NAME,
            total = P005001,
            non_hispanic_white = P005003,
            non_hispanic_black = P005004,
            non_hispanic_asian = P005006,
            hispanic = P005010,
        ) %>%
        dplyr::mutate(
            zcta = zcta %>%
                stringr::str_remove("ZCTA5") %>%
                stringr::str_trim(),
            pct_black_zcta = non_hispanic_black / total,
            pct_white_zcta = non_hispanic_white / total,
            pct_asian_zcta = non_hispanic_asian / total,
            pct_hispanic_zcta = hispanic / total,
            year = 2010
        ) %>%
        dplyr::select(
            zcta, year, pct_black_zcta, pct_white_zcta, pct_asian_zcta, pct_hispanic_zcta
        )


    return(dat)
}

zcta2010 <- get_race_by_zcta_decennial()
zcta2011 <- get_race_by_zcta(2011)
zcta2016 <- get_race_by_zcta(2016)
zcta2021 <- get_race_by_zcta(2021)

# If a ZCTA is not found in 2021, get the 2016 data, and so on
# Furthermore, if the percentage for a given race is 0, use the average
# of all years
race_by_zcta <- dplyr::bind_rows(zcta2021, zcta2016, zcta2011, zcta2010) %>%
    dplyr::arrange(dplyr::desc(year)) %>%
    dplyr::group_by(zcta) %>%
    dplyr::mutate(
        avg_pct_black_zcta = mean(pct_black_zcta, na.rm = TRUE),
        avg_pct_white_zcta = mean(pct_white_zcta, na.rm = TRUE),
        avg_pct_asian_zcta = mean(pct_asian_zcta, na.rm = TRUE),
        avg_pct_hispanic_zcta = mean(pct_hispanic_zcta, na.rm = TRUE),
    ) %>%
    dplyr::mutate(
        pct_black_zcta = dplyr::case_when(
            pct_black_zcta == 0 ~ avg_pct_black_zcta,
            .default = pct_black_zcta
        ),
        pct_asian_zcta = dplyr::case_when(
            pct_asian_zcta == 0 ~ avg_pct_asian_zcta,
            .default = pct_asian_zcta
        ),
        pct_white_zcta = dplyr::case_when(
            pct_white_zcta == 0 ~ avg_pct_white_zcta,
            .default = pct_white_zcta
        ),
        pct_hispanic_zcta = dplyr::case_when(
            pct_hispanic_zcta == 0 ~ avg_pct_hispanic_zcta,
            .default = pct_hispanic_zcta
        ),
    ) %>%
    dplyr::ungroup() %>%
    dplyr::arrange(dplyr::desc(year)) %>%
    dplyr::distinct(zcta, .keep_all = TRUE) %>%
    dplyr::select(year, zcta, pct_black_zcta, pct_white_zcta, pct_asian_zcta, pct_hispanic_zcta)

#  %>%


race_by_zcta %>% dplyr::count(year) %>% print()


arrow::write_parquet(race_by_zcta %>% dplyr::select(-year), file.path(CENSUS_PATH, "race_by_zcta.parquet"))
