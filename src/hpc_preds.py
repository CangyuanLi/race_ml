# Imports

import concurrent.futures
import dataclasses
import functools
import json
import random
from typing import Literal, Optional

import pandas as pd
import polars as pl
import torch
import torch.nn as nn

from data import FLZEmbedBinaryDataset, FLZEmbedDataset
from models import (
    BIFSG,
    BISG,
    FLLSTM,
    FLZLSTM,
    FLBiLSTM,
    FLEmbedBiLSTM,
    FLZBiLSTM,
    FLZEmbedBiLSTM,
    FLZEmbedBiLSTMBinary,
)
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN
from utils.paths import (
    BAYES_PATH,
    CENSUS_PATH,
    CW_PATH,
    ETHNICOLR_PATH,
    FINAL_PATH,
    PPP_PATH,
    RETH_PATH,
    SURGEO_PATH,
)
from utils.utils import make_representative, prepare_name

# Globals

RACES = list(RACES)
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}

# Options

# Functions


def test_model(
    df: pl.DataFrame,
    folder: str,
    sample: str,
    model_num: int,
    tag: str,
    nthreads: Optional[int] = 1,
    chunksize: int = 1024,
    overwrite: bool = False,
) -> pl.DataFrame:
    folder_path = FINAL_PATH / folder
    outpath = folder_path / f"{sample}{model_num}_preds.parquet"

    if not overwrite:
        return pl.read_parquet(outpath)

    with open(folder_path / "hyperparameters.json") as f:
        parameters = json.load(f)

    model_name = parameters["model"]
    if model_name == "FLZEmbedBiLSTM":
        base_model = FLZEmbedBiLSTM
    elif model_name == "FLEmbedBiLSTM":
        base_model = FLEmbedBiLSTM
    elif model_name == "FirstLastBiLSTM":
        base_model = FLBiLSTM
    elif model_name == "FirstLastZctaBiLSTM":
        base_model = FLZBiLSTM
    elif model_name == "FirstLastLSTM":
        base_model = FLLSTM
    elif model_name == "FirstLastZctaLSTM":
        base_model = FLZLSTM
    else:
        raise ValueError("Wrong")

    if "Embed" in model_name:
        model = base_model(
            input_size=VALID_NAME_CHARS_LEN,
            embedding_dim=parameters["embedding_dim"],
            hidden_size=parameters["hidden_size"],
            output_size=len(RACES),
            dropout=parameters["dropout"],
            num_layers=parameters["num_layers"],
        ).to(DEVICE)

    if model_name.startswith("FLZ"):
        infer_func = infer_flz
        data = get_flz(df)
    elif model_name.startswith("FL"):
        infer_func = infer_fl
        data = get_fl(df)

    model.load_state_dict(
        torch.load(
            folder_path / f"model{model_num}.pth",
            map_location=DEVICE,
        )
    )

    test = df.to_dict()
    for race in RACES:
        test[race] = []

    chunked = cutils.chunk_seq(data, chunksize)

    if nthreads == 1:
        results = infer_func(data, model=model, quiet=False)
    else:
        func = functools.partial(infer_func, model=model, quiet=True)

        with tqdm.tqdm(total=len(chunked)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(nthreads) as pool:
                results = []
                for res in pool.map(func, chunked):
                    results += res
                    pbar.update(1)

    for res in results:
        if res is None:
            percentages = [None, None, None, None]
        else:
            percentages = nn.functional.softmax(res, dim=1).tolist()[0]

        for idx, p in enumerate(percentages):
            test[RACE_MAPPER[idx]].append(p)

    preds = (
        pl.DataFrame(test)
        .select("index", "asian", "black", "hispanic", "white")
        .rename(
            {
                "asian": f"{tag}_asian",
                "black": f"{tag}_black",
                "hispanic": f"{tag}_hispanic",
                "white": f"{tag}_white",
            }
        )
    )

    preds.write_parquet(outpath)

    return preds


def test_binary_model(
    df: pl.DataFrame,
    folder: str,
    sample: str,
    model_num: int,
    tag: str,
    nthreads: Optional[int] = 1,
    chunksize: int = 1024,
    overwrite: bool = False,
) -> pl.DataFrame:
    folder_path = FINAL_PATH / folder
    outpath = folder_path / f"{sample}{model_num}_preds.parquet"

    if not overwrite:
        return pl.read_parquet(outpath)

    with open(folder_path / "hyperparameters.json") as f:
        parameters = json.load(f)

    model_name = parameters["model"]
    if model_name == "FLZEmbedBiLSTMBinary":
        base_model = FLZEmbedBiLSTMBinary
    else:
        raise ValueError("Wrong")

    if "Embed" in model_name:
        model = base_model(
            input_size=VALID_NAME_CHARS_LEN,
            embedding_dim=parameters["embedding_dim"],
            hidden_size=parameters["hidden_size"],
            output_size=1,
            dropout=parameters["dropout"],
            num_layers=parameters["num_layers"],
        ).to(DEVICE)

    if model_name.startswith("FLZ"):
        infer_func = infer_flz
        data = get_flz(df)
    elif model_name.startswith("FL"):
        infer_func = infer_fl
        data = get_fl(df)

    model.load_state_dict(
        torch.load(
            folder_path / f"model{model_num}.pth",
            map_location=DEVICE,
        )
    )

    race = "".join(c for c in folder.split("_")[-1] if not c.isdigit())

    test = df.to_dict()
    test[race] = []

    if nthreads == 1:
        results = infer_func(data, model=model, quiet=False)
    else:
        # func = functools.partial(infer_func, model=model, quiet=True)

        # with tqdm.tqdm(total=len(chunked)) as pbar:
        #     with concurrent.futures.ThreadPoolExecutor(nthreads) as pool:
        #         results = []
        #         for res in pool.map(func, chunked):
        #             results += res
        #             pbar.update(1)

    for res in results:
        if res is None:
            pct = None
        else:
            pct = res.tolist()[0][0]

        test[race].append(pct)

    preds = pl.DataFrame(test).select("index", race).rename({race: f"{tag}_{race}"})

    preds.write_parquet(outpath)

    return preds


def get_fl(df: pl.DataFrame) -> list[tuple[str, str]]:
    return df.select("first_name", "last_name").rows()


def get_flz(df: pl.DataFrame) -> list[tuple[str, str, float, float, float, float]]:
    return df.select(
        "first_name",
        "last_name",
        "pct_asian_zcta",
        "pct_black_zcta",
        "pct_hispanic_zcta",
        "pct_white_zcta",
    ).rows()


def infer_fl(batch: list[tuple[str, str]], model: nn.Module, quiet: bool = False):
    with torch.inference_mode():
        model.eval()
        outputs = []
        for fn, ln in tqdm.tqdm(batch, disable=quiet):
            model.zero_grad(set_to_none=True)
            hidden = model.init_hidden(1)
            name = prepare_name(
                fn, ln, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN, DEVICE
            ).unsqueeze(0)
            out, _ = model(name, hidden)
            outputs.append(out)

    return outputs


def is_null(x):
    if x is None:
        return True

    return np.isnan(x)


def get_random_race(choices: list[str], weights: list[float]):
    weights_clean = [0 if is_null(weight) else weight for weight in weights]
    if sum(weights_clean) == 0:
        return None
    else:
        return random.choices(choices, weights_clean)[0]


def get_predictions(df: pl.DataFrame, tag: str) -> pl.DataFrame:
    rand_race = []
    max_race = []
    threshold_race = []
    for pcts in df.select(
        f"{tag}_asian", f"{tag}_black", f"{tag}_hispanic", f"{tag}_white"
    ).rows():
        if all(p is None for p in pcts):
            rand_race.append(None)
            max_race.append(None)
            threshold_race.append(None)

            continue

        rand_race.append(get_random_race(RACES, pcts))

        max_val = max(pcts)
        idx = pcts.index(max_val)
        race = RACE_MAPPER[idx]

        max_race.append(race)
        threshold_race.append(race if max_val > 0.5 else None)

    df = df.with_columns(
        pl.Series(f"{tag}_max_race", max_race),
        pl.Series(f"{tag}_rand_race", rand_race),
        pl.Series(f"{tag}_thresh_race", threshold_race),
    )

    return df


def get_predictions_binary(df: pl.DataFrame, tag: str, race: str) -> pl.DataFrame:
    thresh50 = []
    thresh60 = []
    thresh70 = []
    for pct in df.get_column(f"{tag}_{race}"):
        if pct is None:
            thresh50.append(None)
            thresh60.append(None)
            thresh70.append(None)

            continue

        thresh50.append(race if pct > 0.5 else "not_race")
        thresh60.append(race if pct > 0.6 else "not_race")
        thresh70.append(race if pct > 0.7 else "not_race")

    df = df.with_columns(
        pl.Series(f"{tag}_thresh50_race", thresh50),
        pl.Series(f"{tag}_thresh60_race", thresh60),
        pl.Series(f"{tag}_thresh70_race", thresh70),
    )

    return df


def calculate_confusion(df: pl.DataFrame, tag: str) -> pl.DataFrame:
    for rtype in ("max", "rand", "thresh"):
        for race in RACES:
            p_race = f"{tag}_{rtype}_race"
            df = df.with_columns(
                ((pl.col(p_race) == pl.lit(race)) & (pl.col(f"is_{race}"))).alias(
                    f"{tag}_{rtype}_{race}_tp"
                ),
                ((pl.col(p_race) == pl.lit(race)) & (~pl.col(f"is_{race}"))).alias(
                    f"{tag}_{rtype}_{race}_fp"
                ),
                ((pl.col(p_race) != pl.lit(race)) & (~pl.col(f"is_{race}"))).alias(
                    f"{tag}_{rtype}_{race}_tn"
                ),
                ((pl.col(p_race) != pl.lit(race)) & (pl.col(f"is_{race}"))).alias(
                    f"{tag}_{rtype}_{race}_fn"
                ),
            )

    return df


def calculate_confusion_binary(df: pl.DataFrame, tag: str, race: str) -> pl.DataFrame:
    for rtype in ("thresh50", "thresh60", "thresh70"):
        p_race = f"{tag}_{rtype}_race"
        df = df.with_columns(
            ((pl.col(p_race) == pl.lit(race)) & (pl.col(f"is_{race}"))).alias(
                f"{tag}_{rtype}_{race}_tp"
            ),
            ((pl.col(p_race) == pl.lit(race)) & (~pl.col(f"is_{race}"))).alias(
                f"{tag}_{rtype}_{race}_fp"
            ),
            ((pl.col(p_race) != pl.lit(race)) & (~pl.col(f"is_{race}"))).alias(
                f"{tag}_{rtype}_{race}_tn"
            ),
            ((pl.col(p_race) != pl.lit(race)) & (pl.col(f"is_{race}"))).alias(
                f"{tag}_{rtype}_{race}_fn"
            ),
        )

    return df


def calculate_stats(df: pl.DataFrame, tag: str) -> InferenceStatsDF:
    df = get_predictions(df, tag)
    df = calculate_confusion(df, tag)

    class_sizes = {r: df.filter(pl.col("race_ethnicity") == r).shape[0] for r in RACES}

    final_res = InferenceStatsDF()
    for rtype in ("max", "rand", "thresh"):
        res = {}
        for race in RACES:
            colname = f"{tag}_{rtype}_{race}"
            stats = InferenceStats(
                tp=df.get_column(f"{colname}_tp").sum(),
                tn=df.get_column(f"{colname}_tn").sum(),
                fp=df.get_column(f"{colname}_fp").sum(),
                fn=df.get_column(f"{colname}_fn").sum(),
            )
            stats.accuracy = (stats.tp + stats.tn) / (
                stats.tp + stats.tn + stats.fp + stats.fn
            )
            stats.precision = stats.tp / (stats.tp + stats.fp)
            stats.recall = stats.tp / (stats.tp + stats.fn)
            stats.specificity = stats.tn / (stats.tn + stats.fp)
            stats.f1_score = (2 * stats.precision * stats.recall) / (
                stats.precision + stats.recall
            )
            stats.fpr = stats.fp / (stats.fp + stats.tn)
            stats.fnr = stats.fn / (stats.fn + stats.tp)

            x = (
                df.filter(pl.col("race_ethnicity") == race)
                .get_column(f"{colname}_tp")
                .is_not_null()
                .sum()
            )
            stats.support = x / class_sizes[race]
            stats.class_size = class_sizes[race]

            res[race] = stats

        stats_df = stats_to_df(res)
        if rtype == "max":
            final_res.max = stats_df
        elif rtype == "rand":
            final_res.rand = stats_df
        elif rtype == "thresh":
            final_res.thresh = stats_df

    return final_res


def calculate_stats_binary(df: pl.DataFrame, tag: str, race: str) -> InferenceStatsDF:
    df = get_predictions_binary(df, tag, race)
    df = calculate_confusion_binary(df, tag, race)

    class_size = df.filter(pl.col("race_ethnicity") == race).shape[0]

    final_res = InferenceStatsDF()
    for rtype in ("thresh50", "thresh60", "thresh70"):
        res = {}

        colname = f"{tag}_{rtype}_{race}"
        stats = InferenceStats(
            tp=df.get_column(f"{colname}_tp").sum(),
            tn=df.get_column(f"{colname}_tn").sum(),
            fp=df.get_column(f"{colname}_fp").sum(),
            fn=df.get_column(f"{colname}_fn").sum(),
        )
        stats.accuracy = (stats.tp + stats.tn) / (
            stats.tp + stats.tn + stats.fp + stats.fn
        )
        stats.precision = stats.tp / (stats.tp + stats.fp)
        stats.recall = stats.tp / (stats.tp + stats.fn)
        stats.specificity = stats.tn / (stats.tn + stats.fp)
        stats.f1_score = (2 * stats.precision * stats.recall) / (
            stats.precision + stats.recall
        )
        stats.fpr = stats.fp / (stats.fp + stats.tn)
        stats.fnr = stats.fn / (stats.fn + stats.tp)

        x = (
            df.filter(pl.col("race_ethnicity") == race)
            .get_column(f"{colname}_tp")
            .is_not_null()
            .sum()
        )
        stats.support = x / class_size
        stats.class_size = class_size

        res[race] = stats

        stats_df = stats_to_df(res)
        if rtype == "thresh50":
            final_res.thresh50 = stats_df
        elif rtype == "thresh60":
            final_res.thresh60 = stats_df
        elif rtype == "thresh70":
            final_res.thresh70 = stats_df

    return final_res


def compare(df: pl.DataFrame, against: pl.DataFrame) -> pl.DataFrame:
    against = against.rename(
        {col: f"{col}_1" for col in against.columns if col != "race"}
    )
    comparison = df.join(against, on="race", how="inner", validate="1:1")
    columns = [c for c in df.columns if c != "race"]
    for col in columns:
        comparison = comparison.with_columns(pl.col(col) - pl.col(f"{col}_1"))

    comparison = comparison.select(["race"] + columns)

    return comparison


def stats_to_df(race_stats: dict[str, InferenceStats]) -> pl.DataFrame:
    res = {st_nm: [] for st_nm in InferenceStats().__dict__.keys()}
    res["race"] = []
    for race, stats in race_stats.items():
        res["race"].append(race)
        for st_nm, st_val in stats.__dict__.items():
            res[st_nm].append(st_val)

    return (
        pl.DataFrame(res)
        .select(
            "race",
            "accuracy",
            "precision",
            "recall",  # tpr
            "specificity",  # tnr
            "f1_score",
            "fpr",
            "fnr",
            "support",
            "class_size",
        )
        .with_columns(pl.col("*").exclude("race", "class_size").round(3))
        .rename({"recall": "tpr", "specificity": "tnr", "class_size": "class size"})
    )


def reset_index(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.Series(name="index", values=range(df.shape[0])))

    return df


def infer_flz(
    batch: list[tuple[str, str, float, float, float, float]],
    model: nn.Module,
    quiet: bool = False,
):
    with torch.inference_mode():
        model.eval()
        outputs = []
        for tup in tqdm.tqdm(batch, disable=quiet):
            if any(x is None for x in tup):
                outputs.append(None)
                continue

            fn, ln, asian_pct, black_pct, hisp_pct, white_pct = tup
            model.zero_grad(set_to_none=True)
            hidden = model.init_hidden(1)
            fn = FLZEmbedDataset.encode_name(fn).unsqueeze(0)
            ln = FLZEmbedDataset.encode_name(ln).unsqueeze(0)
            pct = torch.tensor(
                (asian_pct, black_pct, hisp_pct, white_pct), device=DEVICE
            ).unsqueeze(0)
            out, _ = model(fn, ln, pct, hidden)
            outputs.append(out)

    return outputs


def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)

    return [w / total for w in weights]


def test_ensemble_classifier(
    df: pl.DataFrame,
    tags: list[str],
    weights: list[float],
    ensemble_name: str = "ensemble",
    adaptive: bool = False,
) -> pl.DataFrame:
    weights = normalize_weights(weights)

    if adaptive:
        index = df.get_column("index").to_list()
        res = {f"{ensemble_name}_{race}": [] for race in RACES}
        cols = list(res.keys())
        for race in RACES:
            col = f"{ensemble_name}_{race}"
            input_cols = [f"{tag}_{race}" for tag in tags]

            for row in df.select(input_cols).iter_rows():
                valid_weights = [
                    weights[idx] for idx, r in enumerate(row) if not is_null(r)
                ]

                valid_weights = []
                valid_inputs = []
                for r, w in zip(row, weights):
                    if not is_null(r):
                        valid_weights.append(w)
                        valid_inputs.append(r)

                valid_weights = normalize_weights(valid_weights)

                res[col].append(sum(i * w for i, w in zip(valid_inputs, valid_weights)))

        res["index"] = index
        res = pl.DataFrame(res)
    else:
        cols = []
        for race in RACES:
            col = f"{ensemble_name}_{race}"
            df = df.with_columns(pl.lit(0).alias(col))

            for tag, weight in zip(tags, weights, strict=True):
                input_col = f"{tag}_{race}"
                df = df.with_columns(pl.col(col) + (pl.col(input_col) * pl.lit(weight)))

            cols.append(col)

        res = df.select(["index"] + cols)

    res = res.with_columns(pl.sum(cols).alias("sum"))
    for col in cols:
        res = res.with_columns(pl.col(col) / pl.col("sum"))

    res = res.select(pl.exclude("sum"))

    return res


def view_model_preds():
    for sample in ("ppp",):
        print(sample)
        df = pl.read_parquet(FINAL_PATH / f"{sample}_test_sample.parquet").sample(100)
        for i in range(1, 21):
            print(i)
            model = test_binary_model(
                df,
                sample=sample,
                folder="flz_embed_bi_11asian",
                model_num=i,
                tag="flz_embed_bi",
                overwrite=True,
            )

            preds = df.join(model, on="index", how="left")
            print(calculate_stats_binary(preds, tag="flz_embed_bi", race="asian"))


def view_binary_model_preds():
    for sample in ("ppp", "flz"):
        for race in ("asian", "black", "white", "hispanic"):
            df = pl.read_parquet(FINAL_PATH / f"{sample}_test_sample.parquet")
            for i in range(5, 21):
                print(i)
                model = test_binary_model(
                    df,
                    sample=sample,
                    folder=f"flz_embed_bi_11{race}",
                    model_num=i,
                    tag="flz_embed_bi",
                    nthreads=4,
                    overwrite=True,
                )


def test_all():
    sample = "ppp"
    df = pl.read_parquet(FINAL_PATH / f"{sample}_test_sample.parquet")
    pd_df = df.to_pandas()

    overwrite = True

    bisg = test_surgeo(pd_df, sample=sample, tag="bisg", overwrite=overwrite)
    bifsg = test_surgeo(pd_df, sample=sample, tag="bifsg", overwrite=overwrite)
    ibisg = test_bisg(df, sample=sample, overwrite=overwrite)
    ibifsg = test_bifsg(df, sample=sample, overwrite=overwrite)
    eth = test_ethnicolr(pd_df, sample=sample, overwrite=overwrite)
    reth = test_rethnicity(sample=sample)
    model = test_model(
        df,
        sample=sample,
        folder="flz_embed_bi_10",
        model_num=13,
        tag="flz_embed_bi",
        overwrite=False,
    )

    preds = (
        df.join(eth, on=["first_name", "last_name"], how="left")
        .join(reth, on=["first_name", "last_name"], how="left")
        .join(bisg, on="index", how="left")
        .join(bifsg, on="index", how="left")
        .join(ibisg, on="index", how="left")
        .join(ibifsg, on="index", how="left")
        .join(model, on="index", how="left")
    )
    ensemble = test_ensemble_classifier(
        preds, tags=["bisg", "flz_embed_bi"], weights=[1, 0.7]
    )
    ensemble_adaptive = test_ensemble_classifier(
        preds,
        tags=["bifsg", "bisg", "flz_embed_bi"],
        weights=[1, 1, 1],
        ensemble_name="ensemble_adapt",
        adaptive=True,
    )
    preds = preds.join(ensemble, on="index", how="left").join(
        ensemble_adaptive, on="index", how="left"
    )

    bisg_stats = calculate_stats(preds, "bisg")
    bifsg_stats = calculate_stats(preds, "bifsg")
    ibisg_stats = calculate_stats(preds, "ibisg")
    ibifsg_stats = calculate_stats(preds, "ibifsg")
    # ensemble_stats = calculate_stats(preds, "ensemble")

    # reth_stats = calculate_stats(preds, "reth")
    eth_stats = calculate_stats(preds, "eth")
    # ensemble_adapt_stats = calculate_stats(preds, "ensemble_adapt")
    # model_stats = calculate_stats(preds, "flz_embed_bi")

    print(bisg_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    print(ibisg_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    print(bifsg_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    print(ibifsg_stats.max.select("race", "precision", "tpr", "f1_score", "support"))

    print(eth_stats.max.select("race", "precision", "tpr", "f1_score", "support"))

    # print(eth_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    # print(reth_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    # print(model_stats.max.select("race", "precision", "tpr", "f1_score", "support"))
    # print(ensemble_stats.max)
    # print(ensemble_adapt_stats.max)
    # print(bifsg_stats.max)
    # print(bisg_stats.max)

    # print(ensemble_adapt_stats.thresh)

    # ensemble_stats.max.to_pandas().to_latex()

    # print(ensemble_adapt_stats.max.to_pandas().to_markdown())


def main():
    view_binary_model_preds()
    # df = pl.read_parquet(FINAL_PATH / "flz_embed_bi_10/ppp13_preds.parquet")
    # ppp = pl.read_parquet(FINAL_PATH / "ppp_test_sample.parquet")
    # ppp.join(df, on="index", how="left").select(
    #     "first_name", "last_name", "race_ethnicity", pl.col("^^flz_embed_bi_.*$")
    # ).write_csv("temp.csv")
    # Zest Alabama

    # Black

    # TPR - 0.739
    # TNR - 0.968
    # FPR - 0.032
    # FNR - 0.261

    # Hispanic

    # TPR - 0.855
    # TNR - 0.987
    # FPR - 0.013
    # FNR - 0.145

    # White

    # TPR - 0.951
    # TNR - 0.763
    # FPR - 0.237
    # FNR - 0.049


#     Zest PPP

# Black

# TPR- .783
# TNR- .934
# FPR- .066
# FNR- .217

# Hispanic

# TPR- .787
# TNR- .969
# FPR- .031
# FNR- .213

# White

# TPR- .867
# TNR- .843
# FPR- .157
# FNR- .133
#     # prepare_test("flz_test", n=200_000, balanced=False, overwrite=True)
# prepare_test("flz_dups_test", n=200_000, balanced=False, overwrite=True)
# prepare_test("flz_imb_dups_test", n=200_000, balanced=False, overwrite=True)
# df = pl.read_parquet(FINAL_PATH / "lendio_ppp_sample.parquet")
# df = pl.read_parquet(FINAL_PATH / "ppp_test_sample.parquet")
# pd_df = df.to_pandas()
# sample = "ppp"

# bisg = test_surgeo(pd_df, sample=sample, tag="bisg", overwrite=False)
# bifsg = test_surgeo(pd_df, sample=sample, tag="bifsg", overwrite=False)
# eth = test_ethnicolr(pd_df, sample=sample, overwrite=False)
# reth = test_rethnicity(sample=sample)
# model = test_model(
#     df,
#     sample=sample,
#     folder="flz_embed_bi_5",
#     model_num=14,
#     tag="flz_embed_bi",
#     overwrite=False,
# )

# preds = (
#     df.join(eth, on="last_name", how="left")
#     .join(reth, on=["first_name", "last_name"], how="left")
#     .join(bisg, on="index", how="left")
#     .join(bifsg, on="index", how="left")
#     .join(model, on="index", how="left")
# )
# ensemble = test_ensemble_classifier(
#     preds, tags=["bifsg", "flz_embed_bi"], weights=[1, 1]
# )
# preds = preds.join(ensemble, on="index", how="left")

# bisg_stats = calculate_stats(preds, tag="bisg")
# bifsg_stats = calculate_stats(preds, tag="bifsg")
# model_stats = calculate_stats(preds, tag="flz_embed_bi")
# reth_stats = calculate_stats(preds, tag="reth")
# ensemble_stats = calculate_stats(preds, tag="ensemble")
# # print(ensemble_stats.max)
# # print(compare(ensemble_stats.max, bisg_stats.max))
# # print(compare(ensemble_stats.max, bifsg_stats.max))
# # print(compare(model_stats.max, reth_stats.max))
# print(bisg_stats.max)
# print(model_stats.max)
# print(compare(model_stats.max, bisg_stats.max))


if __name__ == "__main__":
    main()
