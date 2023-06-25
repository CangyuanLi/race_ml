import dataclasses
import functools
import json
import random
from typing import Literal

import cutils
import ethnicolr
import numpy as np
import pandas as pd
import polars as pl
import surgeo
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm

from data import FLZEmbedDataset
from models import FLLSTM, FLZLSTM, FLBiLSTM, FLEmbedBiLSTM, FLZBiLSTM, FLZEmbedBiLSTM
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN
from utils.paths import ETHNICOLR_PATH, FINAL_PATH, RETH_PATH, SURGEO_PATH
from utils.utils import prepare_name

RACES = list(RACES)
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}

pd.options.mode.chained_assignment = None

SampleType = Literal["fl", "flz"]
SurgeoModel = Literal["bifsg", "bisg"]


def test_ethnicolr(
    pd_df: pd.DataFrame, sample: str, overwrite: bool = False
) -> pl.DataFrame:
    outpath = ETHNICOLR_PATH / f"{sample}_eth_preds.parquet"
    if not overwrite:
        return pl.read_parquet(outpath)

    eth_pred = ethnicolr.pred_census_ln(
        pd_df[["last_name"]], lname_col="last_name", year=2010
    )
    eth_pred["last_name"] = eth_pred["last_name"].str.lower()
    eth_pred = eth_pred.rename(columns={"api": "asian"})
    eth_pred.columns = [f"eth_{col}" for col in eth_pred.columns]
    eth_pred = eth_pred.rename(columns={"eth_last_name": "last_name"})
    df = pl.from_pandas(eth_pred)

    df.write_parquet(outpath)

    return df


def test_rethnicity(sample: str) -> pl.DataFrame:
    return pl.read_parquet(RETH_PATH / f"{sample}_reth_preds.parquet")


def test_surgeo(
    pd_df: pd.DataFrame, sample: str, tag: SurgeoModel, overwrite: bool = False
) -> pl.DataFrame:
    outpath = SURGEO_PATH / f"{sample}_{tag}_preds.parquet"

    if not overwrite:
        return pl.read_parquet(outpath)

    if tag == "bifsg":
        model = surgeo.BIFSGModel()
        pd_df = model.get_probabilities(
            pd_df["first_name"], pd_df["last_name"], pd_df["zcta"]
        )
    elif tag == "bisg":
        model = surgeo.SurgeoModel()
        pd_df = model.get_probabilities(pd_df["last_name"], pd_df["zcta"])

    pd_df["index"] = pd_df.index
    pd_df = pd_df.rename(columns={"api": "asian"})
    pd_df = pd_df[["index"] + RACES]
    pd_df.columns = [f"{tag}_{col}" for col in pd_df.columns]
    pd_df = pd_df.rename(columns={f"{tag}_index": "index"})

    df = pl.from_pandas(pd_df)
    df.write_parquet(outpath)

    return df


def test_model(
    df: pl.DataFrame,
    folder: str,
    sample: str,
    model_num: int,
    tag: str,
    ncpus: int = 1,
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
    match model_name:
        case "FLZEmbedBiLSTM":
            base_model = FLZEmbedBiLSTM
        case "FLEmbedBiLSTM":
            base_model = FLEmbedBiLSTM
        case "FirstLastBiLSTM":
            base_model = FLBiLSTM
        case "FirstLastZctaBiLSTM":
            base_model = FLZBiLSTM
        case "FirstLastLSTM":
            base_model = FLLSTM
        case "FirstLastZctaLSTM":
            base_model = FLZLSTM
        case _:
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

    if ncpus == 1:
        results = infer_func(data, model=model, quiet=False)
    else:
        func = functools.partial(infer_func, model=model, quiet=True)
        with mp.Pool(processes=ncpus) as pool:
            results = []
            for res in pool.map(func, chunked):
                results += res

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


def prepare_test(
    filename: str, n: int = 25_000, overwrite: bool = False
) -> pl.DataFrame:
    if not overwrite:
        return pl.read_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    test = pl.read_parquet(FINAL_PATH / f"{filename}.parquet").filter(
        pl.arange(0, pl.count()).shuffle(seed=1).over("race_ethnicity") < n
    )
    test = reset_index(test)
    test = test.with_columns(
        is_asian=pl.col("race_ethnicity") == "asian",
        is_black=pl.col("race_ethnicity") == "black",
        is_hispanic=pl.col("race_ethnicity") == "hispanic",
        is_white=pl.col("race_ethnicity") == "white",
    )
    test.write_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    return test


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


@dataclasses.dataclass
class InferenceStats:
    tp: int = None
    tn: int = None
    fp: int = None
    fn: int = None
    accuracy: float = None
    recall: float = None
    precision: float = None
    fpr: float = None
    f1_score: float = None
    support: float = None


@dataclasses.dataclass
class InferenceStatsDF:
    max: pl.DataFrame = None
    rand: pl.DataFrame = None
    thresh: pl.DataFrame = None

    def print(self):
        print("max")
        print(self.max)

        print("rand")
        print(self.rand)

        print("thresh")
        print(self.thresh)


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


def calculate_stats(df: pl.DataFrame, tag: str) -> InferenceStatsDF:
    df = get_predictions(df, tag)
    df = calculate_confusion(df, tag)

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
            stats.f1_score = (2 * stats.precision * stats.recall) / (
                stats.precision + stats.recall
            )
            stats.fpr = stats.fp / (stats.fp + stats.tn)

            tp_col = df.get_column(f"{colname}_tp")
            stats.support = tp_col.is_not_null().sum() / tp_col.shape[0]
            res[race] = stats

        stats_df = stats_to_df(res)
        if rtype == "max":
            final_res.max = stats_df
        elif rtype == "rand":
            final_res.rand = stats_df
        elif rtype == "thresh":
            final_res.thresh = stats_df

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

    return pl.DataFrame(res).select(
        "race", "accuracy", "precision", "recall", "f1_score", "fpr", "support"
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
