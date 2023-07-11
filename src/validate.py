# Imports

import concurrent.futures
import dataclasses
import functools
import json
import random
import string
from typing import Literal, Optional

import cutils
import ethnicolr
import keras
import numpy as np
import pandas as pd
import polars as pl
import surgeo
import tensorflow as tf
import torch
import torch.nn as nn
import tqdm
from keras.utils import pad_sequences
from sklearn.metrics import classification_report

from data import FLEmbedDataset, FLZEmbedBinaryDataset, FLZEmbedDataset
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
from utils.constants import (
    DEVICE,
    RACES,
    RACES_DICT,
    VALID_NAME_CHARS_DICT,
    VALID_NAME_CHARS_LEN,
)
from utils.paths import (
    BAYES_PATH,
    CENSUS_PATH,
    CW_PATH,
    DIST_PATH,
    ETHNICOLR_PATH,
    FINAL_PATH,
    PPP_PATH,
    RETH_PATH,
    SURGEO_PATH,
    TBL_PATH,
)
from utils.utils import make_representative, normalize, prepare_name, softmax

# Globals

RACES = list(RACES)
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}

# Options

pd.options.mode.chained_assignment = None

# Types

SampleType = Literal["fl", "flz"]
SurgeoModel = Literal["bifsg", "bisg"]


@dataclasses.dataclass
class InferenceStats:
    tp: int = None
    tn: int = None
    fp: int = None
    fn: int = None
    accuracy: float = None
    recall: float = None
    precision: float = None
    specificity: float = None
    fpr: float = None
    fnr: float = None
    f1_score: float = None
    support: int = None
    coverage: float = None
    class_size: int = None


@dataclasses.dataclass
class InferenceStatsDF:
    max: pl.DataFrame = None
    rand: pl.DataFrame = None
    thresh: pl.DataFrame = None
    thresh50: pl.DataFrame = None
    thresh60: pl.DataFrame = None
    thresh70: pl.DataFrame = None

    def print(self):
        print("max")
        print(self.max)

        print("rand")
        print(self.rand)

        print("thresh")
        print(self.thresh)


# Functions


def prepare_test(
    filename: str, n: int = 25_000, balanced: bool = False, overwrite: bool = False
) -> pl.DataFrame:
    if not overwrite:
        return pl.read_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    df = pl.read_parquet(FINAL_PATH / f"{filename}.parquet")
    if balanced:
        test = df.filter(
            pl.arange(0, pl.count()).shuffle(seed=1).over("race_ethnicity") < n
        )
    else:
        test = make_representative(df, n=n)

    test = reset_index(test)
    test = test.with_columns(
        is_asian=pl.col("race_ethnicity") == "asian",
        is_black=pl.col("race_ethnicity") == "black",
        is_hispanic=pl.col("race_ethnicity") == "hispanic",
        is_white=pl.col("race_ethnicity") == "white",
    )
    test.write_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    return test


def create_lendio_ppp_file():
    df = (
        pl.from_pandas(pd.read_stata(PPP_PATH / "ppp_all_methods_sociocorr.dta"))
        .select(
            "first_name",
            "last_name",
            "raceethnicity",
            "borrowerzip",
            pl.col("^.*_sg_pct$"),
            pl.col("^.*_bifsg_pct$"),
        )
        .select(pl.exclude("^^true_.*$"))
        .select(pl.exclude("^^false_.*$"))
        .select(pl.exclude("^^random_.*$"))
        .select(pl.exclude("^^pop_black_.*$"))
        .with_columns(
            pl.col("raceethnicity")
            .str.to_lowercase()
            .str.replace_all("black or african american", "black")
        )
        .filter(pl.col("raceethnicity").is_in(RACES))
        .join(
            pl.read_parquet(CW_PATH / "zip_zcta_cw_final.parquet").with_columns(
                pl.col("zip").cast(pl.Int32)
            ),
            left_on="borrowerzip",
            right_on="zip",
            how="left",
        )
        .select("first_name", "last_name", "zcta", "raceethnicity")
        .with_columns(pl.col("first_name", "last_name").str.to_lowercase())
        .join(
            pl.read_parquet(CENSUS_PATH / "race_by_zcta.parquet"), how="left", on="zcta"
        )
        .rename({"raceethnicity": "race_ethnicity"})
    )
    test = reset_index(df)
    test = test.with_columns(
        is_asian=pl.col("race_ethnicity") == "asian",
        is_black=pl.col("race_ethnicity") == "black",
        is_hispanic=pl.col("race_ethnicity") == "hispanic",
        is_white=pl.col("race_ethnicity") == "white",
    )
    test.write_parquet(FINAL_PATH / "lendio_ppp_sample.parquet")


def get_lendio_ppp_sample():
    pass


def get_ppp_sample() -> pl.DataFrame:
    ppp = pl.scan_parquet(FINAL_PATH / "ppp_test.parquet")
    ppp = prepare_test("ppp_test", 200_000, overwrite=True)

    return ppp


def test_ethnicolr(
    pd_df: pd.DataFrame, sample: str, overwrite: bool = False
) -> pl.DataFrame:
    outpath = ETHNICOLR_PATH / f"{sample}_eth_preds.parquet"
    if not overwrite:
        return pl.read_parquet(outpath)

    eth_pred = ethnicolr.pred_fl_reg_name(
        pd_df[["first_name", "last_name"]],
        lname_col="last_name",
        fname_col="first_name",
    )
    eth_pred["last_name"] = eth_pred["last_name"].str.lower()
    eth_pred["first_name"] = eth_pred["first_name"].str.lower()
    eth_pred = eth_pred.rename(columns={"api": "asian"})
    eth_pred.columns = [f"eth_{col}" for col in eth_pred.columns]
    eth_pred = eth_pred.rename(
        columns={
            "eth_last_name": "last_name",
            "eth_first_name": "first_name",
            "eth_nh_black": "eth_black",
            "eth_nh_white": "eth_white",
        }
    )
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


def test_bisg(df: pl.DataFrame, sample: str, overwrite: bool = False) -> pl.DataFrame:
    outpath = BAYES_PATH / f"{sample}_ibisg_preds.parquet"

    if not overwrite:
        return pl.read_parquet(outpath)

    df = (
        BISG(df.select("index", "last_name", "zcta"))
        .get_probabilities("last_name", "zcta")
        .select(pl.exclude("last_name"))
        .select(
            pl.col("*").map_alias(
                lambda colname: f"ibisg_{colname}" if colname != "index" else colname
            )
        )
    )
    df.write_parquet(outpath)

    return df


def test_bifsg(df: pl.DataFrame, sample: str, overwrite: bool = False) -> pl.DataFrame:
    outpath = BAYES_PATH / f"{sample}_ibifsg_preds.parquet"

    if not overwrite:
        return pl.read_parquet(outpath)

    df = (
        BIFSG(df.select("index", "first_name", "last_name", "zcta"))
        .get_probabilities("first_name", "last_name", "zcta")
        .select(pl.exclude("first_name", "last_name"))
        .select(
            pl.col("*").map_alias(
                lambda colname: f"ibifsg_{colname}" if colname != "index" else colname
            )
        )
    )
    df.write_parquet(outpath)

    return df


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
            pct = None
        else:
            pct = res.tolist()[0][0]

        test[race].append(pct)

    preds = pl.DataFrame(test).select("index", race).rename({race: f"{tag}_{race}"})

    preds.write_parquet(outpath)

    return preds


def test_keras_model(tag: str = "keras", overwrite: bool = False):
    outpath = FINAL_PATH / "fl_keras/ppp_preds.parquet"

    if not overwrite and outpath.is_file():
        return pl.read_parquet(outpath)

    model = keras.models.load_model(FINAL_PATH / "fl_keras/model_opt.h5")
    test = pd.read_parquet(FINAL_PATH / "ppp_test_sample.parquet")

    def name2id(name: str, l: int = 15):
        ids = [0] * l
        for i, c in enumerate(name):
            if i < l:
                if c.isalpha():
                    ids[i] = char2id.get(c, char2id["U"])
                elif c in string.punctuation:
                    ids[i] = char2id.get(c, char2id[" "])
                else:
                    ids[i] = char2id.get(c, char2id["U"])
        return ids

    # create ASCII dictionary
    chars = ["E"] + [chr(i) for i in range(97, 123)] + [" ", "U"]
    char2id = {j: i for i, j in enumerate(chars)}

    X_test = [
        name2id(fn.lower()) + name2id(ln.lower())
        for fn, ln in zip(test["first_name"], test["last_name"])
    ]
    y_test = [RACES_DICT[r] for r in test["race_ethnicity"].tolist()]

    feature_len = 30  # cut texts after this number of words

    X_test = pad_sequences(X_test, maxlen=feature_len)
    y_test = tf.keras.utils.to_categorical(y_test, len(RACES))

    y_pred = model.predict(X_test, batch_size=512, verbose=1)
    y_pred_bool = np.argmax(y_pred, axis=1)
    print(classification_report(np.argmax(y_test, axis=1), y_pred_bool))

    preds = {r: [] for r in RACES}
    for row in y_pred:
        # print(row)
        # probs = softmax(row)
        for idx, p in enumerate(row):
            preds[RACE_MAPPER[idx]].append(p)

    preds = {f"{tag}_{k}": v for k, v in preds.items()}

    res = pl.DataFrame(preds).with_columns(
        pl.Series(name="first_name", values=test["first_name"].to_list()),
        pl.Series(name="last_name", values=test["last_name"].to_list()),
        pl.Series(name="zcta", values=test["zcta"].to_list()),
    )
    res.write_parquet(outpath)

    return res


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
            fn = FLEmbedDataset.encode_name(fn).unsqueeze(0)
            ln = FLEmbedDataset.encode_name(ln).unsqueeze(0)
            out, _ = model(fn, ln, hidden)
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
            stats.support = x
            stats.coverage = x / class_sizes[race]
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
            "coverage",
        )
        .with_columns(pl.col("*").exclude("race", "support").round(3))
        .rename(
            {
                "accuracy": "Accuracy",
                "race": "Race",
                "coverage": "Coverage",
                "precision": "Precision",
                "f1_score": "F1 Score",
                "support": "Support",
                "recall": "Recall",
            }
        )
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


def combine_binary_classifiers(dfs: list[pl.DataFrame], tag: str) -> pl.DataFrame:
    initial = dfs[0]
    rest = dfs[1:]

    for df in rest:
        initial = initial.join(df, on="index", how="left")

    cols = [f"{tag}_{r}" for r in RACES]

    initial = (
        initial.with_columns(pl.sum(cols).alias("sum"))
        .with_columns(pl.col(cols) / pl.col("sum"))
        .select(pl.exclude("sum"))
    )

    return initial


def combine_ml_bifsg(tag: str, prob_race_given_name: pl.DataFrame):
    cols = [f"{tag}_{r}" for r in RACES]

    prob_zcta_given_race = prob_race_given_name.join(
        pl.read_csv(DIST_PATH / "prob_zcta_given_race_2010.csv").with_columns(
            pl.col("zcta5").cast(str).str.zfill(5)
        ),
        left_on="zcta",
        right_on="zcta5",
        how="left",
    )

    numer = prob_race_given_name.select(cols) * prob_zcta_given_race.select(
        "asian", "black", "hispanic", "white"
    )
    denom = numer.sum(axis=1)
    probs = numer / denom

    probs = probs.rename({col: f"bayes_{col}" for col in cols})

    res = pl.concat(
        [prob_race_given_name.select("first_name", "last_name", "zcta"), probs],
        how="horizontal",
    )

    return res


def test_lookup_table(test_sample: pl.DataFrame):
    # cols = [f"{tag}_{r}" for r in RACES]
    unwanted_chars = string.digits + string.punctuation + string.whitespace

    def remove_chars(expr: pl.Expr, chars: str = unwanted_chars) -> pl.Expr:
        for char in chars:
            expr = expr.str.replace_all(char, "", literal=True)

        return expr

    test_sample = test_sample.with_columns(
        pl.col("first_name", "last_name")
        .pipe(remove_chars)
        .str.to_uppercase()
        .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
        .str.replace_all(r"\s?III\s*?$", "")
        .str.replace_all(r"\s?IV\s*?$", "")
        .map_alias(lambda colname: f"{colname}_tmp")
    ).with_columns(
        (pl.col("first_name_tmp") + " " + pl.col("last_name_tmp")).alias("name")
    )

    prob_race_given_name = test_sample.join(
        pl.read_csv(DIST_PATH / "prob_race_given_name.csv"), on="name", how="left"
    )

    prob_zcta_given_race = prob_race_given_name.join(
        pl.read_csv(DIST_PATH / "prob_zcta_given_race_2010.csv").with_columns(
            pl.col("zcta5").cast(str).str.zfill(5)
        ),
        left_on="zcta",
        right_on="zcta5",
        how="left",
    )

    cols = ["pct_asian", "pct_black", "pct_hispanic", "pct_white"]
    numer = prob_race_given_name.select(cols) * prob_zcta_given_race.select(
        "asian", "black", "hispanic", "white"
    )
    denom = numer.sum(axis=1)
    probs = numer / denom
    probs = probs.rename({col: f"lookup_{col}" for col in cols}).select(
        pl.col("*").map_alias(lambda colname: colname.replace("pct_", ""))
    )

    res = pl.concat(
        [prob_race_given_name.select("first_name", "last_name", "zcta"), probs],
        how="horizontal",
    )

    return res


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
        df = pl.read_parquet(FINAL_PATH / f"{sample}_test_sample.parquet")
        for i in range(5, 21):
            print(i)
            model = test_model(
                df,
                sample=sample,
                folder="fl_embed_bi_1",
                model_num=i,
                tag="fl_embed_bi",
                overwrite=True,
            )

            preds = df.join(model, on="index", how="left")
            print(calculate_stats(preds, tag="fl_embed_bi").max)


def view_binary_model_preds():
    for sample in ("ppp",):
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
                    overwrite=False,
                )

                preds = df.join(model, on="index", how="left")
                print(
                    calculate_stats_binary(
                        preds, tag="flz_embed_bi", race=race
                    ).thresh60
                )


def test_all():
    sample = "ppp"
    df = pl.read_parquet(FINAL_PATH / f"{sample}_test_sample.parquet")
    pd_df = df.to_pandas()

    overwrite = False

    bisg = test_surgeo(pd_df, sample=sample, tag="bisg", overwrite=overwrite)
    bifsg = test_surgeo(pd_df, sample=sample, tag="bifsg", overwrite=overwrite)
    ibisg = test_bisg(df, sample=sample, overwrite=overwrite)
    ibifsg = test_bifsg(df, sample=sample, overwrite=overwrite)
    eth = test_ethnicolr(pd_df, sample=sample, overwrite=overwrite)
    reth = test_rethnicity(sample=sample)
    keras_model = test_keras_model(tag="keras", overwrite=overwrite)
    bayesian_keras = combine_ml_bifsg("keras", keras_model)
    lookup = test_lookup_table(df)
    lookup.write_parquet("lookup.parquet")
    # model = combine_binary_classifiers(
    #     [
    #         pl.read_parquet(
    #             FINAL_PATH / f"flz_embed_bi_11asian/{sample}15_preds.parquet"
    #         ),
    #         pl.read_parquet(
    #             FINAL_PATH / f"flz_embed_bi_11black/{sample}15_preds.parquet"
    #         ),
    #         pl.read_parquet(
    #             FINAL_PATH / f"flz_embed_bi_11hispanic/{sample}15_preds.parquet"
    #         ),
    #         pl.read_parquet(
    #             FINAL_PATH / f"flz_embed_bi_11white/{sample}15_preds.parquet"
    #         ),
    #     ],
    #     "flz_embed_bi",
    # )

    preds = (
        df.join(eth, on=["first_name", "last_name"], how="left")
        .join(reth, on=["first_name", "last_name"], how="left")
        .join(bisg, on="index", how="left")
        .join(bifsg, on="index", how="left")
        .join(ibisg, on="index", how="left")
        .join(ibifsg, on="index", how="left")
        .join(
            keras_model.unique(["first_name", "last_name"]),
            on=["first_name", "last_name"],
            how="left",
        )
        .join(
            bayesian_keras.unique(["first_name", "last_name", "zcta"]),
            on=["first_name", "last_name", "zcta"],
            how="left",
        )
        .join(
            lookup.unique(["first_name", "last_name", "zcta"]),
            on=["first_name", "last_name", "zcta"],
            how="left",
        )
    )

    ensemble_adaptive = test_ensemble_classifier(
        preds,
        tags=["ibifsg", "ibisg", "bayes_keras"],
        weights=[10, 1, 1.5],
        ensemble_name="ensemble_adapt",
        adaptive=True,
    )
    ensemble_adaptive_imp = test_ensemble_classifier(
        preds,
        tags=["ibifsg", "ibisg", "bayes_keras"],
        weights=[1, 1, 1],
        ensemble_name="ensemble_adapt_imp",
        adaptive=True,
    )
    ensemble_adaptive_imp2 = test_ensemble_classifier(
        preds,
        tags=["ibifsg", "bayes_keras"],
        weights=[1, 0.001],
        ensemble_name="ensemble_adapt_imp2",
        adaptive=True,
    )
    preds = (
        preds.join(ensemble_adaptive, on="index", how="left")
        .join(ensemble_adaptive_imp, on="index", how="left")
        .join(ensemble_adaptive_imp2, on="index", how="left")
    )

    bisg_stats = calculate_stats(preds, "bisg")
    bifsg_stats = calculate_stats(preds, "bifsg")
    ibisg_stats = calculate_stats(preds, "ibisg")
    ibifsg_stats = calculate_stats(preds, "ibifsg")
    ensemble_adapt_stats = calculate_stats(preds, "ensemble_adapt")
    ensemble_adapt_imp_stats = calculate_stats(preds, "ensemble_adapt_imp")
    ensemble_adapt_imp_stats2 = calculate_stats(preds, "ensemble_adapt_imp2")
    reth_stats = calculate_stats(preds, "reth")
    eth_stats = calculate_stats(preds, "eth")
    keras_stats = calculate_stats(preds, "keras")
    bayesian_keras_stats = calculate_stats(preds, "bayes_keras")
    lookup_stats = calculate_stats(preds, "lookup")

    cols = [
        "Race",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "Coverage",
        "Support",
    ]
    print(bisg_stats.max.select(cols))
    print(ibisg_stats.max.select(cols))
    print(bifsg_stats.max.select(cols))
    print(ibifsg_stats.max.select(cols))
    print(eth_stats.max.select(cols))
    print(keras_stats.max.select(cols))
    print(bayesian_keras_stats.max.select(cols))
    print(ensemble_adapt_stats.max.select(cols))
    print(ensemble_adapt_imp_stats.max.select(cols))
    print(ensemble_adapt_imp_stats2.max.select(cols))

    print(ensemble_adapt_imp_stats.max.select(cols + ["fnr"]))

    def write_latex(stats: InferenceStatsDF, path, cols=cols):
        for tag, df in {"max": stats.max, "thresh50": stats.thresh}.items():
            df = df.select(cols).fill_null(0).with_columns(pl.col("*").cast(str))
            if "Support" in cols:
                df = df.with_columns(pl.col("Support").apply(lambda x: f"{int(x):,}"))
            if "Race" in cols:
                df = df.with_columns(pl.col("Race").str.to_titlecase())
            df.to_pandas().to_latex(TBL_PATH / f"{path}_{tag}.tex", index=False)

    write_latex(bisg_stats, "bisg_stats")
    write_latex(ibisg_stats, "ibisg_stats")
    write_latex(bifsg_stats, "bifsg_stats")
    write_latex(ibifsg_stats, "ibifsg_stats")
    write_latex(reth_stats, "reth_stats")
    write_latex(eth_stats, "eth_stats")
    write_latex(keras_stats, "fl_stats")
    write_latex(bayesian_keras_stats, "fl_bayes_stats")
    write_latex(ensemble_adapt_imp_stats, "ensemble_stats")
    write_latex(lookup_stats, "lookup_stats")

    # print(model_stats.max.select("race", "precision", "tpr", "f1_score", "support"))

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
    # view_model_preds()
    # test_keras_model(overwrite=True)
    # stop
    # get_ppp_sample()
    test_all()
    # view_model_preds()
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


if __name__ == "__main__":
    main()
