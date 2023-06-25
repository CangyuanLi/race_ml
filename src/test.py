import datetime
import functools
import json
from pathlib import Path
from typing import Literal

import cutils
import ethnicolr
import pandas as pd
import polars as pl
import surgeo
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm

from data import FLZEmbedDataset
from models import FLLSTM, FLZLSTM, FLBiLSTM, FLEmbedBiLSTM, FLZBiLSTM, FLZEmbedBiLSTM
from utils.constants import (
    DEVICE,
    RACES,
    VALID_NAME_CHARS,
    VALID_NAME_CHARS_DICT,
    VALID_NAME_CHARS_LEN,
)
from utils.inference import calculate_stats
from utils.paths import (
    CENSUS_PATH,
    CW_PATH,
    ETHNICOLR_PATH,
    FINAL_PATH,
    PPP_PATH,
    RETH_PATH,
    SURGEO_PATH,
)
from utils.utils import prepare_name

pd.options.mode.chained_assignment = None

SampleType = Literal["fl", "flz"]
SurgeoModel = Literal["bifsg", "bisg"]


RACES = list(RACES)
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}


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


def reset_index(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(pl.Series(name="index", values=range(df.shape[0])))

    return df


def prepare_test(
    filename: str, n: int = 1_000_000, overwrite: bool = False
) -> pl.DataFrame:
    if not overwrite:
        return pl.read_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    test = pl.read_parquet(FINAL_PATH / f"{filename}.parquet").sample(n=n, seed=1)
    test = reset_index(test)
    test = test.with_columns(
        is_asian=pl.col("race_ethnicity") == "asian",
        is_black=pl.col("race_ethnicity") == "black",
        is_hispanic=pl.col("race_ethnicity") == "hispanic",
        is_white=pl.col("race_ethnicity") == "white",
    )
    test.write_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    return test


def test_ethnicolr(pd_df: pd.DataFrame, overwrite: bool = False) -> pl.DataFrame:
    outpath = ETHNICOLR_PATH / "ppp_eth_preds.parquet"
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


def test_rethnicity() -> pl.DataFrame:
    return pl.read_parquet(RETH_PATH / "ppp_reth_preds.parquet")


def test_surgeo(
    pd_df: pd.DataFrame, tag: SurgeoModel, overwrite: bool = False
) -> pl.DataFrame:
    outpath = SURGEO_PATH / f"ppp_{tag}_preds.parquet"

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
    model_num: int,
    tag: str,
    ncpus: int = 1,
    chunksize: int = 1024,
    overwrite: bool = False,
) -> pl.DataFrame:
    folder_path = FINAL_PATH / folder
    outpath = folder_path / f"ppp{model_num}_preds.parquet"

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
        results = infer_func(data, quiet=False)
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


def main():
    ppp = pl.scan_parquet(PPP_PATH / "ppp_clean.parquet")
    ppp = create_baseline_file(
        ppp, outpath=FINAL_PATH / "ppp_test.parquet", overwrite=True
    )
    ppp = prepare_test("ppp_test", 200_000, overwrite=True)
    ppp.write_csv("temp.csv")
    ppp_pandas = ppp.to_pandas()

    # stats
    with pl.Config(tbl_rows=100):
        print(ppp.get_column("state_abbrev").value_counts())
        print(ppp.get_column("race_ethnicity").value_counts())

    eth = test_ethnicolr(ppp_pandas, overwrite=True)
    reth = test_rethnicity()
    bisg = test_surgeo(ppp_pandas, "bisg", overwrite=True)
    bifsg = test_surgeo(ppp_pandas, "bifsg", overwrite=True)
    model = test_model(
        ppp,
        folder="flz_embed_bi_1",
        model_num=7,
        tag="flz_embed_bi",
        overwrite=False,
    )

    # for i in range(2, 16):
    #     model = test_model(
    #         ppp,
    #         folder="flz_embed_bi_5",
    #         model_num=i,
    #         tag="flz_embed_bi",
    #         overwrite=True,
    #     )

    preds = (
        ppp.join(eth, on="last_name", how="left")
        .join(reth, on=["first_name", "last_name"], how="left")
        .join(bisg, on="index", how="left")
        .join(bifsg, on="index", how="left")
        .join(model, on="index", how="left")
    )

    calculate_stats(preds, tag="bisg")
    calculate_stats(preds, tag="bifsg")
    calculate_stats(preds, tag="eth")
    calculate_stats(preds, tag="reth")
    calculate_stats(preds, tag="flz_embed_bi")


if __name__ == "__main__":
    main()
