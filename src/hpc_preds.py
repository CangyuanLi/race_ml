# Imports


import json
import random
from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import tqdm

from data import FLZEmbedDataset
from models import FLLSTM, FLZLSTM, FLBiLSTM, FLEmbedBiLSTM, FLZBiLSTM, FLZEmbedBiLSTM
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH
from utils.utils import prepare_name

# Globals

RACES = list(RACES)
RACE_MAPPER = {i: r for i, r in enumerate(RACES)}


# Types

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
        test = df.sample(n=n, seed=1)

    test = reset_index(test)
    test = test.with_columns(
        is_asian=pl.col("race_ethnicity") == "asian",
        is_black=pl.col("race_ethnicity") == "black",
        is_hispanic=pl.col("race_ethnicity") == "hispanic",
        is_white=pl.col("race_ethnicity") == "white",
    )
    test.write_parquet(FINAL_PATH / f"{filename}_sample.parquet")

    return test


def get_ppp_sample() -> pl.DataFrame:
    ppp = pl.scan_parquet(FINAL_PATH / "ppp_test.parquet")
    ppp = prepare_test("ppp_test", 200_000, overwrite=False)

    return ppp


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

    if nthreads == 1:
        results = infer_func(data, model=model, quiet=False)

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


def main():
    df = get_ppp_sample()
    sample = "ppp"

    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_6",
            model_num=i,
            tag="flz_embed_bi",
            overwrite=True,
        )

    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_5",
            model_num=i,
            tag="flz_embed_bi",
            overwrite=True,
        )

    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_4",
            model_num=i,
            tag="flz_embed_bi",
            overwrite=True,
        )

    df = pl.read_parquet(FINAL_PATH / "lendio_ppp_sample.parquet")
    sample = "lendio_ppp"
    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_6",
            model_num=i,
            tag="flz_embed_bi",
            nthreads=4,
            overwrite=True,
        )

    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_5",
            model_num=i,
            tag="flz_embed_bi",
            overwrite=True,
        )

    for i in range(1, 16):
        model = test_model(
            df=df,
            sample=sample,
            folder="flz_embed_bi_4",
            model_num=i,
            tag="flz_embed_bi",
            overwrite=True,
        )


if __name__ == "__main__":
    main()
