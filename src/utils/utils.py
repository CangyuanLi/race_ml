from __future__ import annotations

import concurrent.futures
import math
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import polars as pl
import torch
import tqdm

ConcurrentStrategy = Literal["process", "thread", "single"]


def encode_name(
    name: str,
    valid_name_chars_dict: dict[str, int],
    valid_name_chars_len: int,
    device: torch.device,
) -> torch.Tensor:
    encoded = torch.zeros(len(name), valid_name_chars_len, device=device)
    for idx, c in enumerate(name):
        encoded[idx][valid_name_chars_dict[c]] = 1

    return encoded


def normalize(lst: list[float]) -> list[float]:
    sum_ = sum(lst)

    return [x / sum_ for x in lst]


def softmax(lst: list[float]) -> list[float]:
    lst = [math.exp(x) for x in lst]
    sum_ = sum(lst)

    return [x / sum_ for x in lst]


def make_representative(
    df: pl.DataFrame,
    distribution: dict[str, float] = {
        "asian": 0.059,
        "black": 0.126,
        "hispanic": 0.189,
        "white": 0.593,
    },
    n: Optional[int] = None,
) -> pl.DataFrame:
    normalized = normalize(distribution.values())
    distribution = {k: v for k, v in zip(distribution.keys(), normalized)}

    value_counts = {
        race: count
        for race, count in df.get_column("race_ethnicity").value_counts().iter_rows()
    }
    allowed = []
    for race, count in value_counts.items():
        if race not in distribution:
            continue

        if distribution[race] == 0:
            allowed.append(0)
        else:
            allowed.append(count // distribution[race])

    allowed = min(allowed)

    if n is not None:
        if n < allowed:
            allowed = n

    distribution_counts = {
        race: math.floor(allowed * pct) for race, pct in distribution.items()
    }

    dfs = []
    for race, count in distribution_counts.items():
        race_df = df.filter(pl.col("race_ethnicity") == race)
        if count > race_df.shape[0]:
            count = race_df.shape[0]

        dfs.append(race_df.sample(count, seed=1))

    return pl.concat(dfs, how="vertical")


def encode_race(
    race: str, races_dict: dict[str, int], device: torch.device
) -> torch.Tensor:
    return torch.tensor([races_dict[race]], device=device)


def encode_scalar(scalar: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(scalar, device=device).unsqueeze(0).unsqueeze(1)


def prepare_name(
    first_name: str,
    last_name: str,
    valid_name_chars_dict: dict[str, int],
    valid_name_chars_len: int,
    device: torch.device,
):
    return torch.cat(
        [
            encode_name(
                first_name, valid_name_chars_dict, valid_name_chars_len, device
            ),
            encode_name(last_name, valid_name_chars_dict, valid_name_chars_len, device),
        ],
        dim=0,
    )


def coerce_to_ascii(string: str) -> str:
    return string.encode("ascii", errors="ignore").decode("ascii")


def run_concurrent(
    func: Callable,
    iterable: Iterable,
    quiet: bool = False,
    how: ConcurrentStrategy = "process",
    chunksize: int = 1,
    *args,
    **kwargs,
) -> list[object]:
    lst = list(iterable)

    if how == "process":
        executor = concurrent.futures.ProcessPoolExecutor(*args, **kwargs)
    elif how == "thread":
        executor = concurrent.futures.ThreadPoolExecutor(*args, **kwargs)
    elif how == "single":
        return [func(i) for i in tqdm.tqdm(lst, disable=quiet)]

    with tqdm.tqdm(total=len(lst), disable=quiet) as pbar:
        with executor as pool:
            res_list = []
            for res in pool.map(func, lst, chunksize=chunksize):
                res_list.append(res)
                pbar.update(1)

    return res_list
