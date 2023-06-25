import concurrent.futures
from collections.abc import Iterable
from typing import Callable, Literal

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
    **kwargs
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
