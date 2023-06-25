import logging

import polars as pl
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset

from utils.constants import (
    DEVICE,
    RACES_DICT,
    VALID_NAME_CHARS_DICT,
    VALID_NAME_CHARS_LEN,
)
from utils.utils import encode_race, prepare_name

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger()


class FLEmbedDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.data = self.prepare_data(df)

    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
    ) -> list[tuple[str, str, str]]:
        res = [
            (fn, ln, r)
            for fn, ln, r in zip(
                df.get_column("first_name").to_list(),
                df.get_column("last_name").to_list(),
                df.get_column("race_ethnicity").to_list(),
                strict=True,
            )
        ]

        return res

    @staticmethod
    def encode_name(name: str) -> torch.Tensor:
        return torch.tensor(
            [VALID_NAME_CHARS_DICT[char] for char in name], device=DEVICE
        )

    @staticmethod
    def pad_collate(batch):
        fns = []
        fn_lengths = []
        lns = []
        ln_lengths = []
        races = []
        for fn, ln, race in batch:
            fns.append(fn)
            fn_lengths.append(fn.size()[0])
            lns.append(ln)
            ln_lengths.append(ln.size()[0])
            races.append(race)

        fns = pad_sequence(fns, batch_first=True)
        lns = pad_sequence(lns, batch_first=True)
        races = torch.hstack(races)

        return fns, lns, races

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (
            first_name,
            last_name,
            race,
        ) = self.data[index]

        fn = self.encode_name(first_name)
        ln = self.encode_name(last_name)
        race = encode_race(race, RACES_DICT, DEVICE)

        return fn, ln, race


class FLZEmbedDataset(Dataset):
    def __init__(self, df: pl.DataFrame):
        self.data = self.prepare_data(df)

    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
    ) -> list[tuple[str, str, float, float, float, float, str]]:
        res = [
            (fn, ln, pa, pb, ph, pw, r)
            for fn, ln, pa, pb, ph, pw, r in zip(
                df.get_column("first_name").to_list(),
                df.get_column("last_name").to_list(),
                df.get_column("pct_asian_zcta").to_list(),
                df.get_column("pct_black_zcta").to_list(),
                df.get_column("pct_hispanic_zcta").to_list(),
                df.get_column("pct_white_zcta").to_list(),
                df.get_column("race_ethnicity").to_list(),
                strict=True,
            )
        ]

        return res

    @staticmethod
    def encode_name(name: str) -> torch.Tensor:
        return torch.tensor(
            [VALID_NAME_CHARS_DICT[char] for char in name], device=DEVICE
        )

    @staticmethod
    def pad_collate(batch):
        fns = []
        fn_lengths = []
        lns = []
        ln_lengths = []
        pcts = []
        races = []
        for fn, ln, pct, race in batch:
            fns.append(fn)
            fn_lengths.append(fn.size()[0])
            lns.append(ln)
            ln_lengths.append(ln.size()[0])
            pcts.append(pct)
            races.append(race)

        fns = pad_sequence(fns, batch_first=True)
        lns = pad_sequence(lns, batch_first=True)
        pcts = torch.stack(pcts)
        races = torch.hstack(races)

        return fns, lns, pcts, races

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (
            first_name,
            last_name,
            asian_pct,
            black_pct,
            hisp_pct,
            white_pct,
            race,
        ) = self.data[index]

        fn = self.encode_name(first_name)
        ln = self.encode_name(last_name)
        pct = torch.tensor((asian_pct, black_pct, hisp_pct, white_pct), device=DEVICE)
        race = encode_race(race, RACES_DICT, DEVICE)

        return fn, ln, pct, race


class FirstLastZctaDataset(Dataset):
    def __init__(self, data: pl.DataFrame):
        self.data = self.prepare_data(data)

    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
    ) -> list[tuple[str, str, float, float, float, float, str]]:
        res = [
            (fn, ln, pa, pb, ph, pw, r)
            for fn, ln, pa, pb, ph, pw, r in zip(
                df.get_column("first_name").to_list(),
                df.get_column("last_name").to_list(),
                df.get_column("pct_asian_zcta").to_list(),
                df.get_column("pct_black_zcta").to_list(),
                df.get_column("pct_hispanic_zcta").to_list(),
                df.get_column("pct_white_zcta").to_list(),
                df.get_column("race_ethnicity").to_list(),
                strict=True,
            )
        ]

        return res

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (
            first_name,
            last_name,
            asian_pct,
            black_pct,
            hisp_pct,
            white_pct,
            race,
        ) = self.data[index]

        name = prepare_name(
            first_name, last_name, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN, DEVICE
        )
        pct = torch.tensor((asian_pct, black_pct, hisp_pct, white_pct), device=DEVICE)
        race = encode_race(race, RACES_DICT, DEVICE)

        return name, pct, race


def pad_collate_flz(batch):
    name_lengths = []
    names = []
    pcts = []
    races = []
    for name, pct, race in batch:
        names.append(name)
        name_lengths.append(name.size()[0])
        pcts.append(pct)
        races.append(race)

    names = pad_sequence(names, batch_first=True, padding_value=0)
    pcts = torch.stack(pcts)
    races = torch.hstack(races)

    names = pack_padded_sequence(
        names, batch_first=True, lengths=name_lengths, enforce_sorted=False
    )

    return names, pcts, races


class FirstLastDataset(Dataset):
    def __init__(self, data: pl.DataFrame):
        self.data = self.prepare_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        (
            first_name,
            last_name,
            race,
        ) = self.data[index]

        name = prepare_name(
            first_name, last_name, VALID_NAME_CHARS_DICT, VALID_NAME_CHARS_LEN, DEVICE
        )
        race = encode_race(race, RACES_DICT, DEVICE)

        return name, race

    @staticmethod
    def prepare_data(
        df: pl.DataFrame,
    ) -> list[tuple[str, str, str]]:
        res = [
            (fn, ln, r)
            for fn, ln, r in zip(
                df.get_column("first_name").to_list(),
                df.get_column("last_name").to_list(),
                df.get_column("race_ethnicity").to_list(),
                strict=True,
            )
        ]

        return res


def pad_collate_fl(batch):
    name_lengths = []
    names = []
    races = []
    for name, race in batch:
        names.append(name)
        name_lengths.append(name.size()[0])
        races.append(race)

    names = pad_sequence(names, batch_first=True, padding_value=0)
    races = torch.hstack(races)

    names = pack_padded_sequence(
        names, batch_first=True, lengths=name_lengths, enforce_sorted=False
    )

    return names, races
