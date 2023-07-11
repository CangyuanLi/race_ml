import string

import pandas as pd
import polars as pl
import torch
import torch.nn as nn

from utils.constants import DEVICE
from utils.paths import DIST_PATH


class BayesianBaseModel:
    def __init__(self, df: pl.DataFrame):
        self._df = df
        self._races = ("asian", "black", "hispanic", "white")

        self._PROB_RACE_GIVEN_FIRST_NAME = pl.read_csv(
            DIST_PATH / "prob_race_given_first_name.csv"
        )
        self._PROB_FIRST_NAME_GIVEN_RACE = pl.read_csv(
            DIST_PATH / "prob_first_name_given_race.csv"
        )
        self._PROB_RACE_GIVEN_LAST_NAME = pl.read_csv(
            DIST_PATH / "prob_race_given_last_name.csv"
        )
        self._PROB_ZCTA_GIVEN_RACE = self._normalize_zctas(
            pl.read_csv(DIST_PATH / "prob_zcta_given_race_2010.csv"), "zcta5"
        )
        self._PROB_RACE_GIVEN_ZCTA = self._normalize_zctas(
            pl.read_csv(DIST_PATH / "prob_race_given_zcta_2010.csv"), "zcta5"
        )

    @staticmethod
    def _remove_chars(expr: pl.Expr) -> pl.Expr:
        unwanted_chars = string.digits + string.punctuation + string.whitespace
        for char in unwanted_chars:
            expr = expr.str.replace_all(char, "", literal=True)

        return expr

    def _normalize_names(self, df: pl.DataFrame, col: str | list[str]) -> pl.DataFrame:
        df = self._df.with_columns(
            pl.col(col)
            .pipe(self._remove_chars)
            .str.to_uppercase()
            .str.replace_all(r"\s?J\.*?R\.*\s*?$", "")
            .str.replace_all(r"\s?S\.*?R\.*\s*?$", "")
            .str.replace_all(r"\s?III\s*?$", "")
            .str.replace_all(r"\s?IV\s*?$", "")
        )

        return df

    @staticmethod
    def _normalize_zctas(df: pl.DataFrame, col: str) -> pl.DataFrame:
        return df.with_columns(pl.col(col).cast(str).str.zfill(5))


class BIFSG(BayesianBaseModel):
    def __init__(self, df: pl.DataFrame):
        super().__init__(df)

    def get_probabilities(
        self, first_name: str, last_name: str, zcta: str
    ) -> pl.DataFrame:
        df = self._normalize_names(self._df, [first_name, last_name])
        df = self._normalize_zctas(df, zcta)

        prob_first_name_given_race = df.join(
            self._PROB_FIRST_NAME_GIVEN_RACE,
            left_on=first_name,
            right_on="name",
            how="left",
        ).select(self._races)

        prob_race_given_last_name = df.join(
            self._PROB_RACE_GIVEN_LAST_NAME,
            left_on=last_name,
            right_on="name",
            how="left",
        ).select(self._races)

        prob_zcta_given_race = df.join(
            self._PROB_ZCTA_GIVEN_RACE, left_on=zcta, right_on="zcta5", how="left"
        ).select(self._races)

        bifsg_numer = (
            prob_first_name_given_race
            * prob_race_given_last_name
            * prob_zcta_given_race
        )
        bifsg_denom = bifsg_numer.sum(axis=1)
        bifsg_probs = bifsg_numer / bifsg_denom

        return pl.concat([df, bifsg_probs], how="horizontal")


class BISG(BayesianBaseModel):
    def __init__(self, df: pl.DataFrame):
        super().__init__(df)

    def get_probabilities(self, last_name: str, zcta: str) -> pl.DataFrame:
        df = self._normalize_names(self._df, last_name)
        df = self._normalize_zctas(df, zcta)

        prob_race_given_last_name = df.join(
            self._PROB_RACE_GIVEN_LAST_NAME,
            left_on=last_name,
            right_on="name",
            how="left",
        ).select(self._races)

        prob_zcta_given_race = df.join(
            self._PROB_ZCTA_GIVEN_RACE, left_on=zcta, right_on="zcta5", how="left"
        ).select(self._races)

        bisg_numer = prob_race_given_last_name * prob_zcta_given_race
        bisg_denom = bisg_numer.sum(axis=1)
        bisg_probs = bisg_numer / bisg_denom

        return pl.concat([df, bisg_probs], how="horizontal")


class FLZEmbedBiLSTMBinary(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
        **lstm_kwargs,
    ):
        super(FLZEmbedBiLSTMBinary, self).__init__()

        self.hidden_size = hidden_size

        # set layers to at least 2 if dropout is greater than 0
        # since you need 1 layer for the lstm and 1 dropout layer
        if dropout > 0 and num_layers == 1:
            num_layers = 2
        self.num_layers = num_layers

        self.fn_embedding = nn.Embedding(input_size, embedding_dim)
        self.fn_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.ln_embedding = nn.Embedding(input_size, embedding_dim)
        self.ln_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.h2o = nn.Linear(hidden_size * 4 + 4, output_size)
        self.sigmoid = nn.Sigmoid()

        self.init_args = {
            "input_size": input_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "dropout": dropout,
            "num_layers": self.num_layers,
        } | lstm_kwargs

    def forward(self, fn, ln, pct, hidden):
        embedded_fn = self.fn_embedding(fn)
        _, (h_n, _) = self.fn_lstm(embedded_fn, hidden)
        fn_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        embedded_ln = self.ln_embedding(ln)
        _, (h_n, _) = self.ln_lstm(embedded_ln, hidden)
        ln_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        combined = torch.cat((fn_output, ln_output, pct), dim=1)

        output = self.h2o(combined)
        output: torch.Tensor = self.sigmoid(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
        )


class FLZEmbedBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
        **lstm_kwargs,
    ):
        super(FLZEmbedBiLSTM, self).__init__()

        self.hidden_size = hidden_size

        # set layers to at least 2 if dropout is greater than 0
        # since you need 1 layer for the lstm and 1 dropout layer
        if dropout > 0 and num_layers == 1:
            num_layers = 2
        self.num_layers = num_layers

        self.fn_embedding = nn.Embedding(input_size, embedding_dim)
        self.fn_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.ln_embedding = nn.Embedding(input_size, embedding_dim)
        self.ln_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.h2o = nn.Linear(hidden_size * 4 + 4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.init_args = {
            "input_size": input_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "dropout": dropout,
            "num_layers": self.num_layers,
        } | lstm_kwargs

    def forward(self, fn, ln, pct, hidden):
        embedded_fn = self.fn_embedding(fn)
        _, (h_n, _) = self.fn_lstm(embedded_fn, hidden)
        fn_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        embedded_ln = self.ln_embedding(ln)
        _, (h_n, _) = self.ln_lstm(embedded_ln, hidden)
        ln_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        combined = torch.cat((fn_output, ln_output, pct), dim=1)

        output = self.h2o(combined)
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
        )


class FLZBiLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0
    ):
        super(FLZBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.h2o = nn.Linear(hidden_size * 2 + 4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        name: torch.Tensor,
        pct: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        _, (h_n, _) = self.lstm(name, hidden)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)

        combined = torch.cat((h_n, pct), dim=1)
        output = self.h2o(combined)
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(2, batch_size, self.hidden_size, device=DEVICE),
            torch.zeros(2, batch_size, self.hidden_size, device=DEVICE),
        )


class FLZLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0
    ) -> None:
        super(FLZLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.h2o = nn.Linear(hidden_size + 4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        name: torch.Tensor,
        pct: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        output, (h_n, _) = self.lstm(name, hidden)
        combined = torch.cat((h_n.squeeze(0), pct), dim=1)
        output = self.h2o(combined)
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=DEVICE),
            torch.zeros(1, batch_size, self.hidden_size, device=DEVICE),
        )


class FLEmbedBiLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_dim: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0,
        num_layers: int = 1,
        **lstm_kwargs,
    ):
        super(FLEmbedBiLSTM, self).__init__()

        self.hidden_size = hidden_size

        # set layers to at least 2 if dropout is greater than 0
        # since you need 1 layer for the lstm and 1 dropout layer
        if dropout > 0 and num_layers == 1:
            num_layers = 2
        self.num_layers = num_layers

        self.fn_embedding = nn.Embedding(input_size, embedding_dim)
        self.fn_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.ln_embedding = nn.Embedding(input_size, embedding_dim)
        self.ln_lstm = nn.LSTM(
            embedding_dim,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            num_layers=num_layers,
            **lstm_kwargs,
        )
        self.h2o = nn.Linear(hidden_size * 4, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.init_args = {
            "input_size": input_size,
            "embedding_dim": embedding_dim,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "dropout": dropout,
            "num_layers": self.num_layers,
        } | lstm_kwargs

    def forward(self, fn, ln, hidden):
        embedded_fn = self.fn_embedding(fn)
        _, (h_n, _) = self.fn_lstm(embedded_fn, hidden)
        fn_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        embedded_ln = self.ln_embedding(ln)
        _, (h_n, _) = self.ln_lstm(embedded_ln, hidden)
        ln_output = torch.cat((h_n[-2], h_n[-1]), dim=1)

        combined = torch.cat((fn_output, ln_output), dim=1)

        output = self.h2o(combined)
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
            torch.zeros(
                self.num_layers * 2, batch_size, self.hidden_size, device=DEVICE
            ),
        )


class FLBiLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0
    ):
        super(FLBiLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.h2o = nn.Linear(hidden_size * 2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        name: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        output, (h_n, _) = self.lstm(name, hidden)
        h_n = torch.cat((h_n[-2], h_n[-1]), dim=1)
        output = self.h2o(h_n)
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(2, batch_size, self.hidden_size, device=DEVICE),
            torch.zeros(2, batch_size, self.hidden_size, device=DEVICE),
        )


class FLLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0
    ) -> None:
        super(FLLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(
        self,
        name: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ):
        output, (h_n, _) = self.lstm(name, hidden)
        output = self.h2o(h_n.squeeze(0))
        output: torch.Tensor = self.softmax(output)

        return output, hidden

    def init_hidden(self, batch_size: int):
        return (
            torch.zeros(1, batch_size, self.hidden_size, device=DEVICE),
            torch.zeros(1, batch_size, self.hidden_size, device=DEVICE),
        )
