import polars as pl
import torch

from data import FirstLastZctaDataset, pad_collate_flz
from models import FLZBiLSTM
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH

HIDDEN_SIZE = 128


def main():
    train = pl.read_parquet(FINAL_PATH / "flz_train.parquet")

    torch.autograd.detect_anomaly(True)
    model = FLZBiLSTM(VALID_NAME_CHARS_LEN, HIDDEN_SIZE, len(RACES)).to(DEVICE)
    train_model(
        model=model,
        data=train,
        collate_fn=pad_collate_flz,
        dataset_cls=FirstLastZctaDataset,
        outname="flz_bi_lbs",
        overwrite=True,
        lr=0.0001,
        hidden_size=HIDDEN_SIZE,
        batch_size=32,
        weight_decay=0.001,
        nesterov=True,
        momentum=0.99,
        clip_value=None,
        n_epochs=10,
    )


if __name__ == "__main__":
    main()
