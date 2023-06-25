import polars as pl

from data import FirstLastZctaDataset, pad_collate_flz, prepare_data_flz
from models import FLZLSTM
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH

HIDDEN_SIZE = 128


def main():
    train = pl.read_parquet(FINAL_PATH / "train_sample.parquet")

    model = FLZLSTM(VALID_NAME_CHARS_LEN, HIDDEN_SIZE, len(RACES)).to(DEVICE)
    train_model(
        model=model,
        data=train,
        collate_fn=pad_collate_flz,
        dataset_cls=FirstLastZctaDataset,
        prepare_data_fn=prepare_data_flz,
        outname="flz_lbs",
        overwrite=True,
        lr=0.00001,
        hidden_size=HIDDEN_SIZE,
        batch_size=64,
        weight_decay=0.001,
        nesterov=True,
        momentum=0.99,
        max_norm=0.9,
        n_epochs=10,
    )


if __name__ == "__main__":
    main()
