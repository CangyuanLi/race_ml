import polars as pl

from data import FirstLastDataset, pad_collate_fl
from models import FLBiLSTM
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH

DROPOUT = 0.2
HIDDEN_SIZE = 128


def main():
    train = pl.read_parquet(FINAL_PATH / "fl_train.parquet")

    # Low batch size, no dropout
    # model = FirstLastBiLSTM(VALID_NAME_CHARS_LEN, HIDDEN_SIZE, len(RACES)).to(DEVICE)
    # train_model(
    #     model=model,
    #     data=train,
    #     collate_fn=pad_collate_fl,
    #     dataset_cls=FirstLastDataset,
    #     outname="fl_bi_lbs",
    #     overwrite=True,
    #     lr=0.0001,
    #     dropout=0,
    #     hidden_size=HIDDEN_SIZE,
    #     batch_size=32,
    #     weight_decay=0.001,
    #     nesterov=True,
    #     momentum=0.99,
    #     clip_value=None,
    #     n_epochs=10,
    # )

    # Low batch size, dropout of .2
    model = FLBiLSTM(VALID_NAME_CHARS_LEN, HIDDEN_SIZE, len(RACES)).to(DEVICE)
    train_model(
        model=model,
        data=train,
        collate_fn=pad_collate_fl,
        dataset_cls=FirstLastDataset,
        outname="fl_bi_lbs_ldr",
        overwrite=True,
        lr=0.0001,
        dropout=DROPOUT,
        hidden_size=HIDDEN_SIZE,
        batch_size=32,
        weight_decay=0.001,
        nesterov=True,
        momentum=0.99,
        clip_value=None,
        n_epochs=10,
    )

    # Low batch size, dropout of .2, lower hidden size
    model = FLBiLSTM(VALID_NAME_CHARS_LEN, HIDDEN_SIZE, len(RACES)).to(DEVICE)
    train_model(
        model=model,
        data=train,
        collate_fn=pad_collate_fl,
        dataset_cls=FirstLastDataset,
        outname="fl_bi_lbs_ldr",
        overwrite=True,
        lr=0.0001,
        dropout=DROPOUT,
        hidden_size=64,
        batch_size=32,
        weight_decay=0.001,
        nesterov=True,
        momentum=0.99,
        clip_value=None,
        n_epochs=10,
    )


if __name__ == "__main__":
    main()
