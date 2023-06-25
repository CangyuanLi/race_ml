import polars as pl
import torch
import torch.optim as optim

from data import FLZEmbedDataset
from models import FLZEmbedBiLSTM
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH

HIDDEN_SIZE = 128
EMBEDDING_DIM = 512


def main():
    train = pl.read_parquet(FINAL_PATH / "flz_train.parquet")

    # model 1
    model = FLZEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=len(RACES),
        dropout=0.2,
        num_layers=2,
    ).to(DEVICE)
    opt = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.001, momentum=0.9)
    train_model(
        model=model,
        opt=opt,
        data=train,
        dataset_cls=FLZEmbedDataset,
        outname="flz_embed_bi_1",
        overwrite=True,
        batch_size=64,
        weight_decay=0.001,
        clip_value=None,
        n_epochs=15,
        dropout=0.2,
    )

    # model 2
    model = FLZEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=len(RACES),
        dropout=0.2,
        num_layers=2,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    train_model(
        model=model,
        opt=opt,
        data=train,
        dataset_cls=FLZEmbedDataset,
        outname="flz_embed_bi_2",
        overwrite=True,
        batch_size=512,
        weight_decay=0.001,
        clip_value=None,
        n_epochs=15,
        dropout=0.2,
    )

    # model 3
    model = FLZEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=len(RACES),
        dropout=0.2,
        num_layers=4,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    train_model(
        model=model,
        opt=opt,
        data=train,
        dataset_cls=FLZEmbedDataset,
        outname="flz_embed_bi_3",
        overwrite=True,
        batch_size=512,
        weight_decay=0.001,
        clip_value=None,
        n_epochs=15,
        dropout=0.2,
    )

    # model 4
    # to penalize false positive black and asian, weight getting white wrong more
    # heavily. This is because the majority of false positive asian and black are
    # predicting whites to be asian or black.
    model = FLZEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        output_size=len(RACES),
        dropout=0.2,
        num_layers=2,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    train_model(
        model=model,
        opt=opt,
        data=train,
        dataset_cls=FLZEmbedDataset,
        outname="flz_embed_bi_4",
        overwrite=True,
        batch_size=512,
        weight_decay=0.001,
        clip_value=None,
        n_epochs=15,
        class_weights=torch.tensor([1, 1, 1, 2]),
        dropout=0.2,
    )


if __name__ == "__main__":
    main()
