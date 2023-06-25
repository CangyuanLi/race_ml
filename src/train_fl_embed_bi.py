import polars as pl

from data import FLEmbedDataset
from models import FLEmbedBiLSTM
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH

HIDDEN_SIZE = 128
EMBEDDING_DIM = 512


def main():
    train = pl.read_parquet(FINAL_PATH / "fl_train.parquet").head(100)

    model = FLEmbedBiLSTM(
        VALID_NAME_CHARS_LEN,
        EMBEDDING_DIM,
        HIDDEN_SIZE,
        len(RACES),
        dropout=0.2,
    ).to(DEVICE)
    train_model(
        model=model,
        data=train,
        dataset_cls=FLEmbedDataset,
        outname="fl_embed_bi_llr_ldr",
        overwrite=True,
        lr=0.00001,
        batch_size=64,
        weight_decay=0.001,
        nesterov=True,
        momentum=0.99,
        clip_value=None,
        n_epochs=15,
        hidden_size=HIDDEN_SIZE,
        embedding_dim=EMBEDDING_DIM,
        dropout=0.2,
    )


if __name__ == "__main__":
    main()
