import polars as pl
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

    # model 6
    model = FLZEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=512,
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
        outname="flz_embed_bi_6",
        overwrite=True,
        batch_size=256,
        clip_value=None,
        n_epochs=15,
        dropout=0.2,
        sample="flz",
    )


if __name__ == "__main__":
    main()
