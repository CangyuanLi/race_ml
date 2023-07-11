import polars as pl
import torch
import torch.optim as optim

from data import FLZEmbedBinaryDataset, FLZEmbedDataset
from models import FLZEmbedBiLSTM, FLZEmbedBiLSTMBinary
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH


def main():
    train = pl.read_parquet(FINAL_PATH / "flz_asian_train.parquet").sample(
        fraction=1, shuffle=True
    )

    # model 11
    model = FLZEmbedBiLSTMBinary(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=512,
        hidden_size=128,
        output_size=1,
        dropout=0.2,
        num_layers=4,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.001)
    loss_function = torch.nn.BCELoss()

    train_model(
        model=model,
        opt=opt,
        loss_function=loss_function,
        data=train,
        dataset_cls=FLZEmbedBinaryDataset,
        outname="flz_embed_bi_11asian",
        overwrite=True,
        batch_size=256,
        clip_value=None,
        n_epochs=20,
        dropout=0.2,
    )


if __name__ == "__main__":
    main()
