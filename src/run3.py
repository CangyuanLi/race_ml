import polars as pl
import torch
import torch.optim as optim

from data import FLEmbedDataset, FLZEmbedBinaryDataset, FLZEmbedDataset
from models import FLEmbedBiLSTM, FLZEmbedBiLSTM, FLZEmbedBiLSTMBinary
from train import train_model
from utils.constants import DEVICE, RACES, VALID_NAME_CHARS_LEN
from utils.paths import FINAL_PATH


def main():
    train = pl.read_parquet(FINAL_PATH / "fl_dups_train.parquet").sample(
        fraction=1, shuffle=True
    )

    model = FLEmbedBiLSTM(
        input_size=VALID_NAME_CHARS_LEN,
        embedding_dim=256,
        hidden_size=128,
        output_size=len(RACES),
        dropout=0.2,
        num_layers=4,
    ).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
    loss_function = torch.nn.NLLLoss()

    train_model(
        model=model,
        opt=opt,
        loss_function=loss_function,
        data=train,
        dataset_cls=FLEmbedDataset,
        outname="fl_embed_bi_2",
        overwrite=True,
        batch_size=512,
        clip_value=None,
        n_epochs=20,
    )


if __name__ == "__main__":
    main()
