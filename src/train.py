import json
import logging
import time
from typing import Optional

import cutils
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data import LOGGER
from utils.paths import FINAL_PATH


def train_model(
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    loss_function,
    data: pl.DataFrame,
    dataset_cls: Dataset,
    outname: str = "model",
    overwrite: bool = False,
    batch_size: int = 32,
    clip_value: Optional[float] = 1,
    class_weights: Optional[torch.Tensor] = None,
    n_epochs: int = 1,
    start_epoch: int = 1,
    **kwargs,
):
    # Prepare logging and directories
    outdir = FINAL_PATH / outname
    outdir.mkdir(exist_ok=overwrite)

    LOGGER.addHandler(logging.FileHandler(outdir / "train.log", mode="w"))

    loss_fn_name = loss_function._get_name()

    with open(outdir / "hyperparameters.json", "w") as f:
        json.dump(
            {
                "batch_size": batch_size,
                "clip_value": clip_value,
                "epochs": n_epochs,
                "class_weights": class_weights
                if class_weights is None
                else class_weights.tolist(),
                "lr_scheduler": "LinearLR",
                "lr_scheduler_args": "default",
                "loss_function": loss_function._get_name(),
                "optimizer": opt.__class__.__name__,
                "model": model.__class__.__name__,
            }
            | kwargs
            | {
                k: v
                for k, v in opt.__dict__["param_groups"][0].items()
                if k != "params"
            }
            | model.init_args,
            f,
            indent=4,
            sort_keys=True,
        )

    # Data

    dataloader = DataLoader(
        dataset_cls(data),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset_cls.pad_collate,
    )
    n_batches = len(dataloader)

    scheduler = optim.lr_scheduler.LinearLR(opt)

    for epoch in range(start_epoch, n_epochs + 1):
        logging.info(f"----epoch {epoch}----")
        total_loss = 0
        total_time = 0
        start = time.time()

        for batch_no, tup in enumerate(dataloader, start=1):
            batch_start = time.time()

            # data may not be evenly divisible, so we need to get the actual batch size
            current_batch_size = (
                tup[0].size()[0] if batch_no == n_batches else batch_size
            )

            inputs = tup[:-1]
            race = tup[-1]
            if loss_fn_name == "BCELoss":
                race = race.unsqueeze(1)

            model.zero_grad(set_to_none=True)
            hidden = model.init_hidden(current_batch_size)
            output, _ = model(*inputs, hidden)

            loss: torch.Tensor = loss_function(output, race)
            loss.backward()
            total_loss += loss

            if clip_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)

            opt.step()

            batch_elapsed = time.time() - batch_start
            total_time += batch_elapsed
            time_left = (total_time / batch_no) * (n_batches - batch_no)
            if batch_no % 100 == 0 or batch_no in (1, n_batches):
                logging.info(
                    f"batch {batch_no} of {n_batches} done in"
                    f" {batch_elapsed:.2f} seconds, time remaining"
                    f" {cutils.display_time(time_left)}, loss {loss.item()}"
                )

        # epoch loss
        logging.info(f"epoch {epoch} loss: {total_loss / n_batches}")

        # update the learning rate
        before_lr = opt.param_groups[0]["lr"]
        scheduler.step()
        after_lr = opt.param_groups[0]["lr"]
        logging.info(f"lr went from {before_lr} to {after_lr}")

        # save the model
        torch.save(model.state_dict(), outdir / f"model{epoch}.pth")
        logging.info(
            f"finished epoch {epoch} in {cutils.display_time(time.time() - start)}"
        )

    torch.save(model.state_dict(), outdir / "model.pth")
