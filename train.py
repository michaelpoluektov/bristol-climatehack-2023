import os
from pathlib import Path
import torch
import wandb
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import DayDataset
from model.model_lightning import TransformerLightning
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision("medium")


def get_loader(step, is_train=True, num_workers=24):
    return DataLoader(
        DayDataset(step, is_train),
        batch_size=192,
        num_workers=num_workers,
        prefetch_factor=4,
        pin_memory=True,
    )


gpu_id = int(os.getcwd()[-1])
NAME = f"run_640_q_{gpu_id}"
hparams = {
    "embed_size": 640,
    "num_layers_enc": 5,
    "num_layers_dec": 5,
    "num_heads": 16,
    "num_groups": 8,
    "forward_expansion": 3,
    "dropout": 0.1,
    "learning_rate": 2e-5,
    "weight_decay": 5e-6,
    "accumulate_grad_batches": 1,
}


def find_checkpoint(checkpoint_dir):
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")]
    if checkpoints:
        checkpoints.sort(
            key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)),
            reverse=True,
        )
        return os.path.join(checkpoint_dir, checkpoints[0])
    return None


train_loader = get_loader(6, True, 8)
model = TransformerLightning(hparams)
logger = WandbLogger(project="climatehack-bristol", name=NAME)
trainer = pl.Trainer(
    precision="bf16-mixed",
    max_epochs=200,
    logger=logger,
    accelerator="gpu",
    devices=[gpu_id],
    accumulate_grad_batches=hparams["accumulate_grad_batches"],
    limit_train_batches=8192,
    callbacks=[
        ModelCheckpoint(
            monitor="train_loss",
            dirpath=f"checkpoints/{NAME}",
            filename="ckpt_{epoch:02d}_{train_loss:.4f}",
            save_top_k=50,
            mode="min",
            save_last=True,
        ),
    ],
)
logger.log_hyperparams(hparams)
trainer.fit(model, train_loader, ckpt_path=find_checkpoint(f"checkpoints/{NAME}"))
wandb.finish()
