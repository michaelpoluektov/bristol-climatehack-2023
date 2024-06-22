import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import RAdam
from model.model import Transformer


class TransformerLightning(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = Transformer(
            embed_size=self.hparams["embed_size"],
            num_layers_enc=self.hparams["num_layers_enc"],
            num_layers_dec=self.hparams["num_layers_dec"],
            num_heads=self.hparams["num_heads"],
            num_groups=self.hparams["num_groups"],
            forward_expansion=self.hparams["forward_expansion"],
            dropout=self.hparams["dropout"],
        )
        self.model = torch.compile(self.model, fullgraph=True, mode="reduce-overhead")

    def forward(self, x, power, time_ftrs, cst_ftrs):  # type: ignore
        return self.model(x, power, time_ftrs, cst_ftrs)

    def step(self, batch, prefix):
        hrv, power, time_ftrs, cst_ftrs, y = batch
        y_hat = self(hrv, power, time_ftrs, cst_ftrs)
        loss = F.l1_loss(y_hat, y)
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):  # type: ignore
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):  # type: ignore
        return self.step(batch, "val")

    def configure_optimizers(self):  # type: ignore
        optim = RAdam(
            self.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
            decoupled_weight_decay=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optim, mode="min", patience=3, factor=0.4, threshold=0.001
        )
        return {
            "optimizer": optim,
            "lr_scheduler": lr_scheduler,
            "monitor": "train_loss",
        }
