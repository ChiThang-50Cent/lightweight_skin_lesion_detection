import torch
import pytorch_lightning as pl

from torchmetrics import Accuracy


class LitModel(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        model: torch.nn.Module,
        lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        acc = self.train_acc(logits, y)
        self.log("train_acc", acc)

        # print(f"train_loss: {loss}, train_acc: {acc}")

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()

        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, sync_dist=True)
        acc = self.val_acc(logits, y)
        self.log("val_acc", acc, sync_dist=True)

        # print(f"val_loss: {loss}, val_acc: {acc}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer)
        return [optimizer], [scheduler]
