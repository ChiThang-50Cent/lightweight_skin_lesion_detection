import torch
import torchvision
import pytorch_lightning as pl

from torchmetrics import Accuracy


class LitModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes,
        weight: torch.Tensor | None = None,
        lr=1e-3,
        pre_trained=True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = self.choose_model(model_name, pre_trained)
        self.train_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.val_acc = Accuracy(
            task="multiclass", num_classes=num_classes, average="weighted"
        )
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        self.lr = lr

        if pre_trained:
            self.model.eval()

            for param in self.model.parameters():
                param.requires_grad = False

        self.model.classifier[-1] = torch.nn.Linear(
            in_features=self.model.classifier[-1].in_features, out_features=num_classes
        )

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
        return optimizer

    def choose_model(self, model_name, pre_trained) -> torch.nn.Module:
        model = None

        if model_name == "efficientnet":
            model = torchvision.models.efficientnet_v2_s(pre_trained=pre_trained)
        elif model_name == "mobilenetv3":
            model = torchvision.models.mobilenet_v3_large(pre_trained=pre_trained)
        elif model_name == "shufflenet":
            model = torchvision.models.shufflenet_v2_x2_0(pre_trained=pre_trained)

        return model
