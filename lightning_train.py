import pytorch_lightning as pl
import torch

from torchmetrics import Accuracy

class LitModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        acc = self.train_acc(logits, y)
        self.log("train_acc", acc)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss)
        acc = self.val_acc(logits, y)
        self.log("val_acc", acc) 

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9)
        return optimizer
