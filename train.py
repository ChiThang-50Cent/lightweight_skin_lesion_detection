import torch
import argparse
import torchvision
import albumentations as album
import pytorch_lightning as pl

from lightning_train import LitModel

from albumentations.pytorch import ToTensorV2
from dataset import HAM10000_Dataset, HAM10000_DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def get_transform(size):

    train_transform = album.Compose(
        [
            album.SmallestMaxSize(max_size=size),
            album.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.75
            ),
            album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            album.RandomBrightnessContrast(p=0.5),
            ToTensorV2(),
        ]
    )

    return train_transform

def get_dataset(csv_file, root_dir, train_transform):
    full_dataset = HAM10000_Dataset(csv_file, root_dir, train_transform)

    return full_dataset

def get_data_loader(dataset, batch_size, validation_split=0.1):
    dataloader = HAM10000_DataLoader(
        dataset=dataset, batch_size=batch_size, validation_split=validation_split
    )

    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    return train_loader, val_loader


def choose_model(model_name, num_classes):
    model = None

    if model_name == "efficientnet":
        model = torchvision.models.efficientnet_v2_s(num_classes=num_classes)
    elif model_name == "mobilenetv3":
        model = torchvision.models.mobilenet_v3_large(num_classes=num_classes)
    elif model_name == "shufflenet":
        model = torchvision.models.shufflenet_v2_x2_0(num_classes=num_classes)

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", "-n", type=int, help="Number of classes")
    parser.add_argument(
        "--model_name",
        help="Choose model efficientnet, mobilenetv3, shufflenet or mobilevit",
        default="mobilevit",
    )
    parser.add_argument("--csv_file", help="Path to csv file")
    parser.add_argument("--root_dir", help="Root dir of images folder")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    train_transfrom = get_transform(args.img_size)
    dataset = get_dataset(args.csv_file, args.root_dir, train_transfrom)
    train_loader, val_loader = get_data_loader(dataset, args.batch_size)

    model = choose_model(args.model_name, args.num_classes)
    lit_model = LitModel(model=model, weight=dataset.class_weight, num_classes=args.num_classes, lr=args.lr)

    checkpoint_callback = ModelCheckpoint(
        "./saved_model",
        monitor="val_loss",
        save_top_k=1,
        filename=args.model_name + "_{epoch:02d}_{val_loss:.2f}",
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices="auto",
        logger=CSVLogger("./log", name=f"{args.model_name}_logs"),
        callbacks=[
            EarlyStopping("val_loss", patience=7),
            checkpoint_callback
        ],
    )

    trainer.fit(lit_model, train_loader, val_loader)
