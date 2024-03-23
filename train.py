import torch
import argparse
import torchvision
import albumentations as album
import pytorch_lightning as pl

from lightning_train import LitModel
from mobileViT import mobilevit_xxs

from albumentations.pytorch import ToTensorV2
from dataset import HAM10000_Dataset, HAM10000_DataLoader
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

train_transform = album.Compose(
    [
        album.SmallestMaxSize(max_size=256),
        album.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=90, p=0.75
        ),
        album.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        album.RandomBrightnessContrast(p=0.5),
        ToTensorV2(),
    ]
)


def get_data_loader(csv_file, root_dir, batch_size, validation_split=0.1):
    full_dataset = HAM10000_Dataset(csv_file, root_dir, train_transform)

    dataloader = HAM10000_DataLoader(
        dataset=full_dataset, batch_size=batch_size, validation_split=validation_split
    )

    train_loader = dataloader.get_train_loader()
    val_loader = dataloader.get_val_loader()

    return train_loader, val_loader


def choose_model(model_name):
    model = None

    if model_name == "efficientnet":
        model = torchvision.models.efficientnet_v2_s
    elif model_name == "mobilenetv3":
        model = torchvision.models.mobilenet_v3_large
    elif model_name == "shufflenet":
        model = torchvision.models.shufflenet_v2_x2_0
    elif model_name == "mobilevit":
        model = mobilevit_xxs

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
    args = parser.parse_args()

    model = choose_model(args.model_name)

    if args.model_name == "mobilevit":
        model = model(image_size=(256, 256), num_classes=args.num_classes)
    else:
        model = model(num_classes=args.num_classes)

    print(model)

    lit_model = LitModel(model=model, num_classes=args.num_classes, lr=args.lr)

    train_loader, val_loader = get_data_loader(
        args.csv_file, args.root_dir, args.batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        "./saved_model",
        monitor="val_loss",
        save_top_k=1,
        filename=args + "_{epoch:02d}_{val_loss:.2f}",
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
