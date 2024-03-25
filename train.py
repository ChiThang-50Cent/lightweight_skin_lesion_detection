import argparse
import pytorch_lightning as pl

from torch.utils.data import DataLoader

from lightning_train import LitModel
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from dataframe import Init_dataframe
from dataset import Custom_Dataset
from transforms import get_val_transform, get_train_transform
from models import Models


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", "-n", type=int, help="Number of classes")
    parser.add_argument(
        "--model_name",
        help="Choose model efficientnet, mobilenetv3, shufflenetv2",
        default="mobilenetv3",
    )
    parser.add_argument("--csv_file", help="Path to csv file")
    parser.add_argument("--root_dir", help="Root dir of images folder")
    parser.add_argument("--pre_trained", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    df = Init_dataframe(args.csv_file)

    train_transform = get_train_transform(args.img_size)
    train_set = Custom_Dataset(
        df.df_train, root_dir=args.root_dir, transform=train_transform
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    val_transform = get_val_transform(args.img_size)
    val_set = Custom_Dataset(df.df_val, root_dir=args.root_dir, transform=val_transform)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = Models(
        model_name=args.model_name,
        num_classes=args.num_classes,
        feature_extract=False,
        pre_trained=args.pre_trained,
    )

    lit_model = LitModel(
        model=model.get_model(),
        num_classes=args.num_classes,
        lr=args.lr,
    )

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
        logger=CSVLogger("./log", name=f"{args.model_name}_logs", version=0),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(lit_model, train_loader, val_loader)
