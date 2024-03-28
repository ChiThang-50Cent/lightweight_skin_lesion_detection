import os
import argparse

import torch
import numpy as np
import pandas as pd
import itertools

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from dataframe import Init_dataframe
from dataset import Custom_Dataset
from transforms import get_val_transform
from models import Models

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, path, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.savefig(path)

def save_classification_report(report, path):
    df = pd.DataFrame(report).transpose()
    df.to_csv(path)


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
    parser.add_argument("--ckpt_path", help="Checkpoint path")
    parser.add_argument("--eval_path", help="Save eval path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = Init_dataframe(args.csv_file)

    val_transform = get_val_transform(args.img_size)
    val_set = Custom_Dataset(df.df_val, root_dir=args.root_dir, transform=val_transform)
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = Models(
        model_name=args.model_name,
        num_classes=args.num_classes,
        feature_extract=False,
        pre_trained=False,
    ).get_model()

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    model_weights = checkpoint["state_dict"]

    for key in list(model_weights):
        model_weights[key.replace("model.", "")] = model_weights.pop(key)
        
    model.load_state_dict(model_weights)

    model.to(device)
    model.eval()
    y_label = []
    y_predict = []

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images = images.to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    plot_labels = ['akiec', 'bcc', 'bkl', 'df','mel', 'nv', 'vasc']

    confusion_mtx = confusion_matrix(y_label, y_predict)
    report = classification_report(y_label, y_predict, target_names=plot_labels, output_dict=True)

    plot_confusion_matrix(confusion_mtx, path=os.path.join(args.eval_path, f'{args.model_name}_CM.png'), classes=plot_labels)
    save_classification_report(report, path=os.path.join(args.eval_path, f'{args.model_name}_RP.csv'))

    



