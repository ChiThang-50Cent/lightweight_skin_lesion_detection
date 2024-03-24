import torch
from torchvision.transforms import v2


def get_train_transform(size):

    transform = v2.Compose(
        [
            v2.ToImage(), 
            v2.Resize((size, size)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(20),
            v2.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.763038, 0.54564667, 0.57004464],
                std=[0.14092727, 0.15261286, 0.1699712],
            ),
        ]
    )

    return transform


def get_val_transform(size):

    val_transform = v2.Compose(
        [
            v2.ToImage(), 
            v2.Resize((size, size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.763038, 0.54564667, 0.57004464],
                std=[0.14092727, 0.15261286, 0.1699712],
            ),
        ]
    )

    return val_transform
