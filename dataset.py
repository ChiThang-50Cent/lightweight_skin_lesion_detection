import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

class HAM10000_Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None) -> None:
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, f"{self.df['image_id'][index]}.{'jpg'}")
        img = Image.open(img_path)
        img = np.array(img)

        label = self.df["encoded_dx"][index]

        if self.transform:
            img = self.transform(img)

        return img, label
