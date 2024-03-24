import os

from PIL import Image
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None) -> None:
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, f"{self.df['image_id'][index]}.{'jpg'}")
        img = Image.open(img_path)

        label = self.df["encoded_dx"][index]

        if self.transform:
            img = self.transform(img)

        return img, label
