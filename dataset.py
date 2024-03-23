import os
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import LabelEncoder

class HAM10000_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None) -> None:

        self.df, self.label_encoder = self.metadata_process(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, f"{self.df['image_id'][index]}.{'jpg'}")
        img = Image.open(img_path)
        img = np.array(img)
        
        label = self.df['encoded_dx'][index]

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, label 

    def metadata_process(self, csv_file):
        le = LabelEncoder()
        df = pd.read_csv(csv_file)

        le.fit(df['dx'])

        df['encoded_dx'] = le.transform(df['dx'])

        return df, le

class HAM10000_DataLoader():
    def __init__(self, dataset, batch_size=32, validation_split=0.2, shuffer=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffer = shuffer

        self.create_dataloader()

    def create_dataloader(self, random_seed=42):
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))

        if self.shuffer:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]
        
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def get_train_loader(self):
        train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.train_sampler)
        return train_loader
    
    def get_val_loader(self):
        val_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, sampler=self.val_sampler)
        return val_loader



        

    
        