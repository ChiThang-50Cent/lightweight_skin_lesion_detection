import os
import cv2
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset


class Custom_Dataset(Dataset):
    def __init__(self, df, root_dir, transform=None) -> None:
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.kernel = cv2.getStructuringElement(1,(17,17))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        img_path = os.path.join(self.root_dir, f"{self.df['image_id'][index]}.{'jpg'}")
        img = Image.open(img_path)
         
        img = np.array(img)
        grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY ) #1 Convert the original image to grayscale
        blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, self.kernel) #2 Perform the blackHat filtering on the grayscale image to find the hair countours
        _,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY) # intensify the hair countours in preparation for the inpainting algorithm
        dst = cv2.inpaint(img,thresh2,1,cv2.INPAINT_TELEA) # inpaint the original image depending on the mask
        
        img_new = Image.fromarray(dst)

        label = torch.tensor(int(self.df["encoded_dx"][index]))

        if self.transform:
            img_new = self.transform(img_new)

        return img_new, label
