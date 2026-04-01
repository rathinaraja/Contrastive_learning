# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset 

class TileDatasetTrain(Dataset):
    def __init__(self, folder_path, online_transform=None, target_transform=None):
        self.folder_path = folder_path
        self.image_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(folder_path) 
            for f in files if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.online_transform = online_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.online_transform and self.target_transform:
            online_img = self.online_transform(img)
            target_img = self.target_transform(img)
            return online_img
        
        return img

class TileDatasetTest(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(folder_path) 
            for f in files if f.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path  # Return both the transformed image and the path