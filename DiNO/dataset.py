import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Callable

class MultiCropDataset(Dataset):
    def __init__(self, 
                 folder_path: str,
                 global_crops_scale: tuple = (0.4, 1.0),
                 local_crops_scale: tuple = (0.05, 0.4),
                 n_local_crops: int = 6,
                 global_transform: Callable = None,
                 local_transform: Callable = None):
        """
        Dataset returning multiple crops of the images as proposed in DINO.
        
        Args:
            folder_path: Path to image folder
            global_crops_scale: Scale range for global crops
            local_crops_scale: Scale range for local crops
            n_local_crops: Number of local crops
            global_transform: Transformations for global crops
            local_transform: Transformations for local crops
        """
        self.folder_path = folder_path
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_local_crops = n_local_crops
        
        self.image_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(folder_path) 
            for f in files if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            )
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        all_crops = []
        
        # Two global crops
        if self.global_transform is not None:
            all_crops.extend([self.global_transform(img) for _ in range(2)])
        
        # Local crops
        if self.local_transform is not None:
            all_crops.extend([self.local_transform(img) for _ in range(self.n_local_crops)])
        
        return all_crops

class TileDatasetEval(Dataset):
    """Dataset class for evaluation/inference"""
    def __init__(self, folder_path: str, transform: Callable = None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            os.path.join(root, f) 
            for root, _, files in os.walk(folder_path) 
            for f in files if f.lower().endswith(
                (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
            )
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path