import os
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Callable
 
class TileDatasetTrain(Dataset):
    def __init__(self, 
                 root_folder: str,
                 subfolder_names: List[str] = ['Informative_Part1'],
                 online_transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Dataset for training with subfolder structure.
        
        Args:
            root_folder: Root directory containing multiple folders, each with subfolders
            subfolder_names: List of subfolder names to look for in each folder
            online_transform: Transform to be applied to input images
            target_transform: Transform to be applied to target images
        """
        self.root_folder = root_folder
        self.subfolder_names = subfolder_names
        self.online_transform = online_transform
        self.target_transform = target_transform
        
        # Collect all image files from the specified structure
        self.image_files = []
        self._collect_images()
        
    def _collect_images(self):
        """Collect images from all subfolders in the hierarchical structure."""
        # Walk through the root folder
        for folder in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
                
            # Look for specified subfolders
            for subfolder in self.subfolder_names:
                subfolder_path = os.path.join(folder_path, subfolder)
                
                # Skip if subfolder doesn't exist
                if not os.path.exists(subfolder_path):
                    continue
                    
                # Collect images from this subfolder
                for root, _, files in os.walk(subfolder_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            full_path = os.path.join(root, file)
                            self.image_files.append({
                                'path': full_path,
                                'folder': folder,
                                'subfolder': subfolder
                            })

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_info = self.image_files[idx]
        img_path = img_info['path']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a different image if one fails to load
            return self.__getitem__((idx + 1) % len(self))
        
        if self.online_transform and self.target_transform:
            online_img = self.online_transform(img)
            target_img = self.target_transform(img)
            return online_img, img_info['folder'], img_info['subfolder']
        
        return img, img_info['folder'], img_info['subfolder']

class TileDatasetTest(Dataset):
    def __init__(self, 
                 root_folder: str,
                 subfolder_names: List[str] = ['part1'],
                 transform: Optional[Callable] = None):
        """
        Dataset for testing with subfolder structure.
        
        Args:
            root_folder: Root directory containing multiple folders, each with subfolders
            subfolder_names: List of subfolder names to look for in each folder
            transform: Transform to be applied to images
        """
        self.root_folder = root_folder
        self.subfolder_names = subfolder_names
        self.transform = transform
        
        # Collect all image files from the specified structure
        self.image_files = []
        self._collect_images()
        
    def _collect_images(self):
        """Collect images from all subfolders in the hierarchical structure."""
        # Walk through the root folder
        for folder in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
                
            # Look for specified subfolders
            for subfolder in self.subfolder_names:
                subfolder_path = os.path.join(folder_path, subfolder)
                
                # Skip if subfolder doesn't exist
                if not os.path.exists(subfolder_path):
                    continue
                    
                # Collect images from this subfolder
                for root, _, files in os.walk(subfolder_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            full_path = os.path.join(root, file)
                            self.image_files.append({
                                'path': full_path,
                                'folder': folder,
                                'subfolder': subfolder
                            })

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_info = self.image_files[idx]
        img_path = img_info['path']
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a different image if one fails to load
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img = self.transform(img)
            
        return img, img_path, img_info['folder'], img_info['subfolder']