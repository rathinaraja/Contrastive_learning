import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List, Optional

class ContrastiveImageDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[transforms.Compose] = None):
        """
        Dataset for loading images from hierarchical folder structure.
        
        Args:
            root_folder: Root directory containing multiple folders
            transform: Transformations to be applied to images
        """
        self.root_folder = root_folder
        self.transform = transform
        self.image_files = []
        self.subfolder_names = ['Informative_Part1']
        
        # Collect all image files from the specified structure
        self._collect_images()
        
    def _collect_images(self):
        """Collects images from the hierarchical folder structure."""
        for folder in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder)
            
            # Skip if not a directory
            if not os.path.isdir(folder_path):
                continue
                
            # Look for Informative_Part* folders
            for part_folder in self.subfolder_names:
                part_path = os.path.join(folder_path, part_folder)
                
                if not os.path.exists(part_path):
                    continue
                    
                # Collect images from this part folder
                for root, _, files in os.walk(part_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            full_path = os.path.join(root, file)
                            self.image_files.append({
                                'path': full_path,
                                'folder': folder,
                                'part': part_folder
                            })
                            
        print(f"Found {len(self.image_files)} images in {len(os.listdir(self.root_folder))} folders")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_info = self.image_files[idx]
        img_path = img_info['path']
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a different image if one fails to load
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)
            return img1, img2, img_info['folder'], img_info['part']
        
        return img, img, img_info['folder'], img_info['part']

class VisualizationDataset(Dataset):
    """Dataset class for visualization with hierarchical folder structure"""
    def __init__(self, root_folder: str, transform: Optional[transforms.Compose] = None):
        self.root_folder = root_folder
        self.transform = transform
        self.image_files = []
        self.subfolder_names = ['Informative_Part1', 'Informative_Part2', 'Informative_Part3']
        
        self._collect_images()
        
    def _collect_images(self):
        for folder in os.listdir(self.root_folder):
            folder_path = os.path.join(self.root_folder, folder)
            
            if not os.path.isdir(folder_path):
                continue
                
            for part_folder in self.subfolder_names:
                part_path = os.path.join(folder_path, part_folder)
                
                if not os.path.exists(part_path):
                    continue
                    
                for root, _, files in os.walk(part_path):
                    for file in files:
                        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                            full_path = os.path.join(root, file)
                            self.image_files.append({
                                'path': full_path,
                                'folder': folder,
                                'part': part_folder
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
            return self.__getitem__((idx + 1) % len(self))
        
        if self.transform:
            img_tensor = self.transform(img)
            return img_tensor, img_path, img_info['folder'], img_info['part']
        
        return img, img_path, img_info['folder'], img_info['part']