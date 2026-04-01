# training.py
import os
import csv
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import config
import dataset
from model import BYOLNet, BYOL

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\nAvailable GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")    
    else:
        print("\nUsing CPU.")   
    return device

def get_data_augmentations():
    # First set of augmentations (for online network)
    online_transforms = transforms.Compose([ 
        transforms.Resize((config.tile_size, config.tile_size)), 
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            
        # Color augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            
        # Noise and blurring
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Second set of augmentations (for target network)
    target_transforms = transforms.Compose([
        transforms.Resize((config.tile_size, config.tile_size)),
        transforms.RandomRotation(degrees=(-45, 45)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
            
        # Color augmentations
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
            
        # Noise and blurring
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return online_transforms, target_transforms

def prepare_model(device):
    # Initialize the BYOL model and move it to device
    backbone = BYOLNet().to(device)
    if torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)
        print(f"\n{torch.cuda.device_count()} GPUs: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    
    model = BYOL(
        backbone,
        image_size=config.tile_size,
        hidden_layer='encoder',
        projection_size=256,
        projection_hidden_size=4096,
        moving_average_decay=0.99
    ).to(device)
    
    return model

def train_model(model, data_loader, device, num_epochs, optimizer):
    train_losses = []
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        
        for batch in progress_bar:
            batch = batch.to(device)
            
            # Forward pass through BYOL
            loss = model(batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update moving average of target network
            model.update_moving_average()
            
            total_loss += loss.item()
            
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}\n")
    
    return train_losses

def save_training_loss(losses, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        for value in losses:
            writer.writerow([value])
    print(f"\nTraining loss saved to {file_path}")

def save_model(model, file_path):
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), file_path)
    else:
        torch.save(model.state_dict(), file_path)
    print(f"\nModel saved to {file_path}")

def main():
    device = get_device()
    online_transforms, target_transforms = get_data_augmentations()
    
    print("\nLoading training data...")
    dataset1 = dataset.TileDatasetTrain(config.folder_path_train, 
                                     online_transform=online_transforms,
                                     target_transform=target_transforms)
    print("\nNumber of training samples:", len(dataset1))
    data_loader = DataLoader(dataset1, batch_size=config.batch_size_train, 
                           num_workers=config.num_workers, shuffle=True)
    
    print("\nInitializing the model...")
    model = prepare_model(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    
    print("\nStarting training...\n")
    train_losses = train_model(model=model, data_loader=data_loader, 
                             device=device, num_epochs=config.num_epochs, 
                             optimizer=optimizer)
    
    save_training_loss(train_losses, config.training_loss_file)
    save_model(model, "model_byol.pth")
    
    print("\nTraining pipeline completed!")

if __name__ == "__main__":
    main()
