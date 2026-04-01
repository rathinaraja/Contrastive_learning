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
import torch.cuda.amp as amp
import config
import dataset
import model
import losses

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def get_data_transforms(tile_size):
    return transforms.Compose([
        transforms.Resize((tile_size, tile_size)),
        transforms.RandomResizedCrop(tile_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def train_model(model, data_loader, device, config, optimizer):
    train_losses = []
    model.train()
    
    # Initialize mixed precision training
    scaler = amp.GradScaler() if config.use_mixed_precision else None
    
    for epoch in range(config.num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        
        for batch_idx, (img1, img2, folder, part) in enumerate(progress_bar):
            img1, img2 = img1.to(device), img2.to(device)
            
            # Forward pass with mixed precision
            if config.use_mixed_precision:
                with amp.autocast():
                    z1, z2 = model(img1, img2)
                    nt_xent = losses.nt_xent_loss(z1, z2, config.temperature)
                    triplet = losses.triplet_loss(z1, z2)
                    loss = nt_xent + config.triplet_weight * triplet
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                z1, z2 = model(img1, img2)
                nt_xent = losses.nt_xent_loss(z1, z2, config.temperature)
                triplet = losses.triplet_loss(z1, z2)
                loss = nt_xent + config.triplet_weight * triplet
                loss = loss / config.gradient_accumulation_steps
                
                loss.backward()
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Update progress bar with folder info
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'folder': folder[0],
                'part': part[0]
            })
        
        avg_loss = total_loss / len(data_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss:.4f}")
    
    return train_losses

def main():
    config1 = config.Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nPreparing data transforms...")
    transform = get_data_transforms(config1.tile_size)
    
    print("\nLoading dataset...")
    dataset1 = dataset.ContrastiveImageDataset(
        root_folder=config1.folder_path_train,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset1, 
        batch_size=config1.batch_size,
        num_workers=config1.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    print("\nInitializing model...")
    model1 = model.SimCLR(
        temperature=config1.temperature,
        triplet_weight=config1.triplet_weight
    ).to(device)
    
    if torch.cuda.device_count() > 1:
        model1 = nn.DataParallel(model1)
        print(f"Using {torch.cuda.device_count()} GPUs")
    
    optimizer = optim.Adam(model1.parameters(), lr=config1.learning_rate)
    
    print("\nStarting training...")
    losses = train_model(model1, data_loader, device, config1, optimizer)
    
    print("\nSaving results...")
    torch.save(model1.state_dict(), "model_simclr.pth")
    with open(config1.training_loss_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[loss] for loss in losses])
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()