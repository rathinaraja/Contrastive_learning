import os
import csv
import warnings
import numpy as np
import math
warnings.filterwarnings("ignore")
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import config
from dataset import MultiCropDataset
from model import DinoNet, DINO
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

torch.cuda.empty_cache()  # Clears unused memory
torch.backends.cudnn.benchmark = True  # Optimizes GPU memory usage
torch.backends.cudnn.enabled = True

def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"\nUsing {torch.cuda.device_count()} GPUs: "
              f"{[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    else:
        print("\nUsing CPU")
    return device

def get_dino_augmentations(img_size):
    # Following DINO paper's augmentation strategy
    flip_and_color_jitter = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])
    
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # First global crop
    global_transform1 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0), 
                                   interpolation=transforms.InterpolationMode.BICUBIC),
        flip_and_color_jitter,
        transforms.GaussianBlur(23, (1.0, 2.0)),
        normalize,
    ])
    
    # Second global crop
    global_transform2 = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.4, 1.0),
                                   interpolation=transforms.InterpolationMode.BICUBIC),
        flip_and_color_jitter,
        transforms.RandomApply([transforms.GaussianBlur(23, (1.0, 2.0))], p=0.1),
        transforms.RandomSolarize(170, p=0.2),
        normalize,
    ])
    
    # Local crops
    local_transform = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.05, 0.4),
                                   interpolation=transforms.InterpolationMode.BICUBIC),
        flip_and_color_jitter,
        transforms.RandomApply([transforms.GaussianBlur(23, (1.0, 2.0))], p=0.5),
        normalize,
    ])
    
    return global_transform1, global_transform2, local_transform

def prepare_model(device):
    backbone = DinoNet().to(device)
    if torch.cuda.device_count() > 1:
        backbone = nn.DataParallel(backbone)
    
    model = DINO(
        backbone=backbone,
        output_dim=65536,  # DINO paper default
        warmup_teacher_temp=0.04,
        teacher_temp=0.04,
        warmup_teacher_temp_epochs=30,
        student_temp=0.1,
    ).to(device)
    
    return model

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0):
    """
    Create cosine schedule with warmup.
    """
    total_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)
    else:
        warmup_schedule = np.array([])

    iters = np.arange(total_iters - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train_model(model, data_loader, device, num_epochs, optimizer):
    train_losses = []
    model.train()
    
    # Calculate total iterations for the schedulers
    niter_per_ep = len(data_loader)
    
    # Momentum schedule from 0.996 to 1 with cosine scheduling
    momentum_schedule = cosine_scheduler(
        base_value=0.996,
        final_value=1,
        epochs=num_epochs,
        niter_per_ep=niter_per_ep
    )
    
    # Temperature schedule for teacher from 0.04 to 0.07
    temp_schedule = cosine_scheduler(
        base_value=0.04,
        final_value=0.07,
        epochs=num_epochs,
        niter_per_ep=niter_per_ep
    )
    
    global_step = 0
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch")
        
        for batch_idx, crops in enumerate(progress_bar):
            # Move all crops to device
            crops = [crop.to(device) for crop in crops]
            
            # Update momentum
            model.update_teacher(momentum_schedule[global_step])
            
            # Update teacher temperature
            model.teacher_temp = temp_schedule[global_step]
            
            optimizer.zero_grad()
            
            # Enable automatic mixed precision (AMP)
            with torch.cuda.amp.autocast():
                loss = model(crops)
            
            # Scale the loss and backward pass
            scaler.scale(loss).backward()
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()  # Update scaler for next iteration
            
            total_loss += loss.item()
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(data_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}\n")
    
    return train_losses

def save_training_loss(losses, file_path):
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Loss'])  # Header
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
    
    # Get augmentations
    global_transform1, global_transform2, local_transform = get_dino_augmentations(config.tile_size)
    
    print("\nLoading training data...")
    dataset_train = MultiCropDataset(
        folder_path=config.folder_path_train,
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        n_local_crops=6,
        global_transform=global_transform1,
        local_transform=local_transform
    )
    
    print(f"\nNumber of training samples: {len(dataset_train)}")
    
    data_loader = DataLoader(
        dataset_train,
        batch_size=config.batch_size_train,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=True
    )
    
    print("\nInitializing the DINO model...")
    model = prepare_model(device)
    
    # DINO uses a specific optimizer setup with weight decay
    param_groups = [
        {'params': [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=1e-3,
        weight_decay=0.04,
        betas=(0.9, 0.999)
    )
    
    print("\nStarting DINO training...\n")
    train_losses = train_model(
        model=model,
        data_loader=data_loader,
        device=device,
        num_epochs=config.num_epochs,
        optimizer=optimizer
    )
    
    save_training_loss(train_losses, config.training_loss_file)
    save_model(model, "model_dino.pth")
    
    print("\nDINO training pipeline completed!")

if __name__ == "__main__":
    main()