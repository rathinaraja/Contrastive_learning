import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimCLRNet(nn.Module):
    def __init__(self):
        super(SimCLRNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)  # L2 normalize embeddings

class SimCLR(nn.Module):
    def __init__(self, temperature=0.5, triplet_weight=1.0):
        super(SimCLR, self).__init__()
        self.backbone = SimCLRNet()
        self.temperature = temperature
        self.triplet_weight = triplet_weight
        
    def forward(self, x1, x2):
        z1 = self.backbone(x1)
        z2 = self.backbone(x2)
        return z1, z2
    
    def get_embedding(self, x):
        return self.backbone(x)