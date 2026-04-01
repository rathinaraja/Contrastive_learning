# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

class BYOLNet(nn.Module):
    def __init__(self):
        super(BYOLNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, input_size, input_size)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: (64, input_size/2, input_size/2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: (128, input_size/4, input_size/4)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Output: (256, input_size/8, input_size/8)
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # Output: (512, input_size/16, input_size/16) 131k 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),  # Output: (512, input_size/32, input_size/32) 
            nn.BatchNorm2d(512),
            nn.LeakyReLU(), 
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def forward(self, x):
        return self.encoder(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, image_size, hidden_layer='encoder',
                 projection_size=256, projection_hidden_size=4096,
                 moving_average_decay=0.99):
        super(BYOL, self).__init__()
        
        self.online_encoder = backbone
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        
        # Get the output dimension of the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            output = self.online_encoder(dummy_input)
            self.encoder_out_dim = output.shape[1]
        
        # Projector
        self.online_projector = MLP(self.encoder_out_dim, 
                                  projection_hidden_size,
                                  projection_size)
        
        # Predictor
        self.online_predictor = MLP(projection_size,
                                  projection_hidden_size,
                                  projection_size)
        
        # Initialize target network
        self.init_target_network()
    
    def init_target_network(self):
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Disable gradient computation for target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update_moving_average(self):
        # Update target network parameters using EMA
        for online, target in zip(self.online_encoder.parameters(),
                                self.target_encoder.parameters()):
            target.data = self.target_ema_updater(online.data, target.data)
            
        for online, target in zip(self.online_projector.parameters(),
                                self.target_projector.parameters()):
            target.data = self.target_ema_updater(online.data, target.data)
    
    def forward(self, x):
        # Online network forward pass
        online_proj = self.online_projector(self.online_encoder(x))
        online_pred = self.online_predictor(online_proj)
        
        # Target network forward pass
        with torch.no_grad():
            target_proj = self.target_projector(self.target_encoder(x))
        
        # Compute loss
        loss = byol_loss_fn(online_pred, target_proj.detach())
        
        return loss

class EMA:
    def __init__(self, decay):
        self.decay = decay
    
    def __call__(self, online, target):
        return target * self.decay + (1 - self.decay) * online

def byol_loss_fn(online_pred, target_proj):
    online_pred_norm = F.normalize(online_pred, dim=-1)
    target_proj_norm = F.normalize(target_proj, dim=-1)
    
    loss = 2 - 2 * (online_pred_norm * target_proj_norm).sum(dim=-1)
    
    return loss.mean()
