# model.py with enhanced architecture
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

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

class EnhancedFeatureAdapter(nn.Module):
    """
    Adapter module to transform encoder features to be more suitable for contrastive learning
    """
    def __init__(self, input_dim, output_dim=512):
        super(EnhancedFeatureAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.adapter(x)

class BYOL(nn.Module):
    def __init__(self, backbone, image_size, hidden_layer='encoder',
                 projection_size=256, projection_hidden_size=4096,
                 moving_average_decay=0.99, use_adapter=True):
        super(BYOL, self).__init__()
        
        self.online_encoder = backbone
        self.target_encoder = None
        self.target_ema_updater = EMA(moving_average_decay)
        self.use_adapter = use_adapter
        
        # Get the output dimension of the encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size)
            output = self.online_encoder(dummy_input)
            self.encoder_out_dim = output.shape[1]
        
        # Feature adapter to transform features from autoencoder to better suit contrastive learning
        if self.use_adapter:
            self.online_adapter = EnhancedFeatureAdapter(self.encoder_out_dim)
            adapter_output_dim = 512  # Output dimension of the adapter
        else:
            adapter_output_dim = self.encoder_out_dim
        
        # Projector
        self.online_projector = MLP(adapter_output_dim, 
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
        
        if self.use_adapter:
            self.target_adapter = copy.deepcopy(self.online_adapter)
            # Disable gradient computation for target adapter
            for p in self.target_adapter.parameters():
                p.requires_grad = False
        
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
        
        if self.use_adapter:
            for online, target in zip(self.online_adapter.parameters(),
                                    self.target_adapter.parameters()):
                target.data = self.target_ema_updater(online.data, target.data)
            
        for online, target in zip(self.online_projector.parameters(),
                                self.target_projector.parameters()):
            target.data = self.target_ema_updater(online.data, target.data)
    
    def forward(self, x):
        # Online network forward pass
        encoded = self.online_encoder(x)
        
        if self.use_adapter:
            adapted = self.online_adapter(encoded)
            online_proj = self.online_projector(adapted)
        else:
            online_proj = self.online_projector(encoded)
            
        online_pred = self.online_predictor(online_proj)
        
        # Target network forward pass
        with torch.no_grad():
            target_encoded = self.target_encoder(x)
            
            if self.use_adapter:
                target_adapted = self.target_adapter(target_encoded)
                target_proj = self.target_projector(target_adapted)
            else:
                target_proj = self.target_projector(target_encoded)
        
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