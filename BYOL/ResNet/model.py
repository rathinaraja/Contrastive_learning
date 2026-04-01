import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)  # Load pretrained ResNet-50
        self.encoder = nn.Sequential(*list(resnet.children())[:-3])  # Remove FC layer

    def forward(self, x):
        x = self.encoder(x)  # Output shape: (batch, 2048, 1, 1)
        x = torch.flatten(x, start_dim=1)  # Flatten to (batch, 2048)
        return x

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
    def __init__(self, backbone, image_size, projection_size=256, projection_hidden_size=4096, moving_average_decay=0.99):
        super(BYOL, self).__init__()
        
        self.online_encoder = backbone
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_ema_updater = EMA(moving_average_decay)
        
        # Get the output dimension of the ResNet-50 encoder
        self.encoder_out_dim = 2048  # ResNet-50 outputs a 2048-D feature vector
        
        # Projector
        self.online_projector = MLP(self.encoder_out_dim, projection_hidden_size, projection_size)
        
        # Predictor
        self.online_predictor = MLP(projection_size, projection_hidden_size, projection_size)
        
        # Initialize target network
        self.init_target_network()
    
    def init_target_network(self):
        """Initialize the target network for momentum updating."""
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Disable gradients for target network
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False
    
    @torch.no_grad()
    def update_moving_average(self):
        """Update the target network using exponential moving average (EMA)."""
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = self.target_ema_updater(online.data, target.data)
            
        for online, target in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target.data = self.target_ema_updater(online.data, target.data)
    
    def forward(self, x):
        """Computes BYOL loss"""
        # Online network
        online_features = self.online_encoder(x)
        online_proj = self.online_projector(online_features)
        online_pred = self.online_predictor(online_proj)
        
        # Target network (no gradients)
        with torch.no_grad():
            target_features = self.target_encoder(x)
            target_proj = self.target_projector(target_features)
        
        # Compute BYOL loss
        loss = byol_loss_fn(online_pred, target_proj.detach())
        return loss

class EMA:
    """Exponential Moving Average (EMA) for updating target network"""
    def __init__(self, decay):
        self.decay = decay
    
    def __call__(self, online, target):
        return target * self.decay + (1 - self.decay) * online

def byol_loss_fn(online_pred, target_proj):
    """Computes BYOL loss as cosine similarity"""
    online_pred_norm = F.normalize(online_pred, dim=-1)
    target_proj_norm = F.normalize(target_proj, dim=-1)
    loss = 2 - 2 * (online_pred_norm * target_proj_norm).sum(dim=-1)
    return loss.mean()
