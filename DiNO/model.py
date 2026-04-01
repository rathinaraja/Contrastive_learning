import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import copy

class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        
        # Using a more modern backbone for better feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 768, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.last_dim = 768  # Store last dimension for projection layers

    def forward(self, x):
        return self.encoder(x)

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super(DINOHead, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim)
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)
        
        # Initialize last layer differently as mentioned in DINO paper
        self.last_layer.weight.data.normal_(mean=0.0, std=0.01)
        self.last_layer.bias.data.zero_()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.normalize(self.last_layer(x), dim=-1)
        return x

class DINO(nn.Module):
    def __init__(self, backbone, output_dim=65536, warmup_teacher_temp=0.04, 
                 teacher_temp=0.04, warmup_teacher_temp_epochs=30, 
                 student_temp=0.1, center_momentum=0.9):
        super(DINO, self).__init__()
        
        self.student = backbone
        self.teacher = copy.deepcopy(backbone)
        
        # Get the last_dim from the backbone
        if isinstance(backbone, nn.DataParallel):
            last_dim = backbone.module.last_dim
        else:
            last_dim = backbone.last_dim
        
        # Student and teacher projection heads
        self.student_head = DINOHead(last_dim, output_dim)
        self.teacher_head = DINOHead(last_dim, output_dim)
        
        # Teacher network should be in eval mode
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
            
        self.teacher_head.eval()
        for p in self.teacher_head.parameters():
            p.requires_grad = False
            
        # Temperature parameters
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        
        # Center momentum parameter
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, output_dim))
        
    @torch.no_grad()
    def update_teacher(self, m):
        """Updates teacher model using momentum update"""
        for param_student, param_teacher in zip(self.student.parameters(), 
                                              self.teacher.parameters()):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)
            
        for param_student, param_teacher in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            param_teacher.data.mul_(m).add_((1 - m) * param_student.detach().data)
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        """Update center used for teacher output"""
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = batch_center / len(teacher_output)
        
        self.center = self.center * self.center_momentum + \
                     batch_center * (1 - self.center_momentum)
    
    def forward(self, views):
        """
        Forward pass through DINO.
        views: list of different augmented views of the images
        """
        student_output = [self.student_head(self.student(view)) for view in views]
        student_out = torch.cat(student_output, dim=0)
        
        # Teacher forward passes
        with torch.no_grad():
            teacher_output = [self.teacher_head(self.teacher(view)) for view in views]
            teacher_out = torch.cat(teacher_output, dim=0)
            
        # Update the center
        self.update_center(teacher_out)
        
        # Center and normalize the teacher output
        teacher_out = teacher_out - self.center
        teacher_out = teacher_out.detach()
        
        loss = dino_loss(student_out, teacher_out, self.student_temp, self.teacher_temp)
        
        return loss

def dino_loss(student_output, teacher_output, student_temp, teacher_temp):
    """
    Compute DINO loss
    """
    student_out = student_output / student_temp
    teacher_out = F.softmax((teacher_output - teacher_output.mean(dim=0)) / teacher_temp, dim=-1)
    
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:  # Skip cases where student and teacher operate on same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
            
    total_loss /= n_loss_terms
    return total_loss