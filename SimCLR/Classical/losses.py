import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Corrected NT-Xent loss implementation
    """
    batch_size = z1.shape[0]
    features = torch.cat([z1, z2], dim=0)
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z1.device)

    features = F.normalize(features, dim=1)
    
    similarity_matrix = torch.matmul(features, features.T)
    
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    labels = labels[~mask].view(2 * batch_size, -1)
    similarity_matrix = similarity_matrix[~mask].view(2 * batch_size, -1)
    
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(2 * batch_size, -1)
    
    # select only the negatives
    negatives = similarity_matrix[~labels.bool()].view(2 * batch_size, -1)
    
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z1.device)
    
    logits = logits / temperature
    return F.cross_entropy(logits, labels)

def triplet_loss(z1, z2, margin=1.0):
    """
    Corrected triplet loss implementation
    """
    batch_size = z1.shape[0]
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Compute pairwise distances
    dist_matrix = torch.cdist(z1, z2, p=2)
    
    # Get positive pairs (diagonal elements)
    positive_pairs = torch.diag(dist_matrix)
    
    # Create mask for negative pairs
    negative_mask = ~torch.eye(batch_size, dtype=bool, device=z1.device)
    
    # Get hardest negative distance for each anchor
    hard_negatives, _ = torch.min(dist_matrix + (~negative_mask).float() * 1e9, dim=1)
    
    # Compute triplet loss
    loss = F.relu(positive_pairs - hard_negatives + margin)
    return loss.mean()