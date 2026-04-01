import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchvision import transforms
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
import dataset, config
# Import the BYOL-specific components
from torchvision import models
from torch import nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
    
    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return x

class RepresentationAnalyzer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Initialize the feature extractor
        self.model = ResNetFeatureExtractor().to(device)
        
        # Load the state dictionary
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Extract only encoder-related weights
            encoder_state_dict = {}
            for k, v in state_dict.items():
                # Handle different key patterns in the state dict
                if k.startswith('online_encoder.module.encoder.'):
                    new_key = k.replace('online_encoder.module.encoder.', '')
                    encoder_state_dict[new_key] = v
                elif k.startswith('online_encoder.encoder.'):
                    new_key = k.replace('online_encoder.encoder.', '')
                    encoder_state_dict[new_key] = v
                elif k.startswith('encoder.'):
                    new_key = k.replace('encoder.', '')
                    encoder_state_dict[new_key] = v
            
            # Print debugging information
            print("Available keys in loaded state dict:", list(state_dict.keys())[:5])
            print("Mapped keys for encoder:", list(encoder_state_dict.keys())[:5])
            print("Expected keys in model:", list(self.model.encoder.state_dict().keys())[:5])
            
            # Load the state dict into the encoder
            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(f"\nMissing keys: {missing_keys[:5]}...")
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.model.eval()
        
        # Define transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Standard ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, dataloader):
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch_imgs, batch_paths in dataloader:
                if not isinstance(batch_imgs, torch.Tensor):
                    continue
                
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    # Extract features using the encoder
                    feat = self.model(batch_imgs)
                    features.append(feat.cpu().numpy())
                    image_paths.extend(batch_paths)
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("No features were successfully extracted")
            
        features_array = np.vstack(features)
        print(f"\nExtracted features shape: {features_array.shape}")
        return features_array, image_paths

    # The rest of the methods remain unchanged
    def reduce_dimensionality(self, features, n_components=2):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_features = tsne.fit_transform(features_scaled)
        return reduced_features

    def cluster_features(self, features, n_clusters):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        return clusters, n_clusters

    def visualize_clusters(self, reduced_features, clusters, image_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=clusters, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Learned Representations (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(os.path.join(output_dir, 'tsne_clusters.png'))
        plt.close()
        
        # Visualize sample images from each cluster
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            sample_size = min(25, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
            
            nrows = int(np.ceil(np.sqrt(sample_size)))
            ncols = int(np.ceil(sample_size / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            axes = axes.ravel()
            
            for idx, sample_idx in enumerate(sample_indices):
                if idx < len(axes):
                    try:
                        img_path = image_paths[sample_idx]
                        img = Image.open(img_path).convert('RGB')
                        axes[idx].imshow(img)
                        axes[idx].axis('off')
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
                        continue
            
            for idx in range(sample_size, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Sample Images from Cluster {cluster_id}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_samples.png'))
            plt.close()

    def analyze_cluster_distribution(self, clusters, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        sns.countplot(x=clusters)
        plt.title('Distribution of Clusters')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
        plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Standard ResNet input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # You'll need to implement your own dataset class or modify the existing one
    test_dataset = dataset.TileDatasetTest(
        folder_path=config.folder_path_train,
        transform=transform
    )
    
    dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(test_dataset)} images")
    
    try:
        print("Loading model...")
        analyzer = RepresentationAnalyzer('model_byol.pth', device)
        
        print("Extracting features...")
        features, image_paths = analyzer.extract_features(dataloader)
        print(f"Extracted features shape: {features.shape}")
        
        print("Performing dimensionality reduction...")
        reduced_features = analyzer.reduce_dimensionality(features)
        
        print("Performing clustering...")
        n_clusters = 50
        clusters, n_clusters = analyzer.cluster_features(features, n_clusters)
        
        print("Creating visualizations...")
        output_dir = 'visualization_results'
        analyzer.visualize_clusters(reduced_features, clusters, image_paths, output_dir)
        analyzer.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        print(f"Final number of clusters used: {n_clusters}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()