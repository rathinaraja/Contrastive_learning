# results_visualizer.py
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
from model import BYOLNet, BYOL
import dataset
import config 

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

class RepresentationAnalyzer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Initialize only the backbone network
        self.model = BYOLNet().to(device)
        
        # Load the state dictionary
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)  # Added weights_only=True
            
            # Extract only encoder-related weights with proper key mapping
            encoder_state_dict = {}
            for k, v in state_dict.items():
                # Handle the case with additional 'module' in the path
                if k.startswith('online_encoder.module.encoder.'):
                    new_key = k.replace('online_encoder.module.encoder.', 'encoder.')
                    encoder_state_dict[new_key] = v
                elif k.startswith('online_encoder.encoder.'):
                    new_key = k.replace('online_encoder.encoder.', 'encoder.')
                    encoder_state_dict[new_key] = v
                elif k.startswith('encoder.'):
                    encoder_state_dict[k] = v
                    
            # Print the available keys for debugging
            print("Available keys in loaded state dict:", list(state_dict.keys())[:5])
            print("Mapped keys for encoder:", list(encoder_state_dict.keys())[:5])
            print("Expected keys in model:", list(self.model.state_dict().keys())[:5])
            
            # Try to load the mapped state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(encoder_state_dict, strict=False)
            print(f"\nMissing keys: {missing_keys[:5]}...")
            print(f"Unexpected keys: {unexpected_keys[:5]}...")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.model.eval()
        
        # Define transform for inference
        self.transform = transforms.Compose([
            transforms.Resize((config.tile_size, config.tile_size)),
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
                    
                # Move batch to the same device as model
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    # Extract features using the encoder
                    feat = self.model(batch_imgs)
                    
                    # If features are 4D (B, C, H, W), convert to 2D (B, C*H*W)
                    if len(feat.shape) > 2:
                        feat = feat.view(feat.size(0), -1)
                        
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

    def reduce_dimensionality(self, features, n_components=2):
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use t-SNE for dimensionality reduction
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_features = tsne.fit_transform(features_scaled)
        return reduced_features

    def cluster_features(self, features, n_clusters): 
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
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
            
            # Take up to 25 sample images from this cluster
            sample_size = min(25, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
            
            # Create a grid of images
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
            
            # Turn off any unused subplots
            for idx in range(sample_size, len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(f'Sample Images from Cluster {cluster_id}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_samples.png'))
            plt.close()

    def analyze_cluster_distribution(self, clusters, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot cluster distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x=clusters)
        plt.title('Distribution of Clusters')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
        plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize dataset and dataloader
    transform = transforms.Compose([
        transforms.Resize((config.tile_size, config.tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
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
        # Initialize analyzer
        print("Loading model...")
        analyzer = RepresentationAnalyzer('model_byol.pth', device)
        
        # Extract features
        print("Extracting features...")
        features, image_paths = analyzer.extract_features(dataloader)
        print(f"Extracted features shape: {features.shape}")
        
        # Reduce dimensionality
        print("Performing dimensionality reduction...")
        reduced_features = analyzer.reduce_dimensionality(features)
        
        # Perform clustering with optimal number of clusters
        print("Performing clustering...")
        clusters, n_clusters = analyzer.cluster_features(features, 25)  # Let it find optimal clusters
        
        # Visualize results
        print("Creating visualizations...")
        output_dir = 'visualization_results_user_defined_clusters'
        analyzer.visualize_clusters(reduced_features, clusters, image_paths, output_dir)
        analyzer.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        print(f"Final number of clusters used: {n_clusters}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()