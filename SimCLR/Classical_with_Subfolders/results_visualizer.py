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
import dataset, config
from model import SimCLR, SimCLRNet

class RepresentationAnalyzer:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        
        # Initialize the SimCLR backbone network
        self.model = SimCLRNet().to(device)
        
        # Load the state dictionary
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            # Extract only encoder and projection related weights
            backbone_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.backbone.'):
                    new_key = k.replace('module.backbone.', '')
                    backbone_state_dict[new_key] = v
                elif k.startswith('backbone.'):
                    new_key = k.replace('backbone.', '')
                    backbone_state_dict[new_key] = v
                    
            # Load the state dict
            missing_keys, unexpected_keys = self.model.load_state_dict(backbone_state_dict, strict=False)
            print(f"\nMissing keys: {missing_keys[:5] if missing_keys else 'None'}")
            print(f"Unexpected keys: {unexpected_keys[:5] if unexpected_keys else 'None'}")
                
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        self.model.eval()
        
    def extract_features(self, dataloader):
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch_imgs, batch_paths in dataloader:
                if not isinstance(batch_imgs, torch.Tensor):
                    continue
                    
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    # Extract features - use the encoder output before projection
                    feat = self.model.encoder(batch_imgs)
                    
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

    def reduce_dimensionality(self, features, n_components=2, perplexity=30):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        tsne = TSNE(n_components=n_components, 
                   perplexity=perplexity, 
                   n_iter=1000, 
                   random_state=42)
        reduced_features = tsne.fit_transform(features_scaled)
        return reduced_features

    def cluster_features(self, features, n_clusters=5):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, 
                       random_state=42, 
                       n_init=10)
        clusters = kmeans.fit_predict(features_scaled)
        return clusters, kmeans.inertia_

    def find_optimal_clusters(self, features, max_clusters=10):
        inertias = []
        for k in range(2, max_clusters + 1):
            _, inertia = self.cluster_features(features, n_clusters=k)
            inertias.append(inertia)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(2, max_clusters + 1), inertias, 'bx-')
        plt.xlabel('k (number of clusters)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.savefig('elbow_curve.png')
        plt.close()

    def visualize_clusters(self, reduced_features, clusters, image_paths, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        # Create TSNE plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                            c=clusters, cmap='tab10', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title('SimCLR Learned Representations (t-SNE)', pad=20)
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tsne_clusters.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize sample images from each cluster
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            sample_size = min(25, len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, size=sample_size, replace=False)
            
            nrows = int(np.ceil(np.sqrt(sample_size)))
            ncols = int(np.ceil(sample_size / nrows))
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 15))
            axes = np.array(axes).reshape(-1)
            
            for idx, sample_idx in enumerate(sample_indices):
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
            
            plt.suptitle(f'Sample Images from Cluster {cluster_id}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cluster_{cluster_id}_samples.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()

    def analyze_cluster_distribution(self, clusters, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        sns.countplot(x=clusters, palette='tab10')
        plt.title('Distribution of Images Across Clusters', pad=20)
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        
        # Add value labels on top of each bar
        for i in plt.gca().containers:
            plt.gca().bar_label(i)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config1 = config.Config()
    
    # Initialize transform
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize dataset using the new VisualizationDataset class
    test_dataset = dataset.VisualizationDataset(
        folder_path=config1.folder_path_train,  # Update with your data path
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
        analyzer = RepresentationAnalyzer('model_simclr.pth', device)
        
        print("Extracting features...")
        features, image_paths = analyzer.extract_features(dataloader)
        
        # Find optimal number of clusters
        print("Finding optimal number of clusters...")
        analyzer.find_optimal_clusters(features)
        
        print("Performing dimensionality reduction...")
        reduced_features = analyzer.reduce_dimensionality(features, perplexity=30)
        
        print("Performing clustering...")
        clusters, _ = analyzer.cluster_features(features, n_clusters=5)
        
        print("Creating visualizations...")
        output_dir = 'simclr_visualization_results'
        analyzer.visualize_clusters(reduced_features, clusters, image_paths, output_dir)
        analyzer.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 