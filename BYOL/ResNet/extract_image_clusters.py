# results_visualizer.py
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from torchvision import transforms, models
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
from kneed import KneeLocator 
import dataset
import config 
from tqdm import tqdm
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
    
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
                if 'online_encoder' in k:
                    # Remove all prefixes to get base encoder keys
                    new_key = k.split('encoder.')[-1]
                    encoder_state_dict[new_key] = v
                elif 'encoder.' in k:
                    new_key = k.split('encoder.')[-1]
                    encoder_state_dict[new_key] = v
            
            # Debug info
            print("Loading model weights...")
            print("Available keys in loaded state dict:", list(state_dict.keys())[:5])
            print("Mapped keys for encoder:", list(encoder_state_dict.keys())[:5])
            
            # Load the state dict into the encoder
            missing_keys, unexpected_keys = self.model.encoder.load_state_dict(encoder_state_dict, strict=False)
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
            # Add progress bar for feature extraction
            pbar = tqdm(dataloader, desc="Extracting features")
            for batch_imgs, batch_paths in pbar:
                if not isinstance(batch_imgs, torch.Tensor):
                    continue
                    
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    feat = self.model(batch_imgs)
                    
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
        print("Performing dimensionality reduction...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        tsne = TSNE(n_components=n_components, random_state=42)
        reduced_features = tsne.fit_transform(features_scaled)
        return reduced_features

    def find_optimal_clusters(self, features, max_clusters=20):
        print("\nFinding optimal number of clusters...")
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_clusters + 1)
        
        # Add progress bar for cluster analysis
        pbar = tqdm(k_range, desc="Testing cluster sizes")
        for k in pbar:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(features_scaled)
            
            inertias.append(kmeans.inertia_)
            
            labels = kmeans.labels_
            silhouette_avg = silhouette_score(features_scaled, labels)
            silhouette_scores.append(silhouette_avg)
            
            pbar.set_postfix({'silhouette': f'{silhouette_avg:.3f}'})
        
        kl = KneeLocator(
            list(k_range), 
            inertias, 
            curve='convex', 
            direction='decreasing'
        )
        elbow_k = kl.elbow
        
        best_silhouette_k = k_range[np.argmax(silhouette_scores)]
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(k_range, inertias, 'bx-')
        plt.vlines(elbow_k, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed')
        plt.xlabel('k')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        
        plt.subplot(1, 2, 2)
        plt.plot(k_range, silhouette_scores, 'bx-')
        plt.vlines(best_silhouette_k, plt.ylim()[0], plt.ylim()[1], colors='r', linestyles='dashed')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        
        plt.tight_layout()
        plt.savefig('optimal_clusters_analysis.png')
        plt.close()
        
        print(f"Elbow method suggests {elbow_k} clusters")
        print(f"Silhouette analysis suggests {best_silhouette_k} clusters")
        
        optimal_k = round((elbow_k + best_silhouette_k) / 2)
        print(f"Using {optimal_k} clusters (average of both methods)")
        
        return optimal_k

    def save_clustered_images(self, clusters, image_paths, output_dir): 
        print("\nSaving images to cluster folders...")
        
        # Create main output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create cluster folders
        cluster_folders = {}
        for cluster_id in np.unique(clusters):
            cluster_folder = os.path.join(output_dir, f'cluster_{cluster_id}')
            os.makedirs(cluster_folder, exist_ok=True)
            cluster_folders[cluster_id] = cluster_folder
        
        # Copy images to respective cluster folders with progress bar
        for img_idx, (cluster_id, img_path) in enumerate(tqdm(zip(clusters, image_paths), 
                                                            total=len(image_paths),
                                                            desc="Copying images to clusters")):
            try:
                # Get destination folder
                dest_folder = cluster_folders[cluster_id]
                
                # Create destination path
                img_filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_folder, img_filename)
                
                # Copy the image
                shutil.copy2(img_path, dest_path)
                
            except Exception as e:
                print(f"Error copying image {img_path}: {str(e)}")
                continue
        
        print(f"Images saved to cluster folders in {output_dir}")

    def cluster_features(self, features, n_clusters=None):
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features)
        
        print("\nPerforming clustering...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        return clusters, n_clusters

    def analyze_cluster_distribution(self, clusters, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        print("\nAnalyzing cluster distribution...")
        # Get cluster counts
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        
        # Plot cluster distribution
        plt.figure(figsize=(10, 6))
        sns.barplot(x=unique_clusters, y=counts)
        plt.title('Distribution of Clusters')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        
        # Add count labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
            
        plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'))
        plt.close()
        
        # Print cluster statistics
        print("\nCluster distribution statistics:")
        for cluster_id, count in zip(unique_clusters, counts):
            print(f"Cluster {cluster_id}: {count} images")
            
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
        
        # Visualize sample images from each cluster with progress bar
        for cluster_id in tqdm(np.unique(clusters), desc="Generating cluster visualizations"):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
        print("Loading model...")
        analyzer = RepresentationAnalyzer('model_byol.pth', device)
        
        print("Extracting features...")
        features, image_paths = analyzer.extract_features(dataloader)
        print(f"Extracted features shape: {features.shape}")
        
        print("Performing dimensionality reduction...")
        reduced_features = analyzer.reduce_dimensionality(features)
        
        print("Performing clustering...")
        clusters, n_clusters = analyzer.cluster_features(features)
        
        output_dir = 'visualization_results'
        
        # Save images to cluster folders
        analyzer.save_clustered_images(clusters, image_paths, os.path.join(output_dir, 'clustered_images'))
        
        print("Creating visualizations...")
        analyzer.visualize_clusters(reduced_features, clusters, image_paths, output_dir)
        analyzer.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        print(f"Final number of clusters used: {n_clusters}")
        print(f"Clustered images can be found in: {os.path.join(output_dir, 'clustered_images')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()