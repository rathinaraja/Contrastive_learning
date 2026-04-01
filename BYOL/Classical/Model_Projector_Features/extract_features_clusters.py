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
import dataset
import config 
from tqdm import tqdm
import shutil
from model import BYOLNet, MLP  # Import only what we need

class ProjectionFeatureExtractor:
    def __init__(self, model_path, device='cuda', image_size=224):
        self.device = device
        print(f"Using device: {device}")
        
        # Load the complete state dictionary first
        try:
            print(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Print some sample keys to understand the structure
            sample_keys = list(checkpoint.keys())[:5]
            print(f"Sample keys in checkpoint: {sample_keys}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
        # Initialize the encoder (backbone)
        self.encoder = BYOLNet().to(device)
        
        # Calculate encoder output dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, image_size, image_size).to(device)
            encoder_output = self.encoder(dummy_input)
            encoder_dim = encoder_output.shape[1]
            print(f"Encoder output dimension: {encoder_dim}")
        
        # Create projection head with the same architecture as in BYOL
        self.projector = MLP(
            input_dim=encoder_dim,
            hidden_dim=4096,  # Standard projection hidden size in BYOL
            output_dim=256    # Standard projection size in BYOL
        ).to(device)
        
        # Determine key patterns in the checkpoint
        has_module_prefix = any('module.' in k for k in checkpoint.keys())
        has_online_prefix = any(k.startswith('online_') for k in checkpoint.keys())
        module_after_online = any(k.startswith('online_encoder.module.') for k in checkpoint.keys())
        
        print(f"Checkpoint has module prefix: {has_module_prefix}")
        print(f"Checkpoint has online prefix: {has_online_prefix}")
        print(f"Checkpoint has 'module' after 'online_encoder': {module_after_online}")
        
        # Extract encoder and projector weights with proper prefix handling
        encoder_dict = {}
        projector_dict = {}
        
        # Map the keys from checkpoint to our models
        for key, value in checkpoint.items():
            # ENCODER HANDLING - Handle the specific pattern seen in your checkpoint
            if module_after_online and key.startswith('online_encoder.module.encoder.'):
                # Transform 'online_encoder.module.encoder.0.weight' to 'encoder.0.weight'
                parts = key.split('online_encoder.module.encoder.')
                if len(parts) > 1:
                    new_key = 'encoder.' + parts[1]
                    encoder_dict[new_key] = value
                    
            # Previous patterns for backwards compatibility
            elif has_module_prefix and has_online_prefix and key.startswith('module.online_encoder.encoder.'):
                new_key = 'encoder.' + key.split('module.online_encoder.encoder.')[1]
                encoder_dict[new_key] = value
            elif has_module_prefix and key.startswith('module.encoder.'):
                new_key = 'encoder.' + key.split('module.encoder.')[1]
                encoder_dict[new_key] = value
            elif has_online_prefix and key.startswith('online_encoder.encoder.'):
                new_key = 'encoder.' + key.split('online_encoder.encoder.')[1]
                encoder_dict[new_key] = value
            elif key.startswith('encoder.'):
                encoder_dict[key] = value
                
            # PROJECTOR HANDLING - Handle the specific pattern seen in your checkpoint
            if module_after_online and key.startswith('online_projector.module.net.'):
                # Transform 'online_projector.module.net.0.weight' to 'net.0.weight'
                parts = key.split('online_projector.module.net.')
                if len(parts) > 1:
                    new_key = 'net.' + parts[1]
                    projector_dict[new_key] = value
                    
            # Previous patterns for backwards compatibility
            elif has_module_prefix and has_online_prefix and key.startswith('module.online_projector.net.'):
                new_key = 'net.' + key.split('module.online_projector.net.')[1]
                projector_dict[new_key] = value
            elif has_module_prefix and key.startswith('module.projector.net.'):
                new_key = 'net.' + key.split('module.projector.net.')[1]
                projector_dict[new_key] = value
            elif has_online_prefix and key.startswith('online_projector.net.'):
                new_key = 'net.' + key.split('online_projector.net.')[1]
                projector_dict[new_key] = value
            elif key.startswith('projector.net.'):
                new_key = 'net.' + key.split('projector.net.')[1]
                projector_dict[new_key] = value
        
        # Load the encoder weights
        if encoder_dict:
            print(f"Found {len(encoder_dict)} encoder parameters")
            print(f"Sample encoder keys: {list(encoder_dict.keys())[:3]}")
            encoder_load_result = self.encoder.load_state_dict(encoder_dict, strict=False)
            print(f"Encoder load result: {encoder_load_result}")
        else:
            print("WARNING: No encoder parameters found in checkpoint!")
            
        # Load the projector weights
        if projector_dict:
            print(f"Found {len(projector_dict)} projector parameters")
            print(f"Sample projector keys: {list(projector_dict.keys())[:3]}")
            projector_load_result = self.projector.load_state_dict(projector_dict, strict=False)
            print(f"Projector load result: {projector_load_result}")
        else:
            print("WARNING: No projector parameters found in checkpoint!")
            print("This means you're using a randomly initialized projection head,")
            print("which is NOT the trained projection head from your BYOL model!")
                
        # Set models to evaluation mode
        self.encoder.eval()
        self.projector.eval()
        
        print("Feature extractor initialized")
    
    def extract_features(self, dataloader):
        """Extract features from the projection head"""
        features = []
        image_paths = []
        
        with torch.no_grad():
            for batch_imgs, batch_paths in tqdm(dataloader, desc="Extracting projection features"):
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    # Pass through encoder
                    encoder_output = self.encoder(batch_imgs)
                    
                    # Pass through projection head
                    projection_features = self.projector(encoder_output)
                    
                    # Store features and paths
                    features.append(projection_features.cpu().numpy())
                    image_paths.extend(batch_paths)
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("No features were successfully extracted")
            
        features_array = np.vstack(features)
        print(f"Extracted features shape: {features_array.shape}")
        return features_array, image_paths
        
    def save_features_to_file(self, features, image_paths, output_file):
        """
        Save feature vectors along with their corresponding file paths to a file.
        Each feature dimension is saved in its own column.
        
        Args:
            features (numpy.ndarray): Feature vectors
            image_paths (list): List of image file paths
            output_file (str): Path to the output file
        """
        print(f"\nSaving feature vectors to {output_file}...")
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Get feature dimension
        num_samples, feature_dim = features.shape
        
        with open(output_file, 'w') as f:
            # Write header with column names
            header = ["image_path"] + [f"feature_{i+1}" for i in range(feature_dim)]
            f.write(",".join(header) + "\n")
            
            # Write data rows
            for i, (path, feature) in enumerate(zip(image_paths, features)):
                # Format feature vector as comma-separated values
                feature_str = ','.join([f"{val:.6f}" for val in feature])
                # Write row with image path and features
                f.write(f"{path},{feature_str}\n")
                
                # Print progress
                if (i + 1) % 1000 == 0:
                    print(f"  Saved {i + 1}/{num_samples} vectors...")
        
        print(f"Successfully saved all {num_samples} feature vectors to {output_file}")
        print(f"CSV file format: {feature_dim+1} columns (image_path + {feature_dim} features)")
        
    def reduce_dimensionality(self, features, n_components=2):
        print("Performing dimensionality reduction with t-SNE...")
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
        """
        Save images to cluster-specific folders
        """
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

    def cluster_features(self, features, n_clusters):
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
                            c=clusters, cmap='tab20', alpha=0.6)
        plt.colorbar(scatter)
        plt.title('Projection Head Features (t-SNE)')
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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data transformation
    transform = transforms.Compose([
        transforms.Resize((config.tile_size, config.tile_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    test_dataset = dataset.TileDatasetTest(
        folder_path=config.folder_path_train,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(test_dataset)} images")
    
    try:
        # Initialize feature extractor with the model path
        model_path = 'model_byol.pth'  # Adjust if needed
        extractor = ProjectionFeatureExtractor(model_path, device, image_size=config.tile_size)
        
        # Extract projection features
        features, image_paths = extractor.extract_features(dataloader)
        
        # Save features to file
        features_output_file = 'projection_features.csv'
        extractor.save_features_to_file(features, image_paths, features_output_file)
        
        # Reduce dimensionality for visualization
        reduced_features = extractor.reduce_dimensionality(features)
        
        # Cluster the features
        n_clusters = 20  # Set to None for automatic determination
        clusters, n_clusters = extractor.cluster_features(features, n_clusters)
        
        # Create output directory
        output_dir = 'projection_features_results'
        
        # Save images to cluster folders
        extractor.save_clustered_images(clusters, image_paths, os.path.join(output_dir, 'clustered_images'))
        
        # Create visualizations
        extractor.visualize_clusters(reduced_features, clusters, image_paths, output_dir)
        extractor.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        print(f"Final number of clusters used: {n_clusters}")
        print(f"Clustered images can be found in: {os.path.join(output_dir, 'clustered_images')}")
        print(f"Feature vectors saved to: {features_output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()