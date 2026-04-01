# results_visualizer_autoencoder_projection.py
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
from model import BYOL, MLP
from Auto_encoder import AutoEncoder
import dataset
import config 
from tqdm import tqdm
import shutil
import matplotlib.cm as cm
from matplotlib.lines import Line2D

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

class EncoderWrapper(torch.nn.Module):
    """
    Wrapper class for the encoder part of the AutoEncoder
    to handle the adaptive pooling and flattening needed for BYOL
    """
    def __init__(self, encoder):
        super(EncoderWrapper, self).__init__()
        self.encoder = encoder
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten()
    
    def forward(self, x):
        features = self.encoder(x)
        pooled = self.adaptive_pool(features)
        flattened = self.flatten(pooled)
        return flattened

class FeatureExtractor(torch.nn.Module):
    """
    A simplified feature extractor that combines the encoder and projection head
    """
    def __init__(self, encoder, projector):
        super(FeatureExtractor, self).__init__()
        self.encoder = encoder
        self.projector = projector
    
    def forward(self, x):
        encoded = self.encoder(x)
        projected = self.projector(encoded)
        return projected

class AutoEncoderProjectionAnalyzer:
    def __init__(self, autoencoder_path, byol_model_path, device='cuda'):
        self.device = device
        
        print(f"\nLoading pre-trained AutoEncoder from {autoencoder_path}")
        
        # Initialize the AutoEncoder model and move to the specified device BEFORE loading
        autoencoder = AutoEncoder().to(device)
        
        # Load the weights
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
        autoencoder.eval()  # Set to evaluation mode
        
        # Extract encoder part
        encoder = autoencoder.encoder
        
        # Freeze encoder parameters
        for param in encoder.parameters():
            param.requires_grad = False
            
        # Wrap the encoder with pooling and flattening, move to device
        self.encoder_wrapper = EncoderWrapper(encoder).to(device)
        
        # Test the encoder wrapper
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.tile_size, config.tile_size).to(device)
            test_output = self.encoder_wrapper(dummy_input)
            print(f"Encoder wrapper test successful. Output shape: {test_output.shape}")
        
        # Create projection head
        self.projection_head = MLP(
            input_dim=test_output.shape[1],  # Use the output dim from test
            hidden_dim=4096,
            output_dim=256
        ).to(device)
        
        # Load the trained projector weights from BYOL model if available
        try:
            byol_state_dict = torch.load(byol_model_path, map_location=device)
            
            # Extract projection head weights
            projector_state_dict = {}
            for k, v in byol_state_dict.items():
                if k.startswith('online_projector.'):
                    new_key = k.replace('online_projector.', '')
                    projector_state_dict[new_key] = v
            
            # Load the projector weights
            if projector_state_dict:
                missing, unexpected = self.projection_head.load_state_dict(projector_state_dict, strict=False)
                print(f"Loaded projection head. Missing keys: {missing[:5] if len(missing) > 5 else missing}")
                print(f"Unexpected keys: {unexpected[:5] if len(unexpected) > 5 else unexpected}")
            else:
                print("No projection head weights found in BYOL model, using random initialization")
                
        except Exception as e:
            print(f"Could not load projection head weights: {str(e)}")
            print("Using randomly initialized projection head")
        
        # Create a combined feature extractor
        self.feature_extractor = FeatureExtractor(
            self.encoder_wrapper,
            self.projection_head
        ).to(device)
        
        # Set to evaluation mode
        self.feature_extractor.eval()
        
        print("Model initialization complete")
        
    def extract_features(self, dataloader):
        features = []
        image_paths = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Extracting projection features")
            for batch_imgs, batch_paths in pbar:
                if not isinstance(batch_imgs, torch.Tensor):
                    continue
                    
                # Move data to the same device as the model
                batch_imgs = batch_imgs.to(self.device)
                
                try:
                    # Get features directly from feature extractor
                    projection_feat = self.feature_extractor(batch_imgs)
                    
                    # Move features to CPU for numpy conversion
                    features.append(projection_feat.cpu().numpy())
                    image_paths.extend(batch_paths)
                    
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
        
        if not features:
            raise ValueError("No features were successfully extracted")
            
        features_array = np.vstack(features)
        print(f"\nExtracted projection features shape: {features_array.shape}")
        return features_array, image_paths

    def reduce_dimensionality(self, features, n_components=2):
        print("Performing dimensionality reduction...")
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=min(50, len(features)-1))
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
        plt.savefig('optimal_clusters_analysis_ae_projection.png')
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
        plt.title('Distribution of Clusters (AutoEncoder + Projection)')
        plt.xlabel('Cluster ID')
        plt.ylabel('Number of Images')
        
        # Add count labels on top of each bar
        for i, count in enumerate(counts):
            plt.text(i, count, str(count), ha='center', va='bottom')
            
        plt.savefig(os.path.join(output_dir, 'ae_projection_cluster_distribution.png'))
        plt.close()
        
        # Print cluster statistics
        print("\nCluster distribution statistics:")
        for cluster_id, count in zip(unique_clusters, counts):
            print(f"Cluster {cluster_id}: {count} images")
            
    def visualize_clusters(self, reduced_features, clusters, image_paths, output_dir, n_clusters):
        os.makedirs(output_dir, exist_ok=True)
        
        # Select a colormap with enough distinct colors
        if n_clusters <= 10:
            colormap = plt.get_cmap('tab10', n_clusters)  # Up to 10 distinct colors
        elif n_clusters <= 20:
            colormap = plt.get_cmap('tab20', n_clusters)  # Up to 20 distinct colors
        else:
            colormap = plt.get_cmap('gist_ncar', n_clusters)  # More than 20 distinct colors
    
        # Map each cluster to a color
        cluster_colors = [colormap(i) for i in range(n_clusters)]
        color_map = {cluster_id: cluster_colors[i] for i, cluster_id in enumerate(np.unique(clusters))}
        
        # Assign colors to each data point based on cluster assignment
        point_colors = [color_map[cluster_id] for cluster_id in clusters]
        
        # Plot t-SNE visualization
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                              c=point_colors, alpha=0.6)
        
        # Create legend
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=f'Cluster {i}',
                                  markerfacecolor=colormap(i), markersize=10)
                           for i in range(n_clusters)]
        plt.legend(handles=legend_elements, title="Clusters", loc='best')
        
        plt.title('AutoEncoder + Projection Head Representations (t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(os.path.join(output_dir, 'ae_projection_tsne_clusters.png'))
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
            
            plt.suptitle(f'Sample Images from Cluster {cluster_id} (AutoEncoder + Projection)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'ae_projection_cluster_{cluster_id}_samples.png'))
            plt.close()

def main():
    torch.cuda.empty_cache()  # Free up any GPU memory from previous runs
    
    # Make sure we get a clear device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    # Print device information for debugging
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA device {i} name: {torch.cuda.get_device_name(i)}")
            print(f"CUDA device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    
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
        pin_memory=True if torch.cuda.is_available() else False  # Only use pin_memory with CUDA
    )
    
    print(f"Dataset size: {len(test_dataset)} images")
    
    try:
        print("Loading models...")
        # Path to both models
        autoencoder_path = "model_auto_encoder_reconstruction.pth"  # Change this to your actual autoencoder path
        byol_model_path = "model_byol_pretrained_epoch_10.pth"  # Change this to your actual BYOL model path
        
        analyzer = AutoEncoderProjectionAnalyzer(autoencoder_path, byol_model_path, device)
        
        print("Extracting features from projection head with AutoEncoder backbone...")
        features, image_paths = analyzer.extract_features(dataloader)
        print(f"Extracted projection features shape: {features.shape}")
        
        print("Performing dimensionality reduction...")
        reduced_features = analyzer.reduce_dimensionality(features)

        num_clusters = 25  # Pass None for determining number of clusters
        print("Performing clustering...")
        clusters, n_clusters = analyzer.cluster_features(features, num_clusters)
        
        output_dir = 'ae_projection_visualization_results'
        
        # Save images to cluster folders
        analyzer.save_clustered_images(clusters, image_paths, os.path.join(output_dir, 'clustered_images'))
        
        print("Creating visualizations...")
        analyzer.visualize_clusters(reduced_features, clusters, image_paths, output_dir, n_clusters)
        analyzer.analyze_cluster_distribution(clusters, output_dir)
        
        print(f"Analysis complete! Results saved in: {output_dir}")
        print(f"Final number of clusters used: {n_clusters}")
        print(f"Clustered images can be found in: {os.path.join(output_dir, 'clustered_images')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()  # Print the full stack trace
        raise

if __name__ == "__main__":
    main()