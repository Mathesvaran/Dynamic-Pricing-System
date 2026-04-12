import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def apply_kmeans_clustering(data_path: str, n_clusters: int = 3):
    """
    Applies K-Means clustering on the dynamic pricing dataset.
    """
    print(f"====== K-Means Clustering ======")
    print(f"Loading data from {data_path}...")
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}'. Please check the file path.")
        return
        
    df = pd.read_csv(data_path)
    
    # -------------------------------
    # 1. Select Features & Clean Data
    # -------------------------------
    print("Selecting features for clustering...")
    cluster_features = ["base_price", "demand", "rating", "reviews", "stock"]
    
    # Ensure all required features are present
    missing_features = [feat for feat in cluster_features if feat not in df.columns]
    if missing_features:
        print(f"Error: Missing features in dataset: {missing_features}")
        return
        
    # Handle any potential missing values by filling with median
    df_features = df[cluster_features].copy()
    if df_features.isnull().values.any():
        print("Handling missing values using median imputation...")
        df_features = df_features.fillna(df_features.median())

    # -------------------------------
    # 2. Scale Data
    # -------------------------------
    print("Scaling feature data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # -------------------------------
    # 3. Apply KMeans
    # -------------------------------
    print(f"Applying K-Means clustering with k={n_clusters}...")
    # Explicitly setting n_init=10 to suppress FutureWarnings in newer scikit-learn versions
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)
    
    print("\n[ Cluster Distribution ]")
    print(df["cluster"].value_counts().sort_index())
    
    # Calculate average values for each feature per cluster to interpret them
    print("\n[ Average Feature Values per Cluster ]")
    cluster_summary = df.groupby("cluster")[cluster_features].mean().round(2)
    print(cluster_summary)

    # -------------------------------
    # 4. Reduce to 2D using PCA
    # -------------------------------
    print("\nReducing dimensions to 2D using PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # -------------------------------
    # 5. Plot Clusters
    # -------------------------------
    print("Generating cluster plot...")
    plt.figure(figsize=(10, 7))
    
    # Define a color map for better separation
    try:
        # For matplotlib 3.6+
        colors = plt.colormaps['viridis'].resampled(n_clusters)
    except AttributeError:
        # Fallback for older matplotlib versions
        colors = plt.cm.get_cmap('viridis', n_clusters)
    
    for i in range(n_clusters):
        # Scatter points for the current cluster
        plt.scatter(
            X_pca[df["cluster"] == i, 0],
            X_pca[df["cluster"] == i, 1],
            label=f"Cluster {i}",
            c=[colors(i)],
            alpha=0.6,
            edgecolors='w',
            s=70
        )

    # Plot cluster centers in PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_pca[:, 0], centers_pca[:, 1],
        marker='X', s=250, c='red', label='Centroids', edgecolor='black', zorder=5
    )

    # Aesthetically pleasing titles and labels
    plt.title("K-Means Clustering of Products (PCA Reduced)", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
    plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
    plt.legend(title="Groups", loc='upper right', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot in op_image folder inside the script's directory
    output_dir = os.path.join(os.path.dirname(__file__), "op_image")
    os.makedirs(output_dir, exist_ok=True)
    output_plot = os.path.join(output_dir, "kmeans_clusters.png")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved successfully as '{output_plot}'")
    print("================================")

    # Display plot to screen (as requested)
    plt.show()

if __name__ == "__main__":
    dataset_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dynamic_pricing_dataset.csv")
    apply_kmeans_clustering(dataset_file, n_clusters=3)
