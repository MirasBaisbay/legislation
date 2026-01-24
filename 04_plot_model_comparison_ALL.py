import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib.colors import ListedColormap

# ================= CONFIGURATION =================
DATA_DIR = Path("results/viz_data")
ANALYSIS_DIR = Path("results/final_analysis")
FIGURES_DIR = Path("results/figures/model_comparison").resolve()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATASET_LABEL = "All Documents (n=167)"
CLUSTER_THRESHOLD = 8 

BASE_MODELS = ["Qwen3-8B", "BGE-M3", "Jina-v3", "OpenAI"]

PRETTY_NAMES = {
    "OpenAI": "OpenAI-text-embedding-3-large",
    "Qwen3-8B": "Qwen3-8B",
    "BGE-M3": "BGE-M3",
    "Jina-v3": "Jina-v3",
    "Ensemble": "Ensemble"
}

def generate_ensemble_data():
    """Combines all 4 models (Equal Weight)."""
    print(f"Generating Ensemble Data...")
    combined_df = None
    count = 0
    for model in BASE_MODELS:
        file_path = DATA_DIR / f"heatmap_data_{model}.csv"
        if not file_path.exists(): continue
        df = pd.read_csv(file_path).set_index("Country")
        if combined_df is None: combined_df = df
        else: combined_df = combined_df.add(df, fill_value=0)
        count += 1
    
    if count == 0: return False
    ensemble_df = combined_df / count
    ensemble_df.reset_index(inplace=True)
    ensemble_df.to_csv(DATA_DIR / "heatmap_data_Ensemble.csv", index=False)
    return True

def plot_heatmap_clustered(model_name):
    """
    Generates a structured Heatmap sorted by Cluster Intensity (High -> Low).
    """
    file_path = DATA_DIR / f"heatmap_data_{model_name}.csv"
    if not file_path.exists(): return

    df = pd.read_csv(file_path).set_index("Country")
    
    # 1. Sort Columns (Keywords)
    col_means = df.mean(axis=0)
    sorted_keywords = col_means.sort_values(ascending=False).index.tolist()
    df = df[sorted_keywords]
    
    # 2. Perform Clustering (Ward)
    df_filled = df.fillna(0)
    Z = linkage(df_filled, method='ward')
    
    # Get initial clusters
    cluster_labels = fcluster(Z, t=CLUSTER_THRESHOLD, criterion='distance')
    
    # Fallback if too few clusters
    if len(np.unique(cluster_labels)) < 2:
        cluster_labels = fcluster(Z, t=4, criterion='maxclust')

    # 3. RE-SORT LOGIC (Crucial for "Bird's Eye" view)
    # We want to sort Clusters by their Mean Score (Highest block on top)
    df_temp = df.copy()
    df_temp['cluster_id'] = cluster_labels
    df_temp['mean_score'] = df_temp.drop(columns=['cluster_id']).mean(axis=1)
    
    # Calculate average score of each cluster
    cluster_stats = df_temp.groupby('cluster_id')['mean_score'].mean().sort_values(ascending=False)
    
    # Create the new sorted order
    final_order = []
    ordered_cluster_ids = []
    
    for rank, (cid, score) in enumerate(cluster_stats.items()):
        # Get countries in this cluster
        subset = df_temp[df_temp['cluster_id'] == cid]
        # Sort countries INSIDE the cluster by their score (High -> Low)
        subset_sorted = subset.sort_values('mean_score', ascending=False)
        final_order.extend(subset_sorted.index.tolist())
        # Store cluster ID for the sidebar (Ranked 1, 2, 3...)
        ordered_cluster_ids.extend([rank + 1] * len(subset))

    # Apply sorting
    df_sorted = df.loc[final_order]
    
    # 4. Setup Layout (Sidebar + Heatmap)
    plot_height = max(12, len(df_sorted) * 0.22) # Dynamic height
    
    # Use constrained_layout to fix the warning
    fig, (ax_side, ax_main) = plt.subplots(
        1, 2, 
        figsize=(15, plot_height), 
        gridspec_kw={'width_ratios': [0.5, 20], 'wspace': 0.05},
        constrained_layout=True 
    )
    
    # 5. Plot Sidebar (Clusters)
    # Create a distinct color palette for clusters (using Tab10/Set2 for distinctness)
    unique_ranks = len(cluster_stats)
    palette = sns.color_palette("tab10", unique_ranks)
    
    # Create matrix for sidebar (flipped to vertical)
    sidebar_data = np.array(ordered_cluster_ids).reshape(-1, 1)
    
    sns.heatmap(
        sidebar_data,
        ax=ax_side,
        cmap=ListedColormap(palette),
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )
    
    # 6. Plot Main Heatmap
    cmap = "YlOrRd"
    display_name = PRETTY_NAMES.get(model_name, model_name)
    
    sns.heatmap(
        df_sorted, 
        ax=ax_main,
        cmap=cmap, 
        annot=False, 
        linewidths=0, 
        cbar_kws={'label': 'Similarity Score', 'shrink': 0.5, 'location': 'top'}
    )
    
    # 7. Add THICK White Dividing Lines
    # Find indices where cluster changes
    arr_ids = np.array(ordered_cluster_ids)
    change_indices = np.where(arr_ids[:-1] != arr_ids[1:])[0] + 1
    
    for idx in change_indices:
        ax_main.axhline(idx, color='white', linewidth=4)
        ax_side.axhline(idx, color='white', linewidth=4)
        
    # 8. Add Cluster Number Labels to Sidebar
    # Find center of each cluster block
    for rank in range(1, unique_ranks + 1):
        # Find indices for this rank
        indices = np.where(arr_ids == rank)[0]
        if len(indices) > 0:
            center = indices.mean() + 0.5 # +0.5 to center in pixel
            ax_side.text(0.5, center, str(rank), ha='center', va='center', 
                         color='white', fontweight='bold', fontsize=12)

    # 9. Styling
    ax_side.set_title("Grp", fontsize=10, fontweight='bold', rotation=90)
    ax_main.set_title(f"Water Policy Intensity: {display_name}\nSorted by Cluster Intensity (High → Low) | {DATASET_LABEL}", 
                      fontsize=16, fontweight='bold', pad=10)
    ax_main.set_xlabel("Policy Keywords", fontsize=14)
    ax_main.set_ylabel("", fontsize=14)
    
    # Tick formatting
    ax_main.tick_params(axis='y', labelsize=8)
    ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=45, ha="right", fontsize=11)

    # Save
    save_path = FIGURES_DIR / f"Heatmap_Clustered_{model_name}.png"
    plt.savefig(str(save_path), dpi=300)
    plt.close('all')
    print(f"  ✓ Generated Structured Heatmap for {model_name}")

def plot_integrated_relevance(model_name):
    file_path = DATA_DIR / f"heatmap_data_{model_name}.csv"
    if not file_path.exists(): return

    df = pd.read_csv(file_path)
    numeric_df = df.drop(columns=["Country"], errors="ignore")
    keyword_means = numeric_df.mean(axis=0).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    
    if model_name == "Ensemble":
        colors = plt.cm.Greens(np.linspace(0.4, 1.0, len(keyword_means)))
    else:
        norm = plt.Normalize(keyword_means.min(), keyword_means.max())
        colors = plt.cm.viridis(norm(keyword_means.values))
    
    bars = plt.barh(keyword_means.index, keyword_means.values, color=colors)
    plt.gca().invert_yaxis()
    
    max_val = keyword_means.max()
    plt.xlim(0, max_val * 1.15) 
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                 f'{width:.4f}', va='center', fontsize=10, fontweight='bold')

    display_name = PRETTY_NAMES.get(model_name, model_name)
    plt.title(f"Global Keyword Relevance: {display_name}\n(Mean Cosine Similarity) | {DATASET_LABEL}", 
              fontsize=15, fontweight='bold')
    plt.xlabel("Mean Score", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f"Bar_Integrated_{model_name}.png"
    plt.savefig(str(save_path), dpi=300)
    plt.close('all')
    print(f"  ✓ Generated Bar Chart for {model_name}")

def plot_concordance_matrix():
    input_path = ANALYSIS_DIR / "concordance_matrix.csv"
    if not input_path.exists(): return

    df = pd.read_csv(input_path, index_col=0)
    df = df.rename(index=PRETTY_NAMES, columns=PRETTY_NAMES)
    
    plt.figure(figsize=(10, 9)) 
    sns.set_theme(style="white") 

    mask = np.triu(np.ones_like(df, dtype=bool), k=1)
    cmap = sns.color_palette("RdYlBu_r", as_cmap=True)

    ax = sns.heatmap(
        df, mask=mask, cmap=cmap, vmax=1.0, vmin=0.0, center=0.5,
        square=True, linewidths=0.5, cbar_kws={"shrink": .7, "label": "Kendall's τ"},
        annot=True, fmt=".3f", annot_kws={"size": 14}
    )
    
    plt.title(f"Model Agreement (Concordance)\n{DATASET_LABEL}", fontsize=18, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    ax.tick_params(left=False, bottom=False)
    
    save_path = FIGURES_DIR / "concordance_matrix.png"
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  ✓ Generated Concordance Matrix")

def main():
    print(f"\n=== GENERATING {DATASET_LABEL} VISUALIZATIONS ===")
    if generate_ensemble_data():
        models_to_plot = BASE_MODELS + ["Ensemble"]
    else:
        models_to_plot = BASE_MODELS
    
    for model in models_to_plot:
        plot_heatmap_clustered(model)
        plot_integrated_relevance(model)
        
    plot_concordance_matrix()

if __name__ == "__main__":
    main()