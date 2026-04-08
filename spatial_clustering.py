import json
import os
from collections import defaultdict
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.spatial import Delaunay
# --- NEW IMPORTS FOR CLUSTERING ---
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# -----------------------------------

# =========================================================================
# === 1. V9.0 CLASSIFICATION RULES & LOGIC ===
# =========================================================================

CELL_CLASSIFICATION_THRESHOLD = 0.05  # Threshold for positivity
MIN_POS_PROPORTION_V9 = 0.45
MIN_CLASSIFICATION_SCORE_V9 = 0.05
OUTPUT_PLOT_PATH = 'spatial_clusters_DBSCAN_Normal.png'

CLASSIFICATION_RULES = {
    # T cells: Mutual exclusion for CD4/CD8
    "Treg": {"pos": ["CD45", "CD3e", "CD4", "FoxP3"], "neg": ["CD8", "CD20", "CD68", "PanCK"]},
    "CD8+ T Cell": {"pos": ["CD45", "CD3e", "CD8"], "neg": ["CD4", "FoxP3", "CD20", "CD68", "PanCK"]},
    "CD4+ T Cell": {"pos": ["CD45", "CD3e", "CD4"], "neg": ["CD8", "FoxP3", "CD20", "CD68", "PanCK"]},

    "B Cell": {"pos": ["CD45", "CD20"], "neg": ["CD3e", "CD68", "CD163", "CD66", "CD11c", "PanCK"]},
    "NK Cell": {"pos": ["CD45"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD66", "CD11c", "PanCK", "CD31"]},

    # Myeloid Cells
    "Neutrophil": {"pos": ["CD45", "CD66"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD11c", "PanCK"]},
    "Macrophage (CD163+)": {"pos": ["CD45", "CD163"], "neg": ["CD3e", "CD20", "CD66", "CD11c", "PanCK"]},
    "Myeloid Cell (CD68+)": {"pos": ["CD45", "CD68"], "neg": ["CD3e", "CD20", "CD66", "CD163", "CD11c", "PanCK"]},
    "Dendritic Cell": {"pos": ["CD45", "CD11c"], "neg": ["CD3e", "CD20", "CD66", "CD68", "CD163", "PanCK"]},
    "Monocyte/MDSC-like": {"pos": ["CD45", "CD68"], "neg": ["CD3e", "CD20", "CD31", "PanCK", "Vimentin"]},

    # Stromal / Non-immune Cells
    "Epithelial Cell": {"pos": ["PanCK"], "neg": ["CD45", "Vimentin"]},
    "Fibroblast": {"pos": ["Vimentin"], "neg": ["CD45", "PanCK"]},
    "Endothelial Cell": {"pos": ["CD31"], "neg": ["CD45", "PanCK", "Vimentin"]},
    "Non_Immune_Unspec": {"pos": [], "neg": ["CD45", "PanCK", "Vimentin", "CD31"]},
    "Background/Junk": {"pos": [], "neg": ["CD45", "PanCK", "Vimentin", "CD31"]}
}


def fuzzy_classify_primary_label(cell: Dict[str, Any]) -> str:
    """Applies V9.0 rules and returns ONLY the primary cell type label."""
    signals = cell.get("Signals", {})
    best_relative_score = float('-inf')
    best_type = "Unclassified_By_New_Rules"

    for cell_type, rule in CLASSIFICATION_RULES.items():
        required_pos = rule["pos"]

        pos_count = sum(1 for marker in required_pos if signals.get(marker, 0) > CELL_CLASSIFICATION_THRESHOLD)
        neg_count = sum(1 for marker in rule["neg"] if signals.get(marker, 0) > CELL_CLASSIFICATION_THRESHOLD)

        if required_pos:
            pos_proportion = pos_count / len(required_pos)
            if pos_proportion >= MIN_POS_PROPORTION_V9:
                current_score = pos_proportion - (neg_count * 0.2)
            else:
                current_score = float('-inf')
        else:
            if all(signals.get(marker, 0) <= CELL_CLASSIFICATION_THRESHOLD for marker in rule["neg"]):
                current_score = 1.0
            else:
                current_score = -1.0

        if current_score > best_relative_score:
            best_relative_score = current_score
            best_type = cell_type

    return best_type if best_relative_score >= MIN_CLASSIFICATION_SCORE_V9 else "Unclassified_By_New_Rules"


# =========================================================================
# === 2. DATA LOADING FUNCTION ===
# =========================================================================

def load_all_cell_data(root_dir: str) -> list:
    """Recursively loads all cell objects from JSON files."""
    all_cells_list = []
    print(f"--- 🚀 Starting to load cell data from '{root_dir}' for plotting ---")

    for dirpath, _, filenames in os.walk(root_dir):
        json_files = [f for f in filenames if f.endswith('.json')]
        if json_files:
            for filename in json_files:
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        all_cells_list.extend(data)
                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")

    print(f"--- ✅ Data loading complete. Total cells collected: {len(all_cells_list)} ---")
    return all_cells_list


# =========================================================================
# === 3. DELAUNAY TRIANGULATION PLOTTING (FOR CONTACT MAPPING) ===
# =========================================================================
def analyze_cell_interactions(df_cells: pd.DataFrame,
                              output_plot_path: str = 'cell_interaction_plot_Normal.png'):
    """Performs Delaunay triangulation and plots the raw connectivity map."""
    df_unique_points = df_cells.drop_duplicates(subset=['x', 'y'], keep='first')
    points = df_unique_points[['x', 'y']].values

    # 将 df_unique_points 中的 cell_type 映射到 points 数组
    cell_types_of_points = df_unique_points['cell_type'].values

    if len(points) < 4:
        print("  ❌ Not enough unique points to perform triangulation. Skipping.")
        return

    # 1. Perform Delaunay triangulation
    tri = Delaunay(points, qhull_options='Qz QJ1e-6')

    # 过滤掉无效的索引
    max_valid_index = len(points) - 1
    safe_simplices = tri.simplices[~np.any(tri.simplices > max_valid_index, axis=1)]

    if safe_simplices.size == 0:
        print("  ❌ Error: All simplices were invalid after filtering. Cannot plot triangulation.")
        return

    # 2. --- Generate a plot of the triangulation ---
    print("--- 🎨 2. Generating Delaunay Triangulation Plot (Connectivity) ---")
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # **绘制连接线** (保持细且为灰色，表示一般的空间连接)
    # 使用 triplot 绘制所有边
    plt.triplot(points[:, 0], points[:, 1], safe_simplices, color='gray', linewidth=0.15, alpha=0.3)

    # 3. **根据细胞类型绘制点** (着色部分)

    # 获取唯一的细胞类型并设置颜色映射
    cell_types = df_unique_points['cell_type'].unique()
    num_cell_types = len(cell_types)

    # 使用 tab20 Colormap，它有 20 种颜色，适合多种细胞类型
    colors_map = plt.cm.get_cmap('tab20', num_cell_types)
    color_dict = {cell_type: colors_map(i) for i, cell_type in enumerate(cell_types)}

    # 分别绘制每种细胞类型的点
    for cell_type in cell_types:
        # 获取属于当前细胞类型的所有点的索引
        indices = np.where(cell_types_of_points == cell_type)[0]

        plt.scatter(
            points[indices, 0],
            points[indices, 1],
            color=color_dict[cell_type],
            label=cell_type,
            s=10,  # 点大小适中
            alpha=0.8
        )

    plt.title(f"Cell-Cell Interaction Plot (Normal) - Colored by Cell Type")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().set_aspect('equal', adjustable='box')

    # 放置图例在图片外右侧
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2, title="Cell Type")

    plt.savefig(output_plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✅ Interaction plot saved to {output_plot_path}")


# =========================================================================
# === 4. DBSCAN SPATIAL CLUSTERING (FOR CLUSTER/NICHE DETECTION) ===
# =========================================================================

def generate_spatial_clusters(df_cells: pd.DataFrame, eps: float = 0.01, min_samples: int = 10) -> pd.DataFrame:
    """
    Applies DBSCAN to cell coordinates to identify spatially dense clusters (niches).

    Args:
        df_cells: DataFrame containing 'x' and 'y' coordinates.
        eps: The maximum distance (after scaling) between two samples for one
             to be considered as in the neighborhood of the other.
        min_samples: The number of samples (cells) in a neighborhood for a
                     point to be considered as a core point.

    Returns:
        The input DataFrame with a new 'cluster_id' column.
    """
    print("\n--- 🧠 3. Performing DBSCAN Spatial Clustering ---")

    coords = df_cells[['x', 'y']].values

    # 1. Scaling the coordinates is crucial for DBSCAN on un-normalized data
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    # 2. Apply DBSCAN
    # Note: eps and min_samples are critical and often require manual tuning.
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_scaled)

    df_cells['cluster_id'] = db.labels_
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = list(db.labels_).count(-1)

    print(f"  ✅ DBSCAN complete: Found {n_clusters} clusters. {n_noise} cells marked as noise (-1).")

    return df_cells


def plot_spatial_clusters(df_clustered: pd.DataFrame, output_plot_path: str = 'spatial_clusters_DBSCAN.png'):
    """Generates a scatter plot of cells colored by their cluster ID."""

    print("--- 🎨 4. Generating Spatial Cluster Plot ---")

    # Separate noise from clusters
    df_noise = df_clustered[df_clustered['cluster_id'] == -1]
    df_clusters = df_clustered[df_clustered['cluster_id'] != -1]

    # Calculate n_clusters directly from the filtered DataFrame to ensure consistency
    n_clusters = df_clusters['cluster_id'].nunique()

    plt.figure(figsize=(12, 12))

    # 绘制 DBSCAN 散点图
    # 无论有没有找到簇，都先进行一次散点图绘制，以确保 'scatter' 变量被定义
    # 这里的颜色处理逻辑需要简化，避免 n_clusters=0 时 ListedColormap 报错
    scatter = plt.scatter(
        df_clustered['x'],
        df_clustered['y'],
        # 使用 cluster_id 作为颜色，让 Matplotlib 自动处理颜色循环
        c=df_clustered['cluster_id'],
        # 对负值（噪音）使用灰色，对正值（簇）使用颜色循环
        cmap='tab20' if n_clusters > 0 else ListedColormap(['gray']),
        s=5,
        alpha=0.8
    )

    plt.title(f'Spatial Cell Clustering (DBSCAN) - {n_clusters} Clusters')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.gca().set_aspect('equal', adjustable='box')

    # 只有在找到有效簇 (n_clusters > 0) 时，才尝试绘制复杂的 Cluster ID 图例
    if n_clusters > 0:
        # 使用 scatter.legend_elements() 获取所有簇的图例元素
        legend1 = plt.legend(*scatter.legend_elements(), title="Cluster ID", loc="lower left")
        plt.gca().add_artist(legend1)
    else:
        # 如果只找到噪音点 (n_clusters = 0)，则只绘制噪音点图例
        noise_handle = Line2D([0], [0], marker='o', color='w', label='Noise (-1)',
                              markerfacecolor='gray', markersize=5)
        plt.legend(handles=[noise_handle], title="Cluster ID", loc="lower left")

    plt.savefig(output_plot_path, dpi=300)
    print(f"  ✅ Spatial cluster plot saved to {output_plot_path}")


# =========================================================================
# === 5. DOT PLOT MATRIX GENERATION (Unmodified for brevity) ===
# =========================================================================

def generate_dot_plot_matrix(df_cells: pd.DataFrame, marker_list: list,
                             output_path: str = 'dot_plot_matrix_V14_final.png'):
    # ... (Keep this function here, but I will not paste its body for brevity,
    # as the user did not ask to modify it) ...
    print("--- 🔬 1. Calculating Dot Plot Statistics ---")

    # 1. Calculate Average Expression (for Color)
    mean_expression = df_cells.groupby('cell_type')[marker_list].mean()

    # 2. Calculate Percentage Positive (for Size)
    positive_cells = (df_cells[marker_list] > CELL_CLASSIFICATION_THRESHOLD).groupby(df_cells['cell_type']).sum()
    total_cells_per_type = df_cells['cell_type'].value_counts()
    percentage_positive = (positive_cells.T / total_cells_per_type).T * 100

    # 3. Restructure DataFrames for Plotting
    plot_data = []
    for cell_type in mean_expression.index:
        for marker in marker_list:
            plot_data.append({
                'cell_type': cell_type,
                'marker': marker,
                'mean_expression': mean_expression.loc[cell_type, marker],
                'percentage_positive': percentage_positive.loc[cell_type, marker]
            })

    df_plot = pd.DataFrame(plot_data)

    cell_type_order = [
        "Treg", "CD8+ T Cell", "CD4+ T Cell", "B Cell", "NK Cell",
        "Neutrophil", "Macrophage (CD163+)", "Myeloid Cell (CD68+)",
        "Dendritic Cell", "Monocyte/MDSC-like",
        "Epithelial Cell", "Fibroblast", "Endothelial Cell",
        "Non_Immune_Unspec", "Background/Junk", "Unclassified_By_New_Rules"
    ]

    df_plot['cell_type'] = pd.Categorical(
        df_plot['cell_type'],
        categories=cell_type_order,
        ordered=True
    )

    df_plot = df_plot.sort_values('cell_type')

    print("--- 🖼️ 2. Generating Dot Plot (Color Scale Adjusted) ---")

    vmax_cap = 1.5
    vmin_val = df_plot['mean_expression'].min()

    plt.figure(figsize=(12, 0.5 * len(cell_type_order)))

    scatter = sns.scatterplot(
        data=df_plot,
        x='marker',
        y='cell_type',
        size='percentage_positive',
        hue='mean_expression',
        sizes=(20, 500),
        palette='vlag',
        hue_norm=(vmin_val, vmax_cap),
        linewidth=0,
        edgecolor='black'
    )

    plt.gca().invert_yaxis()
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.xticks(rotation=90)
    plt.xlabel('Marker')
    plt.ylabel('Cell Type')
    plt.title('Dot Plot Matrix of Cell Marker Expression by Cell Type')

    try:
        scatter.legend_.remove()
    except Exception:
        pass

    legend_percentages = [1, 25, 50, 75, 100]
    min_dot_size, max_dot_size = 20, 500
    min_p, max_p = 0, 100

    def map_to_markersize(percentage, min_p, max_p, min_s, max_s):
        if max_p == min_p: return min_s
        p_norm = max(0, percentage)
        return min_s + (max_s - min_s) * (p_norm - min_p) / (max_p - min_p)

    size_handles = []
    size_labels = []
    for p in legend_percentages:
        ms = map_to_markersize(p, min_p, max_p, min_dot_size, max_dot_size)
        size_handles.append(Line2D([0], [0], marker='o', color='gray',
                                   markersize=np.sqrt(ms) * 0.7,
                                   linestyle=''))
        size_labels.append(f'{p:.0f}%')

    l1 = plt.legend(size_handles, size_labels, title="Percent Positive (%)",
                    bbox_to_anchor=(1.05, 1.0), loc='upper left', frameon=False)
    plt.gca().add_artist(l1)

    norm = plt.Normalize(vmin_val, vmax_cap)
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    sm.set_array([])

    cbar_ax = plt.gcf().add_axes([0.9, 0.1, 0.03, 0.35])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Mean Expression (A.U.)', rotation=270, labelpad=15)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(output_path, dpi=300)
    print(f"✅ Dot Plot Matrix saved to {output_path}")


# =========================================================================
# === 6. MAIN EXECUTION ===
# =========================================================================

if __name__ == "__main__":

    root_directory = 'Normal_full_tiff_50ch' #'./'

    # 1. Load All Cell Data
    raw_cell_list = load_all_cell_data(root_directory)

    if not raw_cell_list:
        print("❌ Fatal Error: No cell data found. Cannot proceed.")
    else:
        # 2. Define All Markers
        all_markers = [
            "CD45", "CD3e", "CD4", "CD8", "FoxP3", "CD20", "CD68", "CD163", "CD66", "CD11c",
            "PanCK", "Vimentin", "CD31", "CD45RO", "GranzymeB", "PD1", "PDL1", "VISTA"
        ]

        # 3. Convert to DataFrame and Apply Classification
        data_for_df = []
        for cell_dict in raw_cell_list:
            signals = cell_dict.get('Signals', {})
            row = {m: signals.get(m, 0.0) for m in all_markers}
            # *** ADD COORDINATES FOR SPATIAL ANALYSIS ***
            row['x'] = cell_dict.get('Centroid X', 0)
            row['y'] = cell_dict.get('Centroid Y', 0)
            row['cell_type'] = fuzzy_classify_primary_label(cell_dict)
            data_for_df.append(row)

        df_cells = pd.DataFrame(data_for_df)

        # 4. Filter out Unclassified cells
        df_cells_filtered = df_cells[df_cells['cell_type'] != 'Unclassified_By_New_Rules'].copy()

        if not df_cells_filtered.empty:

            # 1. 计算 Count
            cell_counts = df_cells_filtered['cell_type'].value_counts()
            # 2. 计算 Proportion
            cell_proportions = df_cells_filtered['cell_type'].value_counts(normalize=True) * 100

            # 3. 合并成一个DataFrame并打印
            composition_df = pd.concat([cell_counts, cell_proportions.round(2)], axis=1,
                                       keys=['Count', 'Proportion (%)'])
            composition_df.index.name = 'Cell Type'

            print("\n--- ✅ Normal 状态细胞类型组成 ---")
            print(composition_df)
            print("-" * 40)

            # 5. Run Delaunay Plot (Connectivity Map)
            analyze_cell_interactions(df_cells_filtered)

            # 6. *** NEW: Run DBSCAN Spatial Clustering ***
            # NOTE: eps=0.01 and min_samples=10 are starting values.
            # You may need to tune 'eps' if too few/many clusters are found.
            # df_clustered = generate_spatial_clusters(df_cells_filtered.copy(), eps=0.01, min_samples=10)
            df_clustered = generate_spatial_clusters(df_cells_filtered.copy(), eps=0.08, min_samples=10)

            ########################################################################################################################
            # 7. **【新增】分析每个簇的细胞组成**
            # 排除噪音点 (-1) 进行分析
            df_analysis = df_clustered[df_clustered['cluster_id'] != -1].copy()

            # new added for color cells
            if not df_analysis.empty:
                print("\n--- 🔬 Normal 空间簇细胞组成分析 ---")

                # 1. 按 cluster_id 和 cell_type 分组计数
                cluster_composition = df_analysis.groupby(['cluster_id', 'cell_type']).size().unstack(fill_value=0)

                # 2. 计算每个簇的总细胞数
                cluster_composition['Total_Cells_In_Cluster'] = cluster_composition.sum(axis=1)

                # 3. 计算每个簇的细胞类型比例 (行百分比)
                # 提取细胞类型列 (排除 Total_Cells_In_Cluster)
                cell_type_cols = cluster_composition.columns[:-1]
                cluster_proportions = (
                        cluster_composition[cell_type_cols]
                        .div(cluster_composition['Total_Cells_In_Cluster'], axis=0) * 100
                ).round(2)

                # 4. 打印最终结果
                print("\n簇内细胞类型比例 (%):")
                print(cluster_proportions)

                # 打印簇的总览
                print("\n簇总细胞数:")
                print(cluster_composition['Total_Cells_In_Cluster'])


            # 7. *** NEW: Plot Spatial Clusters ***
            plot_spatial_clusters(df_clustered, OUTPUT_PLOT_PATH)

            # 8. Run Dot Plot Generation
            # generate_dot_plot_matrix(df_cells_filtered, all_markers)
        else:
            print("❌ Error: DataFrame is empty after filtering/classification. Cannot proceed.")