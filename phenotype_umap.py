import json
import os
from collections import defaultdict
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# =========================================================================
# === 1. V9.0 CLASSIFICATION RULES & LOGIC (Merged) ===
# =========================================================================

CELL_CLASSIFICATION_THRESHOLD = 0.5
MIN_POS_PROPORTION_V9 = 0.45
MIN_CLASSIFICATION_SCORE_V9 = 0.1

CLASSIFICATION_RULES = {
    "Treg": {"pos": ["CD45", "CD3e", "CD4", "FoxP3"], "neg": ["CD8", "CD20", "CD68", "PanCK"]},
    "CD8+ T Cell": {"pos": ["CD45", "CD3e", "CD8"], "neg": ["CD4", "FoxP3", "CD20", "CD68", "PanCK"]},
    "CD4+ T Cell": {"pos": ["CD45", "CD3e", "CD4"], "neg": ["CD8", "FoxP3", "CD20", "CD68", "PanCK"]},
    "B Cell": {"pos": ["CD45", "CD20"], "neg": ["CD3e", "CD68", "CD163", "CD66", "CD11c", "PanCK"]},
    "NK Cell": {"pos": ["CD45"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD66", "CD11c", "PanCK", "CD31"]},
    "Neutrophil": {"pos": ["CD45", "CD66"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD11c", "PanCK"]},
    "Macrophage (CD163+)": {"pos": ["CD45", "CD163"], "neg": ["CD3e", "CD20", "CD66", "CD11c", "PanCK"]},
    "Myeloid Cell (CD68+)": {"pos": ["CD45", "CD68"], "neg": ["CD3e", "CD20", "CD66", "CD163", "CD11c", "PanCK"]},
    "Dendritic Cell": {"pos": ["CD45", "CD11c"], "neg": ["CD3e", "CD20", "CD66", "CD68", "CD163", "PanCK"]},
    "Monocyte/MDSC-like": {"pos": ["CD45", "CD68"], "neg": ["CD3e", "CD20", "CD31", "PanCK", "Vimentin"]},
    "Epithelial Cell": {"pos": ["PanCK"], "neg": ["CD45", "Vimentin"]},
    "Fibroblast": {"pos": ["Vimentin"], "neg": ["CD45", "PanCK"]},
    "Endothelial Cell": {"pos": ["CD31"], "neg": ["CD45", "PanCK", "Vimentin"]},
    "Non_Immune_Unspec": {"pos": [], "neg": ["CD45", "PanCK", "Vimentin", "CD31"]},
    "Background/Junk": {"pos": [], "neg": ["CD45", "PanCK", "Vimentin", "CD31"]}
}

FUNCTIONAL_OVERLAYS = {
    "T_Effector_Mem": ["CD3e", "CD45RO", "GranzymeB"],
    "T_Exhausted_PD1+": ["CD3e", "PD1"],
    "Macrophage_TME_Reg": ["CD163", "PDL1", "VISTA"]
}


def fuzzy_classify_with_overlay(cell: Dict[str, Any]) -> str:
    # Logic remains the same, but now returns only the primary label string
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
            if "CD45" in rule["neg"]:
                if all(signals.get(marker, 0) <= CELL_CLASSIFICATION_THRESHOLD for marker in rule["neg"]):
                    current_score = 1.0
                else:
                    current_score = -1.0
            else:
                current_score = -neg_count

        if current_score > best_relative_score:
            best_relative_score = current_score
            best_type = cell_type

    primary_label = best_type if best_relative_score >= MIN_CLASSIFICATION_SCORE_V9 else "Unclassified_By_New_Rules"

    # We ignore functional overlays for the UMAP plot primary color assignment
    return primary_label


# =========================================================================
# === 2. DATA LOADING FUNCTION (Replaced recursive_cell_analysis) ===
# =========================================================================

def load_all_cell_data(root_dir: str) -> list:
    """Recursively loads all cell objects from JSON files for UMAP plotting."""
    all_cells_list = []
    print(f"--- 🚀 Starting to load cell data from '{root_dir}' for UMAP ---")

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
                        print(f"  ✅ Loaded {len(data)} cells from {file_path}")
                    else:
                        print(f"  ⚠️ Skipping {file_path}: content is not a list.")
                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")

    print(f"--- ✅ Data loading complete. Total cells collected: {len(all_cells_list)} ---")
    return all_cells_list


# =========================================================================
# === 3. UMAP GENERATION AND PLOTTING FUNCTION (V10.0 with Scaling/Tuning) ===
# =========================================================================

# def generate_umap_plot(all_cell_data: pd.DataFrame, marker_list: list,
#                        output_path: str = 'umap_cell_types_V10_FINAL_Fig2_50biomarker.png'):
#     # ... (Step 1: Prepare Data for UMAP) ...
#     X = all_cell_data[marker_list].values
#
#     # 1. Data Scaling (StandardScaler)
#     print("--- Scaling Data (StandardScaler) ---")
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # **NEW FIX (V10.1): Adding Gaussian Jitter to prevent Spectral Failure**
#     print("--- Adding Jitter to Scaled Data ---")
#     jitter = np.random.normal(0, 1e-4, size=X_scaled.shape)
#     X_scaled_jittered = X_scaled + jitter
#
#     # 2. Run UMAP (Tuned parameters, using the jittered data)
#     reducer = umap.UMAP(
#         n_components=2,
#         random_state=42,
#         n_neighbors=8,
#         min_dist=0.3,
#         metric='euclidean'
#     )
#     # 💥 CRITICAL CHANGE: Use X_scaled_jittered instead of X_scaled
#     umap_coordinates = reducer.fit_transform(X_scaled_jittered)
#
#     # 3. Add UMAP coordinates to the DataFrame
#     all_cell_data['UMAP_1'] = umap_coordinates[:, 0]
#     all_cell_data['UMAP_2'] = umap_coordinates[:, 1]
#
#     # 4. Plotting
#     print("--- 🖼️ 2. Generating Plot ---")
#     plt.figure(figsize=(10, 10))
#
#     # To ensure Unclassified_By_New_Rules is often gray/last in the legend, we sort the types
#     cell_types_order = sorted(all_cell_data['cell_type'].unique())
#     if "Unclassified_By_New_Rules" in cell_types_order:
#         cell_types_order.remove("Unclassified_By_New_Rules")
#         cell_types_order.append("Unclassified_By_New_Rules")  # Move to end
#
#     sns.scatterplot(
#         x='UMAP_1', y='UMAP_2',
#         hue='cell_type',
#         data=all_cell_data,
#         s=10,
#         linewidth=0,
#         alpha=0.8,
#         palette='tab20',
#         hue_order=cell_types_order
#     )
#
#     plt.title('UMAP of Cell Marker Signals Colored by Cell Type ')  #(V9.0 Final)
#     plt.xlabel('UMAP_1')
#     plt.ylabel('UMAP_2')
#     plt.legend(title='Cell Type', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(output_path)
#     print(f"✅ UMAP plot saved to {output_path}")

def generate_umap_plot(all_cell_data: pd.DataFrame, marker_list: list,
                       output_path_base: str = 'umap_cell_types_V10_FINAL_Fig2_50biomarker'):
    # === Step 1, 2, 3 保持不变 ===
    X = all_cell_data[marker_list].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    jitter = np.random.normal(0, 1e-4, size=X_scaled.shape)
    X_scaled_jittered = X_scaled + jitter

    reducer = umap.UMAP(
        n_components=2,
        random_state=42,
        n_neighbors=8,
        min_dist=0.3,
        metric='euclidean'
    )
    umap_coordinates = reducer.fit_transform(X_scaled_jittered)

    all_cell_data['UMAP_1'] = umap_coordinates[:, 0]
    all_cell_data['UMAP_2'] = umap_coordinates[:, 1]

    # === Step 4: Plotting (重点修改这里) ===
    print("--- 🖼️ 2. Generating Plots (PNG & PDF) with Enhanced Fonts ---")

    # 稍微调大画布宽度 (从10改到12)，防止大字体图例挤压主图
    plt.figure(figsize=(12, 10))

    cell_types_order = sorted(all_cell_data['cell_type'].unique())
    if "Unclassified_By_New_Rules" in cell_types_order:
        cell_types_order.remove("Unclassified_By_New_Rules")
        cell_types_order.append("Unclassified_By_New_Rules")

    sns.scatterplot(
        x='UMAP_1', y='UMAP_2',
        hue='cell_type',
        data=all_cell_data,
        s=12,  # 散点稍微调大一点点，更清晰
        linewidth=0,
        alpha=0.8,
        palette='tab20',
        hue_order=cell_types_order
    )

    # 1. 标题字号调整 (fontsize=20)
    plt.title('UMAP of Cell Marker Signals Colored by Cell Type', fontsize=20, pad=15)

    # 2. 坐标轴标签字号调整 (fontsize=16)
    plt.xlabel('UMAP_1', fontsize=16)
    plt.ylabel('UMAP_2', fontsize=16)

    # 3. 坐标轴刻度数字调大 (fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 4. 图例 (Legend) 修改：重点调整这里
    plt.legend(
        title='Cell Type',
        title_fontsize=16,  # 图例标题“Cell Type”的字号
        fontsize=14,  # 细胞类型名称（Treg, B Cell等）的字号
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        markerscale=1.5  # 让图例里的小圆点也变大一点，方便区分颜色
    )

    plt.tight_layout()

    # --- SAVE BOTH FORMATS ---
    plt.savefig(f"{output_path_base}.png", dpi=300)
    plt.savefig(f"{output_path_base}.pdf", format='pdf', bbox_inches='tight')

    plt.close()
    print(f"✅ UMAP plots saved as:\n   1. {output_path_base}.png\n   2. {output_path_base}.pdf")

# =========================================================================
# === 4. MAIN EXECUTION ===
# =========================================================================

if __name__ == "__main__":

    root_directory = './'  # Assumes JSON files are in the current or sub-directories

    # 1. Load All Cell Data
    raw_cell_list = load_all_cell_data(root_directory)

    if not raw_cell_list:
        print("❌ Fatal Error: No cell data found. Cannot generate UMAP plot.")
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
            # Create a dictionary of marker signals for the row
            row = {m: signals.get(m, 0.0) for m in all_markers}
            # Classify the cell
            row['cell_type'] = fuzzy_classify_with_overlay(cell_dict)
            data_for_df.append(row)

        df_cells = pd.DataFrame(data_for_df)

        # 4. Run UMAP and Plot
        generate_umap_plot(df_cells, all_markers)