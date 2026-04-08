"""
Evaluation script for H&E to multiplex protein prediction model (WITH CELL OVERLAY AND JSON EXPORT).

This script loads a 50-channel FCN model, runs inference using a **weighted-blending, sliding-window approach**,
slices the output to 50 target channels, and saves:
1. A 50-channel TIFF file.
2. A colorful PNG visualization with cell centroids overlaid and a legend.
3. A JSON file containing classified cell coordinates and signals.
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import tifffile
import argparse
from typing import Tuple, Optional, List, Dict, Any
from pathlib import Path
import pdb
from PIL import Image
import cv2
from scipy.signal import convolve2d
from skimage.morphology import dilation, disk
from scipy.ndimage import gaussian_filter
import glob
import torch.serialization
from torchvision.transforms import InterpolationMode
import pandas as pd
import json
import re
# --- Imports for Visualization and Data Handling ---
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import pandas as pd
import json
import re
# -------------------------------------------------

# =======================================================================
# ✨ CONFIGURATION CONSTANTS
# =======================================================================
VIPS_DLL_PATH = r"C:\vips-dev-w64-all-8.17.2\bin"  # !!! Replace with your local libvips bin directory !!!

if os.name == 'nt':
    try:
        os.environ['PATH'] = VIPS_DLL_PATH + os.pathsep + os.environ.get('PATH', '')
        print(f"✅ Temporarily added DLL search path: {VIPS_DLL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not set VIPS DLL path: {e}")

try:
    import pyvips
except (ImportError, OSError):
    pyvips = None
    print("Warning: pyvips not installed. Multi-resolution TIFF output will fall back to single-layer tifffile.")

BATCH_SIZE = 32
NUM_WORKERS = 8
PATCH_SIZE = 128
WHITE_THRESHOLD = 220
BACKGROUND_RATIO = 0.8
TIFF_COMPRESSION = "lzw"
NUM_MODEL_CHANNELS = 50  # Model outputs 50 channels
PIXEL_SIZE_MPP = 0.8
ROI_OFFSET_X_PX  = 0
ROI_OFFSET_Y_PX  = 0

# --- Full 50-Channel Marker List (Model Output) ---
MARKER_LIST_50CH = [
    "DAPI", "CD45", "CD68", "CD14", "PD1", "FoxP3", "CD8", "HLA-DR", "PanCK", "CD3e",
    "CD4", "aSMA", "CD31", "Vimentin", "CD45RO", "Ki67", "CD20", "CD11c", "Podoplanin", "PDL1",
    "GranzymeB", "CD38", "CD141", "CD21", "CD163", "BCL2", "LAG3", "EpCAM", "CD44", "ICOS",
    "GATA3", "Gal3", "CD39", "CD34", "TIGIT", "ECad", "CD40", "VISTA", "HLA-A", "MPO",
    "PCNA", "ATM", "TP63", "IFNg", "Keratin8/18", "IDO1", "CD79a", "HLA-E", "CollagenIV", "CD66"
]

# --- 50 Target Channel Colors and Names for Visualization ---
TARGET_CHANNEL_COLORS = {
    "DAPI": "blue",
    "CD45": "lime",
    "CD68": "brown",
    "CD14": "tan",
    "PD1": "darkolivegreen",
    "FoxP3": "plum",
    "CD8": "darkgreen",
    "HLA-DR": "yellow",
    "PanCK": "firebrick",
    "CD3e": "green",
    "CD4": "lightgreen",
    "aSMA": "purple",
    "CD31": "cyan",
    "Vimentin": "indigo",
    "CD45RO": "mediumseagreen",
    "Ki67": "orange",
    "CD20": "hotpink",
    "CD11c": "beige",
    "Podoplanin": "darkcyan",
    "PDL1": "red",
    "GranzymeB": "darkslategray",
    "CD38": "palevioletred",
    "CD141": "wheat",
    "CD21": "lightpink",
    "CD163": "saddlebrown",
    "BCL2": "dodgerblue",
    "LAG3": "forestgreen",
    "EpCAM": "indianred",
    "CD44": "olivedrab",
    "ICOS": "darkseagreen",
    "GATA3": "springgreen",
    "Gal3": "chocolate",
    "CD39": "mediumorchid",
    "CD34": "teal",
    "TIGIT": "darkviolet",
    "ECad": "lightcoral",
    "CD40": "mediumvioletred",
    "VISTA": "peru",
    "HLA-A": "khaki",
    "MPO": "darkgoldenrod",
    "PCNA": "darkorange",
    "ATM": "gold",
    "TP63": "tomato",
    "IFNg": "limegreen",
    "Keratin8/18": "salmon",
    "IDO1": "rosybrown",
    "CD79a": "deeppink",
    "HLA-E": "lemonchiffon",
    "CollagenIV": "slategray",
    "CD66": "darkkhaki"
}



SELECTED_MARKERS = list(TARGET_CHANNEL_COLORS.keys())
COLOR_RGBS = np.array([to_rgb(c) for c in TARGET_CHANNEL_COLORS.values()], dtype=np.float32)

# Get the indices of the 50 target channels from the full 50-channel list
TARGET_INDICES = [MARKER_LIST_50CH.index(m) for m in SELECTED_MARKERS]


# =======================================================================
# ✨ MODEL PREDICTION / INFERENCE FUNCTIONS
# =======================================================================

def pad_patch(patch: np.ndarray,
              original_size: Tuple[int, int],
              x_center: int,
              y_center: int,
              patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Pads the given patch if its size is less than patch_size x patch_size pixels."""
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]

    if current_height == patch_size and current_width == patch_size:
        return patch

    # Calculate padding needed
    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    # Apply padding
    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else (
    (pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)

    # Ensure the patch is exactly patch_size x patch_size
    padded_patch = padded_patch[:patch_size, :patch_size]

    return padded_patch


def box_blur(image_array, window_size=8):
    """Apply box blur to an image array using convolution."""
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    blurred_array = convolve2d(image_array, kernel, mode='same')
    return blurred_array


def normalize_image(image, min_value, max_value):
    """Normalize image values to 0-255 range."""
    if max_value == min_value:
        return np.zeros_like(image, dtype=np.uint8)

    return ((image - min_value) * 255. / (max_value - min_value)).astype(np.uint8)


def get_model(num_outputs: int) -> nn.Module:
    """Creates and returns the model architecture (ConvNext_small for regression)."""
    model = models.convnext_small(weights=None)
    # Replace the final classification layer with a linear layer for regression outputs
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model


class ImageDataset(Dataset):
    """Dataset class for loading H&E image patches from either ZARR or PNG."""

    def __init__(self, image_path: str, transform: Optional[dict] = None, stride_size: int = 8,
                 exclude_background: bool = True):
        self.image_path = image_path
        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size // 2
        self.stride_size = stride_size
        self.center_half = max(1, stride_size // 2)

        # Load image based on file type
        if image_path.endswith('.zarr'):
            # Load ZARR files (multi-channel)
            reader = Reader(parse_url(image_path, mode="r"))
            zarr_data = list(reader())[0].data[0].compute()

            if zarr_data.ndim == 4 and zarr_data.shape[0] == 1:
                zarr_data = zarr_data[0]  # Drop the Z dimension if present (1, C, H, W)

            # Determine channel axis and take first 3 channels for H&E
            if zarr_data.shape[-1] >= 3 and zarr_data.ndim >= 3:  # Assuming HxWxC
                channels = zarr_data[..., :3].transpose(2, 0, 1)  # C, H, W
            elif zarr_data.shape[0] >= 3 and zarr_data.ndim >= 3:  # Assuming C, H, W
                channels = zarr_data[:3]
            else:
                raise ValueError(f"Could not extract 3 H&E channels from ZARR with shape: {zarr_data.shape}")

            height, width = channels[0].shape

            # Crop to center 100x100 pixels for ZARR images (as per reference script)
            center_y, center_x = height // 2, width // 2
            crop_size = 100
            half_crop = crop_size // 2

            y_start = max(0, center_y - half_crop)
            y_end = min(height, center_y + half_crop)
            x_start = max(0, center_x - half_crop)
            x_end = min(width, center_x + half_crop)

            self.he_zarr = [channel[y_start:y_end, x_start:x_end] for channel in channels]

        else:  # Handle TIFF/PNG/JPG/SVS
            # Use tifffile for robust TIFF/SVS loading
            img = tifffile.imread(image_path)
            if img.ndim == 4: img = img[0]  # Drop Z-dim if present
            if img.ndim == 3 and img.shape[-1] > 3: img = img[..., :3]  # Take first 3 channels
            if img.ndim == 2: img = np.stack([img] * 3, axis=-1)

            if img.dtype != np.uint8:
                # Normalize to 0-255 if not uint8
                img = (img / np.max(img) * 255).astype(np.uint8)

            if img.shape[-1] == 3:
                # Split channels (H, W, C) -> list of (H, W)
                self.he_zarr = [img[:, :, i] for i in range(3)]
            else:
                raise ValueError(f"Could not load 3 H&E channels from image: {image_path}")

        # Create grid of patch centers
        height, width = self.he_zarr[0].shape
        self.coords = []
        for y in range(0, height, stride_size):
            for x in range(0, width, stride_size):
                if exclude_background:
                    x_start = max(0, x - self.center_half)
                    x_end = min(width, x + self.center_half)
                    y_start = max(0, y - self.center_half)
                    y_end = min(height, y + self.center_half)

                    center_region = np.mean([channel[y_start:y_end, x_start:x_end]
                                             for channel in self.he_zarr], axis=0)
                    avg_value = np.mean(center_region)

                    if avg_value < WHITE_THRESHOLD:
                        self.coords.append((x, y))
                else:
                    self.coords.append((x, y))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single patch from the image."""
        X, Y = self.coords[idx]

        # Extract patch
        b = np.clip(Y - self.ps, 0, self.he_zarr[0].shape[0])
        t = np.clip(Y + self.ps, 0, self.he_zarr[0].shape[0])
        l = np.clip(X - self.ps, 0, self.he_zarr[0].shape[1])
        r = np.clip(X + self.ps, 0, self.he_zarr[0].shape[1])

        # Slice the channels (C, H, W -> H, W, C)
        he_patch = np.array([channel[b:t, l:r] for channel in self.he_zarr]).transpose(1, 2, 0)

        # Pad the extracted patch
        he_patch = pad_patch(he_patch, self.he_zarr[0].shape, X, Y)

        # Apply transforms
        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)

        return patch, X, Y


def create_tissue_mask(he_zarr: List[np.ndarray], dilation_radius=9) -> np.ndarray:
    """Create a tissue mask by identifying non-white areas and dilating."""
    he_array = np.stack(he_zarr, axis=-1)
    tissue_mask = np.any((he_array < 220), axis=-1)
    dilated_mask = dilation(tissue_mask, disk(dilation_radius))
    return dilated_mask


def postprocess_predictions(predictions, tissue_mask, apply_border_threshold=False, bg_percentile=90,
                            max_percentile=99.9):
    """Postprocess model predictions with background thresholding, normalization and masking."""
    processed = np.zeros_like(predictions, dtype=np.uint8)

    height, width = tissue_mask.shape
    pad = 50
    border_mask = np.zeros_like(tissue_mask, dtype=bool)
    border_mask[:pad, :] = True
    border_mask[-pad:, :] = True
    border_mask[:, :pad] = True
    border_mask[:, -pad:] = True

    for i in range(predictions.shape[0]):
        channel = predictions[i]

        if apply_border_threshold:
            bg_values = channel[border_mask]
            bg_threshold = np.percentile(bg_values, bg_percentile) if len(bg_values) > 0 else 0
        else:
            bg_threshold = np.percentile(channel, bg_percentile)

        vmax = np.percentile(channel, max_percentile)

        if vmax <= bg_threshold:
            bg_threshold = 0

        clipped = np.clip(channel, bg_threshold, vmax)
        normalized = normalize_image(clipped, bg_threshold, vmax)
        blurred = box_blur(normalized)
        # masked = blurred * tissue_mask # Masking commented out in reference script
        masked = blurred

        processed[i] = masked

    return processed


# =======================================================================
# ✨ UTILITY FUNCTIONS
# =======================================================================
def load_and_process_cell_data(
        image_path: str, W: int, H: int, selected_markers: list
) -> Optional[pd.DataFrame]:
    """Loads cell data, performs unit conversion/ROI offset correction, and filters."""
    cell_path_txt = Path(image_path).with_suffix('.txt')
    cell_path_tsv = Path(image_path).with_suffix('.tsv')

    # 优先寻找 .txt，其次 .tsv
    if cell_path_txt.exists():
        cell_path = cell_path_txt
    elif cell_path_tsv.exists():
        cell_path = cell_path_tsv
    else:
        print(f"❌ Failed to load cell data: {Path(image_path).stem}.txt or .tsv not found.")
        return None

    print(f"🔎 Found cell data file: {cell_path.name}")
    try:
        # FIX 1: 统一使用 Tab 分隔符 (\t)
        cell_data = pd.read_csv(cell_path, sep='\t', encoding='utf-8-sig')
        original_columns = cell_data.columns.tolist()  # 保存原始列名用于单位判断

        # ====================================================================
        # ✨ 坐标系转换和对齐 (关键修复)
        # ====================================================================

        # 1. 尝试识别 Centroid 列的原始名称和单位
        centroid_x_orig_name = next((col for col in original_columns if 'Centroid X' in col and (
                    'µm' in col or '碌m' in col or 'um' in col.lower() or 'px' in col.lower())), None)
        centroid_y_orig_name = next((col for col in original_columns if 'Centroid Y' in col and (
                    'µm' in col or '碌m' in col or 'um' in col.lower() or 'px' in col.lower())), None)

        if centroid_x_orig_name is None or centroid_y_orig_name is None:
            # 如果找不到带单位后缀的列，则假设它们是 Centroid X 和 Centroid Y
            # 并且假设它们是像素单位。
            centroid_x_orig_name = 'Centroid X'
            centroid_y_orig_name = 'Centroid Y'
            is_micron = False
        else:
            # 检查单位：如果原始名称中包含 'µm' 或 '碌m' 或 'um'，则认为是微米单位
            is_micron = 'µm' in centroid_x_orig_name or '碌m' in centroid_x_orig_name or 'um' in centroid_x_orig_name.lower()

        # 2. 单位转换：如果是微米，除以 MPP；否则，因子为 1.0 (已经是像素)
        conversion_factor = (1.0 / PIXEL_SIZE_MPP) if is_micron else 1.0

        cell_data['Centroid X aligned_float'] = cell_data[centroid_x_orig_name].astype(float) * conversion_factor
        cell_data['Centroid Y aligned_float'] = cell_data[centroid_y_orig_name].astype(float) * conversion_factor

        # 3. ROI 偏移校正 (WSI 坐标 -> ROI 局部坐标)
        cell_data['Centroid X aligned_float'] -= ROI_OFFSET_X_PX
        cell_data['Centroid Y aligned_float'] -= ROI_OFFSET_Y_PX

        # 4. 舍入并设置最终的 Centroid X/Y 列 (FIX 3: 最终对齐的坐标必须是整数)
        cell_data['Centroid X'] = cell_data['Centroid X aligned_float'].round(0).astype(int)
        cell_data['Centroid Y'] = cell_data['Centroid Y aligned_float'].round(0).astype(int)

        # ====================================================================
        # ✨ 标准化其他列名
        # ====================================================================
        # 保留您的列名清理逻辑，但只针对非 Centroid 列
        current_cols = cell_data.columns.tolist()
        new_columns = []
        for col in current_cols:
            if col in ['Centroid X', 'Centroid Y']:
                new_columns.append(col)
                continue

            # 清理所有非字母数字字符（包括空格、损坏的微米符号等）
            cleaned_col = re.sub(r'[^a-zA-Z0-9]+', '', col).strip()

            if 'NucleusArea' in cleaned_col:
                new_columns.append('Nucleus: Area')
            else:
                new_columns.append(col)  # 保留其他列名

        cell_data.columns = new_columns
        # ====================================================================

    except Exception as e:
        print(f"❌ Failed to read cell data from {cell_path.name}. Error: {e}")
        return None

    # Required columns for classification/visualization
    required_cols = ['Centroid X', 'Centroid Y', 'Nucleus: Area']

    if not all(col in cell_data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in cell_data.columns]
        print(f"❌ Cell data file missing required columns: {missing}")
        return None

    # Data cleaning and range filtering (using actual image dimensions W and H)
    cell_data = cell_data.dropna(subset=required_cols)
    initial_count = len(cell_data)

    # Filter out centroids outside the image dimensions
    cell_data = cell_data[
        (cell_data['Centroid X'] >= 0) & (cell_data['Centroid X'] < W) &
        (cell_data['Centroid Y'] >= 0) & (cell_data['Centroid Y'] < H)
        ].copy()

    filtered_count = len(cell_data)
    print(f"✅ Loaded {initial_count} cells, kept {filtered_count} cells after filtering.")

    return cell_data



# =========================================================================
# === 1. V9.0 CLASSIFICATION RULES & LOGIC ===
# =========================================================================

CELL_CLASSIFICATION_THRESHOLD = 0.2  # Threshold for positivity
MIN_POS_PROPORTION_V9 = 0.45
MIN_CLASSIFICATION_SCORE_V9 = 0.05

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

CLASSIFICATION_COLORS_MAP = {
    # T cell subsets
    "Treg": '#800080',               # Purple – Regulatory T cells
    "CD8+ T Cell": '#00FF00',        # Bright Green – Cytotoxic T cells
    "CD4+ T Cell": '#00FFFF',        # Cyan – Helper T cells

    # B and NK cells
    "B Cell": '#FF69B4',             # Hot Pink – B cells
    "NK Cell": '#90EE90',            # Light Green – Natural Killer cells

    # Myeloid lineage
    "Neutrophil": '#DAA520',         # Goldenrod – Neutrophils
    "Macrophage (CD163+)": '#FFA500',# Orange – M2-like macrophages
    "Myeloid Cell (CD68+)": '#CD853F',# Peru – General macrophages
    "Dendritic Cell": '#F4A460',     # Sandy Brown – Dendritic cells
    "Monocyte/MDSC-like": '#A0522D', # Sienna – Suppressive myeloid cells

    # Stromal / Non-immune
    "Epithelial Cell": '#FF00FF',    # Magenta – Epithelial cells
    "Fibroblast": '#A9A9A9',         # Dark Gray – Fibroblasts
    "Endothelial Cell": '#0000FF',   # Blue – Endothelial cells

    # Fallback / ambiguous
    "Non_Immune_Unspec": '#D3D3D3',  # Light Gray – Unspecified non-immune
    "Background/Junk": '#FFFFFF',    # White – Background or noise
}


# Create a mapping from marker name to the target 10-channel index
MARKER_TO_INDEX = {marker: i for i, marker in enumerate(SELECTED_MARKERS)}


def classify_and_export_cells_to_json(
        cell_data: pd.DataFrame, prediction_10ch: np.ndarray, markers: list,
        marker_to_index: dict, output_path_json: Path
) -> None:
    """Classifies cells based on prediction signals and exports data to JSON."""
    H, W, C = prediction_10ch.shape
    classified_cells = []
    for index, row in cell_data.iterrows():
        cx, cy = int(row['Centroid X']), int(row['Centroid Y'])
        if not (0 <= cy < H and 0 <= cx < W): continue
        cell_signals = prediction_10ch[cy, cx, :]
        max_sig = np.max(cell_signals)
        norm_signals = cell_signals / max_sig if max_sig > 0 else cell_signals
        best_match = "Unclassified"
        for cell_type, rules in CLASSIFICATION_RULES.items():
            is_match = True
            for pos_marker in rules.get("pos", []):
                idx = marker_to_index.get(pos_marker)
                if idx is not None and norm_signals[idx] < CELL_CLASSIFICATION_THRESHOLD:
                    is_match = False;
                    break
            if not is_match: continue
            for neg_marker in rules.get("neg", []):
                idx = marker_to_index.get(neg_marker)
                if idx is not None and norm_signals[idx] >= CELL_CLASSIFICATION_THRESHOLD:
                    is_match = False;
                    break
            if is_match: best_match = cell_type; break
        cell_output = {
            "Centroid X": cx, "Centroid Y": cy,
            "Nucleus Area": float(row['Nucleus: Area']),
            "Classification": best_match,
            "Signals": {markers[i]: round(float(norm_signals[i]), 4) for i in range(C)}
        }
        classified_cells.append(cell_output)
    try:
        with open(output_path_json, 'w') as f:
            json.dump(classified_cells, f, indent=2)
        print(f"✅ Saved {len(classified_cells)} classified cell annotations to {output_path_json}")
    except Exception as e:
        print(f"❌ Error exporting JSON: {e}")


def create_visualization_and_save_rgb(
        prediction_50ch: np.ndarray,
        markers: list,
        colors: np.ndarray,
        output_path_png: Path,
        cell_data: Optional[pd.DataFrame] = None
) -> None:
    """
    Creates an RGB image where color is determined by the most dominant marker
    and includes cell centroids (colored by their classification) and a legend.
    """
    H, W, C = prediction_50ch.shape
    if C != 50:
        print(f"❌ Visualization failed: Prediction has {C} channels, not 50.")
        return

    # --- 1. Create Color Composite Image based on Dominant Channel (PIXEL COLOR) ---
    global_max = np.max(prediction_50ch)
    if global_max == 0:
        print("❌ Visualization failed: All prediction values are zero.")
        return
    normalized_pred = prediction_50ch / global_max
    max_channel_index = np.argmax(normalized_pred, axis=-1)
    rgb_image = np.zeros((H, W, 3), dtype=np.float32)

    for i in range(C):
        mask = (max_channel_index == i)
        rgb_image[mask] = normalized_pred[mask, i][:, np.newaxis] * colors[i]

    vmax = np.percentile(rgb_image, 99.5)
    if vmax > 0:
        rgb_image = np.clip(rgb_image / vmax, 0, 1)

    # --- 2. Create Matplotlib Figure ---
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150, facecolor='black')
    ax.imshow(rgb_image)
    ax.axis('off')

    # --- 3. Overlay Cell Centroids (CENTROID COLOR) ---
    if cell_data is not None and not cell_data.empty:
        try:
            if 'Classification' not in cell_data.columns:
                print("⚠️ Warning: 'Classification' column missing in cell_data. Skipping classification colors.")
                cell_colors = 'lime'
            else:
                cell_colors = cell_data['Classification'].map(
                    lambda x: CLASSIFICATION_COLORS_MAP.get(x, 'white')
                ).to_numpy()

            ax.scatter(
                cell_data['Centroid X'], cell_data['Centroid Y'],
                s=cell_data['Nucleus: Area'] * 0.05,
                c=cell_colors,
                marker='o',
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5
            )
            print(f"✅ Overlaid {len(cell_data)} cell centroids (colored by classification) on PNG.")

            # Add a separate legend for cell classifications
            classification_legend_elements = [
                Patch(facecolor=color, edgecolor='black', label=label)
                for label, color in CLASSIFICATION_COLORS_MAP.items()
                if label in cell_data['Classification'].unique()
            ]
            if classification_legend_elements:
                ax.legend(handles=classification_legend_elements, loc='upper left', bbox_to_anchor=(-0.15, 1.0),
                          title="Cell Classifications", title_fontsize='medium', labelcolor='white', facecolor='black',
                          edgecolor='grey', framealpha=0.8, fontsize='small')
        except KeyError as e:
            print(f"⚠️ Error drawing cell centroids: Required column missing ({e})")
        except Exception as e:
            print(f"⚠️ Error drawing cell centroids: {e}")

    # --- 4. Add Marker Legend (PIXEL COLOR LEGEND) ---
    try:
        legend_elements = [Patch(facecolor=TARGET_CHANNEL_COLORS[marker], edgecolor='w', label=marker) for marker in
                           markers]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.25, 1.0), title="Dominant Marker",
                  title_fontsize='medium', labelcolor='white', facecolor='black', edgecolor='grey', framealpha=0.8,
                  fontsize='small')
    except NameError:
        print("⚠️ Warning: TARGET_CHANNEL_COLORS is not defined, skipping marker legend.")

    # --- 5. Save the Figure ---
    plt.savefig(output_path_png, bbox_inches='tight', pad_inches=0.1, dpi=300, facecolor='black')
    plt.close(fig)
    print(f"✅ Saved visualization to {output_path_png}")


# =======================================================================
# ✨ CORE PROCESSING FUNCTION
# =======================================================================

def process_image(model: nn.Module,
                  image_path: str,
                  output_path_dir: str,
                  device: torch.device,
                  num_model_channels: int,
                  stride_size: int,
                  exclude_background: bool = True,
                  apply_border_threshold: bool = False,
                  smooth_sigma: float = 1.0,
                  postprocess_image: bool = True,
                  target_indices: list = TARGET_INDICES,
                  target_markers: list = SELECTED_MARKERS):
    """
    Process a single H&E image using sliding window inference with weighted blending,
    then save predictions and cell annotations.
    """

    # ------------------ Path Setup ------------------
    output_path_obj = Path(output_path_dir)
    image_path_obj = Path(image_path)
    stem_name = image_path_obj.stem

    output_path_tiff_50ch = output_path_obj / (stem_name + "_50ch.tiff")
    output_path_png = output_path_obj / (stem_name + "_overlay.png")
    output_path_json = output_path_obj / (stem_name + "_cells.json")

    # ------------------ Data and Patch Setup (Uses ImageDataset) ------------------
    transform = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),  # Converts HWC, uint8 to CHW, float in [0.0, 1.0]
            transforms.Resize(224, interpolation=InterpolationMode.BILINEAR, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Use smaller stride for overlapping patches, as per reference script
    overlap_stride = max(1, stride_size // 2)

    try:
        dataset = ImageDataset(image_path, transform=transform, stride_size=overlap_stride,
                               exclude_background=exclude_background)
    except ValueError as e:
        print(f"❌ Error loading image or dataset: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # Get image dimensions from dataset
    height, width = dataset.he_zarr[0].shape
    H, W = height, width
    print(f"🖼️ Image dimensions (H, W): {H}, {W}")

    # ------------------ Inference Setup ------------------
    # Initialize output array and weight array for weighted blending
    raw_output_50ch = np.zeros((num_model_channels, H, W), dtype=np.float32)
    weight_map = np.zeros((H, W), dtype=np.float32)

    # Create tissue mask for postprocessing
    tissue_mask = create_tissue_mask(dataset.he_zarr)

    # Create a Gaussian weight kernel for smoother blending
    kernel_size = stride_size * 2
    y_k, x_k = np.mgrid[0:kernel_size, 0:kernel_size]
    center = kernel_size // 2
    # Standard deviation set to quarter of the kernel size, as in the reference script
    weight_kernel = np.exp(-((x_k - center) ** 2 + (y_k - center) ** 2) / (2 * (kernel_size / 4) ** 2))

    # ------------------ Inference Loop ------------------
    model.eval()
    model.to(device)

    autocast_context = torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda')

    with torch.no_grad():
        with autocast_context:
            for patches, X, Y in tqdm(dataloader, desc="Running Inference"):
                patches = patches.to(device)
                predictions = model(patches)

                # Fill in predictions with weighted blending
                for pred, x, y in zip(predictions, X.cpu().numpy(), Y.cpu().numpy()):
                    pred = pred.detach().cpu().numpy()  # shape (50,) for classifier model

                    # Center of the current patch window (for blending)
                    half_size = kernel_size // 2
                    t = np.clip(y - half_size, 0, H)
                    b = np.clip(y + half_size, 0, H)
                    l = np.clip(x - half_size, 0, W)
                    r = np.clip(x + half_size, 0, W)

                    # Get the portion of the weight kernel that fits the image bounds
                    kernel_h, kernel_w = b - t, r - l
                    weight = weight_kernel[:kernel_h, :kernel_w]

                    # --- ADAPTATION FOR CLASSIFIER MODEL OUTPUT (Batch, 50) ---
                    # Tile the 50-channel vector output to a spatial map for blending
                    if pred.ndim == 1:
                        pred_spatial = pred[:, np.newaxis, np.newaxis]
                        # Create a spatial prediction map for blending (50, H_blend, W_blend)
                        pred_map = np.tile(pred_spatial, (1, kernel_h, kernel_w))
                    elif pred.ndim == 3:
                        # Crop spatial map output to the kernel size
                        pred_h, pred_w = pred.shape[1], pred.shape[2]
                        center_h, center_w = pred_h // 2, pred_w // 2
                        half_k = kernel_size // 2
                        pred_map_l = center_w - half_k
                        pred_map_r = center_w + half_k
                        pred_map_t = center_h - half_k
                        pred_map_b = center_h + half_k
                        pred_map = pred[:, pred_map_t:pred_map_b, pred_map_l:pred_map_r]
                        pred_map = pred_map[:, :kernel_h, :kernel_w]
                    else:
                        raise RuntimeError(f"Unexpected prediction shape: {pred.shape}. Expected (50,) or (50, H, W).")

                    for c in range(num_model_channels):
                        raw_output_50ch[c, t:b, l:r] += pred_map[c] * weight

                    # Add weights to the weight map
                    weight_map[t:b, l:r] += weight

    # ------------------ Finalize Predictions ------------------
    print("Post-processing predictions...")

    # Normalize by weights
    weight_map = np.maximum(weight_map, 1e-8)
    for c in range(num_model_channels):
        raw_output_50ch[c] /= weight_map

    # Apply postprocessing (thresholding, normalization, box blur, masking)
    if postprocess_image:
        processed_output_50ch = postprocess_predictions(raw_output_50ch, tissue_mask,
                                                        apply_border_threshold=apply_border_threshold)
    else:
        # Simple normalization to 0-255 based on 99.9 percentile
        max_val_50ch = np.percentile(raw_output_50ch.flatten(), 99.9)
        if max_val_50ch > 0:
            processed_output_50ch = np.clip(raw_output_50ch / max_val_50ch * 255, 0, 255).astype(np.uint8)
        else:
            processed_output_50ch = raw_output_50ch.astype(np.uint8)

    # ------------------ Slice and Save 10 Target Channels ------------------
    # Reshape (C, H, W) to (H, W, C) for slicing and visualization
    processed_output_50ch_HWC = processed_output_50ch.transpose(1, 2, 0)

    # Slice to the 10 target channels (H, W, 10)
    prediction_50ch_HWC = processed_output_50ch_HWC[..., target_indices]

    # Save 10-channel prediction as TIFF (C, H, W)
    prediction_50ch_CHW = prediction_50ch_HWC.transpose(2, 0, 1)
    tifffile.imwrite(output_path_tiff_50ch, prediction_50ch_CHW, compression=TIFF_COMPRESSION)
    print(f"✅ Saved 10-channel prediction to {output_path_tiff_50ch} (Shape: {prediction_50ch_CHW.shape})")

    # ------------------ Load Cell Data and Classify ------------------
    cell_data = load_and_process_cell_data(image_path, W, H, target_markers)

    if cell_data is not None and not cell_data.empty:
        # Classification and JSON export
        classify_and_export_cells_to_json(
            cell_data=cell_data.copy(),  # Pass a copy to avoid unexpected mutation if JSON loading fails
            prediction_10ch=prediction_50ch_HWC,
            markers=target_markers,
            marker_to_index=MARKER_TO_INDEX,
            output_path_json=output_path_json
        )

        # Load the exported JSON to get the classifications back for visualization
        try:
            with open(output_path_json, 'r') as f:
                json_data = json.load(f)
            classified_df = pd.DataFrame(json_data)
            # Merge classification column back into original cell_data based on Centroid X/Y
            if classified_df is not None:
                print("✅ Loaded classifications back for visualization.")

                # ----------------------------------------------------
                # 修正 2：消除浮点数/整数合并 UserWarning
                # 强制将质心坐标四舍五入到最近的整数，并转换为整数类型 (int)
                # 这消除了浮点数精度不匹配导致的 UserWarning 和潜在的合并失败。
                # ----------------------------------------------------

                # 转换原始数据 (cell_data) 的合并键
                cell_data['Centroid X'] = cell_data['Centroid X'].round(0).astype(int)
                cell_data['Centroid Y'] = cell_data['Centroid Y'].round(0).astype(int)

                # 转换分类结果 (classified_df) 的合并键
                classified_df['Centroid X'] = classified_df['Centroid X'].round(0).astype(int)
                classified_df['Centroid Y'] = classified_df['Centroid Y'].round(0).astype(int)

            cell_data = cell_data.merge(
                classified_df[['Centroid X', 'Centroid Y', 'Classification']],
                on=['Centroid X', 'Centroid Y'],
                how='left',
                suffixes=('', '_classified')  # 添加后缀以避免覆盖原始列
            )
            print("✅ Loaded classifications back for visualization.")
        except Exception as e:
            print(
                f"⚠️ Could not re-load classified JSON for visualization: {e}. Plotting will default to 'Unclassified'.")
            if 'Classification' not in cell_data.columns:
                cell_data['Classification'] = 'Unclassified'

    # ------------------ Visualization ------------------
    create_visualization_and_save_rgb(
        prediction_50ch=prediction_50ch_HWC,
        markers=target_markers,
        colors=COLOR_RGBS,
        output_path_png=output_path_png,
        cell_data=cell_data
    )
    print(f"--- Finished processing {stem_name} ---")


# =======================================================================
# ✨ MAIN EXECUTION BLOCK
# =======================================================================

def main():
    parser = argparse.ArgumentParser(description='Run inference on H&E images')

    # Arguments are configured to the user's specified defaults
    parser.add_argument('--input_dir', type=str, default='Normal_ROI', help='Path to input image(s).')
    parser.add_argument('--output_dir', type=str, default='Normal_full_tiff_50ch', help='Path to output directory.')
    parser.add_argument('--model_path', type=str, default='best_model_single.pth',
                        help='Path to the trained model (.pth).')

    parser.add_argument('--stride_size', type=int, default=32,
                        help='Stride size for sliding window inference. (Overlap stride is derived from this).')
    parser.add_argument('--exclude_background', action='store_true', default=True,
                        help='Exclude white background patches (default: True).')
    parser.add_argument('--apply_border_threshold', action='store_true', default=False,
                        help='Use border regions for background thresholding in postprocessing (default: False).')
    parser.add_argument('--smooth_sigma', type=float, default=2.0,
                        help='Sigma for Gaussian smoothing (0 to disable, applied to 50ch output).')
    parser.add_argument('--postprocess_image', action='store_true', default=False,
                        help='Whether to apply full postprocessing (thresholding, normalization, box blur).')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    num_model_channels = NUM_MODEL_CHANNELS  # 50 channels
    model = get_model(num_outputs=num_model_channels)

    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        model = nn.DataParallel(model)

    try:
        # Load state dict
        # CRITICAL FIX: Add weights_only=False to bypass new PyTorch safety check for older checkpoints
        state_dict = torch.load(args.model_path, map_location=device, weights_only=False)

        if 'model_state_dict' in state_dict:
            # If it's a DataParallel model, ensure we load it correctly
            if isinstance(model, nn.DataParallel):
                # DataParallel state dicts often have 'module.' prefix
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict['model_state_dict'].items()}
                model.module.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict['model_state_dict'])

        else:
            model.load_state_dict(state_dict)  # Assume it's a raw state dict if key is missing
        print(f"✅ Loaded model weights from {args.model_path}")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}. Please check --model_path.")
        return

    # Determine image paths
    input_path = Path(args.input_dir)
    image_paths = []
    if input_path.is_file():
        image_paths.append(input_path)
    elif input_path.is_dir():
        for ext in ['*.tiff', '*.tif', '*.png', '*.jpg', '*.jpeg', '*.svs', '*.ome.zarr']:
            image_paths.extend(input_path.glob(f'**/{ext}'))

    if not image_paths:
        print(f"❌ No images found in {args.input_dir}")
        return

    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        process_image(
            model, str(image_path), str(output_dir), device, num_model_channels,
            args.stride_size, args.exclude_background, args.apply_border_threshold,
            args.smooth_sigma, args.postprocess_image, TARGET_INDICES, SELECTED_MARKERS
        )


if __name__ == '__main__':
    main()