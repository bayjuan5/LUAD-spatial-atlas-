"""
Timed Petri Net Simulation for Cell Type Dynamics in Lung Adenocarcinoma Progression
WITH REAL EXPERIMENTAL DATA LOADING

This module models temporal evolution of cell populations across five pathological states:
Normal → AAH → AIS → MIA → IAC
"""

import json
import os
from typing import Dict, Any, List
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = True
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300


# ============================================================================
# REAL DATA LOADING FUNCTIONS
# ============================================================================

def load_all_cell_data(root_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Recursively loads all cell objects from JSON files and assigns a 'category'
    based on keywords in the file path/name.

    Returns a dictionary with categories as keys: {Normal, AAH, AIS, MIA, IAC, Other}
    and lists of cell dictionaries as values.
    """
    cells_by_category = defaultdict(list)
    CATEGORIES = ["Normal", "AAH", "AIS", "MIA", "IAC"]
    print(f"--- 🚀 Starting to load and categorize cell data from '{root_dir}' ---")

    for dirpath, _, filenames in os.walk(root_dir):
        json_files = [f for f in filenames if f.endswith('.json')]

        if json_files:
            # Determine category from directory path
            path_components = os.path.normpath(dirpath).split(os.sep)
            category = "Other"

            # Check for category keyword in dirpath
            for cat_keyword in CATEGORIES:
                if cat_keyword in path_components:
                    category = cat_keyword
                    break

            for filename in json_files:
                file_path = os.path.join(dirpath, filename)

                # If category is still 'Other', check the filename
                if category == "Other":
                    for cat_keyword in CATEGORIES:
                        if cat_keyword in filename:
                            category = cat_keyword
                            break

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    if isinstance(data, list):
                        # Add the category to each cell dictionary
                        for cell in data:
                            cell['category'] = category
                            cells_by_category[category].append(cell)

                except Exception as e:
                    print(f"  ❌ Error reading {file_path}: {e}")

    # Convert defaultdict to regular dict
    cells_by_category = dict(cells_by_category)

    print(f"--- ✅ Data loading complete ---")
    # Print category counts
    for category in ["Normal", "AAH", "AIS", "MIA", "IAC", "Other"]:
        count = len(cells_by_category.get(category, []))
        if count > 0:
            print(f"  {category}: {count} cells")

    return cells_by_category


# ============================================================================
# PETRI NET MODEL CLASS
# ============================================================================

class TimedPetriNet:
    """Timed Petri Net Model for Cell Dynamics in Lung Adenocarcinoma Progression"""

    def __init__(self):
        """Initialize the Petri Net model with state space and cell types"""
        self.states = ['Normal', 'AAH', 'AIS', 'MIA', 'IAC']

        self.cell_types = [
            'Treg', 'CD8+ T Cell', 'CD4+ T Cell', 'B Cell', 'NK Cell',
            'Neutrophil', 'Macrophage (CD163+)', 'Myeloid Cell (CD68+)',
            'Dendritic Cell', 'Monocyte/MDSC-like',
            'Epithelial Cell', 'Fibroblast', 'Endothelial Cell', 'Non_Immune_Unspec'
        ]

        # Data containers
        self.cell_counts = {}
        self.biomarker_data = {}

        # Classification rules
        self.rules = {
            "Treg": {"pos": ["CD45", "CD3e", "CD4", "FoxP3"], "neg": ["CD8", "CD20", "CD68", "PanCK"]},
            "CD8+ T Cell": {"pos": ["CD45", "CD3e", "CD8"], "neg": ["CD4", "FoxP3", "CD20", "CD68", "PanCK"]},
            "CD4+ T Cell": {"pos": ["CD45", "CD3e", "CD4"], "neg": ["CD8", "FoxP3", "CD20", "CD68", "PanCK"]},
            "B Cell": {"pos": ["CD45", "CD20"], "neg": ["CD3e", "CD68", "CD163", "CD66", "CD11c", "PanCK"]},
            "NK Cell": {"pos": ["CD45"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD66", "CD11c", "PanCK", "CD31"]},
            "Neutrophil": {"pos": ["CD45", "CD66"], "neg": ["CD3e", "CD20", "CD68", "CD163", "CD11c", "PanCK"]},
            "Macrophage (CD163+)": {"pos": ["CD45", "CD163"], "neg": ["CD3e", "CD20", "CD66", "CD11c", "PanCK"]},
            "Myeloid Cell (CD68+)": {"pos": ["CD45", "CD68"],
                                     "neg": ["CD3e", "CD20", "CD66", "CD163", "CD11c", "PanCK"]},
            "Dendritic Cell": {"pos": ["CD45", "CD11c"], "neg": ["CD3e", "CD20", "CD66", "CD68", "CD163", "PanCK"]},
            "Monocyte/MDSC-like": {"pos": ["CD45", "CD68"], "neg": ["CD3e", "CD20", "CD31", "PanCK", "Vimentin"]},
            "Epithelial Cell": {"pos": ["PanCK"], "neg": ["CD45", "Vimentin"]},
            "Fibroblast": {"pos": ["Vimentin"], "neg": ["CD45", "PanCK"]},
            "Endothelial Cell": {"pos": ["CD31"], "neg": ["CD45", "PanCK", "Vimentin"]},
            "Non_Immune_Unspec": {"pos": [], "neg": ["CD45", "PanCK", "Vimentin", "CD31"]}
        }

    def _get_total_cells(self, state):
        """Calculates the total number of cells for a given state."""
        return sum(self.cell_counts.get(state, {}).values())

    def _print_state_summary(self, state, total_cells):
        """Print cell type distribution for a state"""
        print(f"\n✓ State '{state}': {total_cells} total cells")
        for ct in self.cell_types:
            count = self.cell_counts[state].get(ct, 0)
            if count > 0:
                pct = (count / total_cells) * 100 if total_cells > 0 else 0
                print(f"    {ct:25s}: {count:5d} cells ({pct:5.1f}%)")

    def load_real_data_from_dict(self, cells_by_category):
        """Load experimental cell count data from grouped cell dictionaries."""
        print("=" * 80)
        print("LOADING REAL EXPERIMENTAL CELL DATA")
        print("=" * 80)

        self.cell_counts = {}
        self.biomarker_data = {}

        for state in self.states:
            self.cell_counts[state] = {ct: 0 for ct in self.cell_types}
            self.biomarker_data[state] = {ct: [] for ct in self.cell_types}

            if state not in cells_by_category or not cells_by_category[state]:
                print(f"⚠ Warning: No data found for state '{state}'")
                continue

            cells = cells_by_category[state]
            total_cells = len(cells)

            # Count and store biomarker data
            for cell in cells:
                cell_type = cell.get('cell_type', cell.get('Classification', 'Non_Immune_Unspec'))
                if cell_type in self.cell_counts[state]:
                    self.cell_counts[state][cell_type] += 1
                    signals = cell.get('Signals', {})
                    self.biomarker_data[state][cell_type].append(signals)

            self._print_state_summary(state, total_cells)

        print("=" * 80 + "\n")

    def calculate_change_rate(self, state1, state2, cell_type):
        """Calculate percentage change rate between two states"""
        count1 = self.cell_counts.get(state1, {}).get(cell_type, 0)
        count2 = self.cell_counts.get(state2, {}).get(cell_type, 0)
        if count1 == 0:
            return 0.0
        return ((count2 - count1) / count1) * 100

    def get_change_matrix(self):
        """Generate change rate matrix across all transitions"""
        transitions = [f"{self.states[i]} → {self.states[i + 1]}" for i in range(len(self.states) - 1)]
        data = []
        for ct in self.cell_types:
            row = [ct]
            for i in range(len(self.states) - 1):
                rate = self.calculate_change_rate(self.states[i], self.states[i + 1], ct)
                row.append(f"{rate:+.2f}%")
            data.append(row)
        return pd.DataFrame(data, columns=['Cell Type'] + transitions)

    def get_trend_data(self):
        """Extract cell count trends across all states"""
        data = []
        for state in self.states:
            row = {'State': state}
            for ct in self.cell_types:
                row[ct] = self.cell_counts[state].get(ct, 0)
            data.append(row)
        return pd.DataFrame(data)

    def print_summary(self):
        """Print the model summary."""
        print("=" * 80)
        print("Timed Petri Net Model Summary")
        print("=" * 80)
        print(f"State Space: {self.states}")
        print(f"Number of Cell Types: {len(self.cell_types)}")
        print(f"Number of Classification Rules: {len(self.rules)}")
        print("\nTotal Cell Count per State:")
        for state in self.states:
            total = self._get_total_cells(state)
            print(f"  {state}: {total} cells")
        print("=" * 80)

    def plot_petri_net(self, current_state=0, save_path=None):
        """Visualize enhanced Petri Net state transition diagram with detailed cell information"""
        fig, ax = plt.subplots(figsize=(24, 14))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.axis('off')
        x_positions = np.linspace(2, 14, len(self.states))

        # Color scheme for different cell categories
        immune_color = '#3498db'  # Blue
        structural_color = '#e74c3c'  # Red
        myeloid_color = '#f39c12'  # Orange

        for idx, (state, x) in enumerate(zip(self.states, x_positions)):
            # State circle styling
            if idx == current_state:
                color, ec, lw = '#e3f2fd', '#1976d2', 4
            elif idx < current_state:
                color, ec, lw = '#c8e6c9', '#388e3c', 4
            else:
                color, ec, lw = '#f5f5f5', '#9e9e9e', 3

            # Draw main state circle
            circle_radius = 0.5
            ax.add_patch(Circle((x, 6), circle_radius, color=color, ec=ec, linewidth=lw, zorder=3))

            # Add token if active
            if idx <= current_state:
                token_color = '#0d47a1' if idx == current_state else '#1b5e20'
                ax.add_patch(Circle((x, 6), 0.12, color=token_color, zorder=4))

            # State name and total count
            ax.text(x, 7.2, state, ha='center', fontsize=18, fontweight='bold', zorder=5)
            total_cells = self._get_total_cells(state)
            ax.text(x, 6.8, f'n = {total_cells:,}', ha='center', fontsize=13,
                    color='#424242', fontweight='bold', zorder=5)

            # Get top 5 cell types for this state
            state_counts = self.cell_counts.get(state, {})
            sorted_cells = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            # Display top cell types below the circle
            y_start = 5.2
            for i, (cell_type, count) in enumerate(sorted_cells):
                y_pos = y_start - (i * 0.35)

                # Color code by cell category
                if cell_type in ['Treg', 'CD8+ T Cell', 'CD4+ T Cell', 'B Cell', 'NK Cell']:
                    cell_color = immune_color
                elif cell_type in ['Epithelial Cell', 'Fibroblast', 'Endothelial Cell']:
                    cell_color = structural_color
                else:
                    cell_color = myeloid_color

                # Short name for display
                short_name = cell_type.replace('Macrophage (CD163+)', 'Mac CD163+')
                short_name = short_name.replace('Myeloid Cell (CD68+)', 'Mye CD68+')
                short_name = short_name.replace('Monocyte/MDSC-like', 'Mono/MDSC')

                pct = (count / total_cells * 100) if total_cells > 0 else 0
                ax.text(x, y_pos, f'{short_name[:15]}', ha='center', fontsize=9,
                        color=cell_color, fontweight='bold', zorder=5)
                ax.text(x, y_pos - 0.15, f'{count:,} ({pct:.1f}%)', ha='center',
                        fontsize=8, color='#616161', zorder=5)

            # Draw transition arrows and boxes
            if idx < len(self.states) - 1:
                tx = x + 1.2

                # Transition box
                t_color = '#a5d6a7' if idx < current_state else '#eeeeee'
                t_edge = '#2e7d32' if idx < current_state else '#757575'
                ax.add_patch(Rectangle((tx - 0.2, 5.85), 0.4, 0.4, color=t_color,
                                       ec=t_edge, linewidth=2.5, zorder=3))
                ax.text(tx, 6.05, f't{idx + 1}', ha='center', va='center',
                        fontsize=13, fontweight='bold', zorder=4)

                # Calculate change rate for most significant cell type
                next_state = self.states[idx + 1]
                max_change = 0
                max_change_cell = ''
                for cell_type in self.cell_types:
                    change = self.calculate_change_rate(state, next_state, cell_type)
                    if abs(change) > abs(max_change):
                        max_change = change
                        max_change_cell = cell_type

                # Display key transition info
                if max_change_cell:
                    change_color = '#d32f2f' if max_change < 0 else '#388e3c'
                    arrow = '↓' if max_change < 0 else '↑'
                    short_cell = max_change_cell[:12]
                    ax.text(tx, 5.4, f'{short_cell}', ha='center', fontsize=8,
                            color='#424242', style='italic', zorder=5)
                    ax.text(tx, 5.1, f'{arrow}{abs(max_change):.0f}%', ha='center',
                            fontsize=9, color=change_color, fontweight='bold', zorder=5)

        # Add legend
        legend_x = 1.5
        legend_y = 9
        ax.text(legend_x, legend_y, 'Cell Type Categories:', fontsize=12, fontweight='bold')

        categories = [
            ('Immune Cells (T, B, NK)', immune_color),
            ('Structural Cells', structural_color),
            ('Myeloid Cells', myeloid_color)
        ]

        for i, (label, color) in enumerate(categories):
            y = legend_y - 0.4 - (i * 0.4)
            ax.add_patch(Rectangle((legend_x - 0.1, y - 0.1), 0.3, 0.2,
                                   color=color, alpha=0.7))
            ax.text(legend_x + 0.3, y, label, fontsize=10, va='center')

        # Add title with subtitle
        fig.text(0.5, 0.96, 'Timed Petri Net: Lung Adenocarcinoma Progression Model',
                 ha='center', fontsize=22, fontweight='bold')
        fig.text(0.5, 0.93, 'Showing top 5 cell populations per stage with transition dynamics',
                 ha='center', fontsize=14, style='italic', color='#616161')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def plot_trend_lines(self, selected_cell_types=None, save_path=None):
        """Plot cell type evolution trends with enhanced visualization"""
        df = self.get_trend_data()

        # If no selection, show ALL cell types
        if selected_cell_types is None:
            selected_cell_types = self.cell_types

        # Create figure with subplots: main plot + individual mini plots
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.3)

        # Main comprehensive plot
        ax_main = fig.add_subplot(gs[0])
        colors = plt.cm.tab20(np.linspace(0, 1, len(selected_cell_types)))

        for idx, ct in enumerate(selected_cell_types):
            values = df[ct].values
            ax_main.plot(df['State'], values, marker='o', linewidth=2.5,
                         label=ct, color=colors[idx], markersize=8,
                         markeredgecolor='white', markeredgewidth=1.5, alpha=0.8)

            # Add value labels at each point
            for i, val in enumerate(values):
                if val > 0:  # Only show non-zero values
                    ax_main.annotate(f'{int(val)}',
                                     xy=(i, val),
                                     xytext=(0, 8),
                                     textcoords='offset points',
                                     ha='center',
                                     fontsize=7,
                                     alpha=0.6)

        ax_main.set_xlabel('Disease Stage', fontsize=14, fontweight='bold', labelpad=10)
        ax_main.set_ylabel('Cell Count', fontsize=14, fontweight='bold', labelpad=10)
        ax_main.set_title('Cell Type Evolution Across Disease Progression',
                          fontsize=18, fontweight='bold', pad=15)
        ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
                       frameon=True, shadow=True, ncol=1)
        ax_main.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.set_ylim(bottom=0)

        # Subplot 1: Immune cells comparison
        ax_immune = fig.add_subplot(gs[1])
        immune_cells = ['Treg', 'CD8+ T Cell', 'CD4+ T Cell', 'B Cell', 'NK Cell',
                        'Neutrophil', 'Macrophage (CD163+)', 'Myeloid Cell (CD68+)']
        immune_colors = plt.cm.Set2(np.linspace(0, 1, len(immune_cells)))

        for idx, ct in enumerate(immune_cells):
            if ct in selected_cell_types:
                ax_immune.plot(df['State'], df[ct], marker='s', linewidth=2,
                               label=ct, color=immune_colors[idx], markersize=7, alpha=0.8)

        ax_immune.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
        ax_immune.set_title('Immune Cell Populations', fontsize=14, fontweight='bold')
        ax_immune.legend(loc='best', fontsize=8, ncol=2, frameon=True)
        ax_immune.grid(True, alpha=0.2, linestyle='--')
        ax_immune.spines['top'].set_visible(False)
        ax_immune.spines['right'].set_visible(False)
        ax_immune.set_ylim(bottom=0)

        # Subplot 2: Structural cells comparison
        ax_structural = fig.add_subplot(gs[2])
        structural_cells = ['Epithelial Cell', 'Fibroblast', 'Endothelial Cell',
                            'Dendritic Cell', 'Monocyte/MDSC-like']
        structural_colors = plt.cm.Set3(np.linspace(0, 1, len(structural_cells)))

        for idx, ct in enumerate(structural_cells):
            if ct in selected_cell_types:
                ax_structural.plot(df['State'], df[ct], marker='^', linewidth=2,
                                   label=ct, color=structural_colors[idx], markersize=7, alpha=0.8)

        ax_structural.set_xlabel('Disease Stage', fontsize=12, fontweight='bold')
        ax_structural.set_ylabel('Cell Count', fontsize=12, fontweight='bold')
        ax_structural.set_title('Structural & Myeloid Cell Populations', fontsize=14, fontweight='bold')
        ax_structural.legend(loc='best', fontsize=8, ncol=2, frameon=True)
        ax_structural.grid(True, alpha=0.2, linestyle='--')
        ax_structural.spines['top'].set_visible(False)
        ax_structural.spines['right'].set_visible(False)
        ax_structural.set_ylim(bottom=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def plot_bar_comparison(self, state='Normal', save_path=None):
        """Plot the cell type distribution bar chart for a specified state"""
        counts = [self.cell_counts[state].get(ct, 0) for ct in self.cell_types]
        fig, ax = plt.subplots(figsize=(14, 8))

        # Create DataFrame and sort by count
        df_counts = pd.DataFrame({'Cell Type': self.cell_types, 'Count': counts}).sort_values('Count', ascending=False)

        # Define color palette - gradient from deep blue to light blue
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(df_counts)))

        # Create bars with gradient colors
        bars = ax.bar(range(len(df_counts)), df_counts['Count'], color=colors,
                      edgecolor='white', linewidth=2, alpha=0.9)

        # Add value labels on top of bars
        for i, (idx, row) in enumerate(df_counts.iterrows()):
            height = row['Count']
            if height > 0:  # Only show label if count > 0
                ax.text(i, height, f'{int(height)}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Styling
        ax.set_xticks(range(len(df_counts)))
        ax.set_xticklabels(df_counts['Cell Type'], rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Cell Count', fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title(f'Cell Type Distribution - {state}',
                     fontsize=18, fontweight='bold', pad=20)

        # Add subtle grid
        ax.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)

        # Add total count annotation
        total = df_counts['Count'].sum()
        ax.text(0.98, 0.98, f'Total: {int(total)} cells',
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def plot_heatmap(self, save_path=None):
        """Plot the change rate heatmap"""
        data = []
        for ct in self.cell_types:
            row = [self.calculate_change_rate(self.states[i], self.states[i + 1], ct)
                   for i in range(len(self.states) - 1)]
            data.append(row)
        transitions = [f"{self.states[i]} → {self.states[i + 1]}" for i in range(len(self.states) - 1)]

        fig, ax = plt.subplots(figsize=(10, 10))
        max_abs = max(abs(min(min(row) for row in data if row)),
                      abs(max(max(row) for row in data if row))) if any(data) else 100
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=-max_abs, vmax=max_abs)
        ax.set_xticks(range(len(transitions)))
        ax.set_yticks(range(len(self.cell_types)))
        ax.set_xticklabels(transitions, rotation=45, ha='right')
        ax.set_yticklabels(self.cell_types)

        for i in range(len(self.cell_types)):
            for j in range(len(transitions)):
                color = "white" if abs(data[i][j]) > 0.5 * max_abs else "black"
                ax.text(j, i, f'{data[i][j]:.1f}%', ha="center", va="center", color=color, fontsize=8)

        plt.colorbar(im, ax=ax, label='Change Rate (%)')
        ax.set_title('State Transition Change Rate Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def export_data(self, filepath='petri_net_data.xlsx'):
        """Export analysis data to an Excel file"""
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                trend_df = self.get_trend_data()
                trend_df.to_excel(writer, sheet_name='Trend Data', index=False)

                change_df = self.get_change_matrix()
                change_df_export = change_df.copy()
                for col in change_df_export.columns[1:]:
                    change_df_export[col] = change_df_export[col].astype(str).str.rstrip('%').astype(float)
                change_df_export.to_excel(writer, sheet_name='Change Rate', index=False)

                rules_data = []
                for ct, rule in self.rules.items():
                    rules_data.append({
                        'Cell Type': ct,
                        'Positive Markers': ', '.join(rule['pos']),
                        'Negative Markers': ', '.join(rule['neg'])
                    })
                rules_df = pd.DataFrame(rules_data)
                rules_df.to_excel(writer, sheet_name='Classification Rules', index=False)

            print(f"✅ Data successfully exported to {filepath}")
        except ImportError:
            print("\n❌ ERROR: The 'openpyxl' library is required to export data to Excel.")
            print("Please install it using: pip install openpyxl")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TIMED PETRI NET ANALYSIS")
    print("Cell Type Dynamics in Lung Adenocarcinoma Progression")
    print("=" * 80 + "\n")

    # --- STEP 1: LOAD REAL EXPERIMENTAL DATA ---
    root_directory = './'  # Change this to your data directory
    cells_by_category = load_all_cell_data(root_directory)

    # --- STEP 2: INITIALIZE THE MODEL ---
    model = TimedPetriNet()

    # --- STEP 3: LOAD THE REAL DATA INTO MODEL ---
    model.load_real_data_from_dict(cells_by_category)

    # --- STEP 4: RUN ANALYSIS AND VISUALIZATION ---

    # Print summary
    model.print_summary()

    # Draw Petri Net
    print("\n📊 Drawing Petri Net state diagram...")
    model.plot_petri_net(current_state=0, save_path='petri_net_diagram.png')

    # Draw trend lines
    print("📈 Drawing cell type evolution trend lines...")
    model.plot_trend_lines(
        selected_cell_types=None,  # Show ALL cell types
        save_path='cell_trend_lines.png'
    )

    # Draw bar chart for each state
    print("📊 Drawing cell distribution for all states...")
    for state in model.states:
        model.plot_bar_comparison(state=state, save_path=f'cell_bar_{state}.png')

    # Draw heatmap
    print("🔥 Drawing change rate heatmap...")
    model.plot_heatmap(save_path='change_rate_heatmap.png')

    # View change rate matrix
    print("\n📋 State Transition Change Rate Matrix:")
    change_matrix = model.get_change_matrix()
    print(change_matrix)

    # Export data
    print("\n💾 Exporting data to Excel...")
    model.export_data('petri_net_results.xlsx')

    print("\n✅ All visualizations and analysis completed successfully!")