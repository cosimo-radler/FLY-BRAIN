#!/usr/bin/env python3
"""
09_Laplacian_eigenval_viz.py

Standalone script for creating clean eigenvalue distribution plots.
This loads the matrix files directly to avoid import issues.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import data_io

def find_and_load_matrices():
    """
    Find and load the most recent Laplacian matrix files, properly categorized.
    """
    matrices_dir = data_io.RESULTS_DIR / "laplacian_matrices"
    if not matrices_dir.exists():
        print(f"No matrices directory found at {matrices_dir}")
        return {}
    
    # Find all npz files
    npz_files = list(matrices_dir.glob("*_laplacian_matrices_*.npz"))
    
    if not npz_files:
        print("No matrix files found")
        return {}
    
    print(f"Found {len(npz_files)} matrix files")
    
    # Group by region and model type, keep only latest files
    region_model_files = {}
    for file_path in npz_files:
        # Extract region and model type from filename
        name_parts = file_path.name.replace('.npz', '').split('_')
        
        if len(name_parts) >= 4:  # e.g., al_original_laplacian_matrices_timestamp
            region = name_parts[0].upper()
            
            # Determine model type based on filename pattern
            if len(name_parts) == 4:  # e.g., al_laplacian_matrices_timestamp (original from first run)
                clean_model = 'Original'
            elif name_parts[1] == 'original':
                clean_model = 'Original'
            elif name_parts[1] == 'configuration' and name_parts[2] == 'model':
                clean_model = 'Configuration Model'
            elif name_parts[1] == 'coarsened':
                clean_model = 'Coarsened'
            else:
                print(f"Skipping unrecognized file pattern: {file_path.name}")
                continue
            
            key = (region, clean_model)
            
            # Get timestamp for comparison
            timestamp = name_parts[-1]
            
            # Keep only the latest file for each region-model combination
            if key not in region_model_files:
                region_model_files[key] = (file_path, timestamp)
            else:
                existing_timestamp = region_model_files[key][1]
                if timestamp > existing_timestamp:
                    region_model_files[key] = (file_path, timestamp)
    
    # Load the matrices
    matrices_data = {}
    for (region, model), (file_path, timestamp) in region_model_files.items():
        try:
            data = np.load(file_path)
            
            if region not in matrices_data:
                matrices_data[region] = {}
            
            matrices_data[region][model] = {
                'eigenvals_normalized': data['eigenvals_normalized'],
                'eigenvals_standard': data['eigenvals_standard']
            }
            print(f"Loaded {region} {model} from {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return matrices_data

def create_eigenvalue_distributions(matrices_data):
    """
    Create clean eigenvalue distribution plots for all regions and model types.
    """
    if not matrices_data:
        print("No matrix data available")
        return
    
    regions = sorted(matrices_data.keys())
    
    # Get all model types present
    all_models = set()
    for region_data in matrices_data.values():
        all_models.update(region_data.keys())
    model_types = sorted(list(all_models))
    
    print(f"Creating eigenvalue distributions for:")
    print(f"Regions: {regions}")
    print(f"Model types: {model_types}")
    
    # Create subplots
    n_regions = len(regions)
    n_models = len(model_types)
    
    fig, axes = plt.subplots(n_regions, n_models, figsize=(5*n_models, 4*n_regions))
    
    # Handle single row/column cases
    if n_regions == 1 and n_models == 1:
        axes = [[axes]]
    elif n_regions == 1:
        axes = [axes]
    elif n_models == 1:
        axes = [[ax] for ax in axes]
    
    # Define colors for different model types
    colors = {'Original': '#1f77b4', 'Configuration Model': '#ff7f0e', 'Coarsened': '#2ca02c'}
    
    for i, region in enumerate(regions):
        for j, model_type in enumerate(model_types):
            if model_type not in matrices_data[region]:
                axes[i][j].text(0.5, 0.5, f'No {model_type}\ndata available', 
                               ha='center', va='center', transform=axes[i][j].transAxes,
                               fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray'))
                axes[i][j].set_title(f'{region} - {model_type}\nNormalized Laplacian', fontweight='bold')
                axes[i][j].set_xlabel('Eigenvalue')
                axes[i][j].set_ylabel('Density')
                continue
            
            eigenvals = matrices_data[region][model_type]['eigenvals_normalized']
            
            # Plot histogram
            axes[i][j].hist(eigenvals, bins=50, alpha=0.7, density=True, 
                           color=colors.get(model_type, 'gray'), edgecolor='black', linewidth=0.3)
            axes[i][j].set_title(f'{region} - {model_type}\nNormalized Laplacian', fontweight='bold')
            axes[i][j].set_xlabel('Eigenvalue')
            axes[i][j].set_ylabel('Density')
            axes[i][j].grid(True, alpha=0.3)
            
            # Add vertical lines for key eigenvalues
            if len(eigenvals) > 1:
                # Algebraic connectivity (second smallest eigenvalue)
                axes[i][j].axvline(eigenvals[1], color='red', linestyle='--', 
                                  label=f'λ₂ = {eigenvals[1]:.3f}', linewidth=2)
            
            # Maximum eigenvalue
            axes[i][j].axvline(eigenvals[-1], color='orange', linestyle='--', 
                              label=f'λₘₐₓ = {eigenvals[-1]:.3f}', linewidth=2)
            
            axes[i][j].legend(fontsize=9, loc='upper right')
            
            # Set reasonable x-axis limits
            axes[i][j].set_xlim(0, min(2.0, eigenvals[-1] * 1.1))
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = data_io.RESULTS_FIGURES / f"eigenvalue_distributions_all_regions_all_models_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nClean eigenvalue distributions saved to {fig_path}")
    
    plt.show()

def main():
    """
    Main function to create eigenvalue distribution plots.
    """
    print("="*60)
    print("EIGENVALUE DISTRIBUTION VISUALIZATION")
    print("="*60)
    
    # Load matrix data
    print("\n1. Loading Laplacian matrix files...")
    matrices_data = find_and_load_matrices()
    
    if not matrices_data:
        print("No matrix data found. Make sure you've run 09_Laplacian.py first.")
        return
    
    # Create plots
    print("\n2. Creating eigenvalue distribution plots...")
    create_eigenvalue_distributions(matrices_data)
    
    print("\n" + "="*60)
    print("EIGENVALUE VISUALIZATION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main() 