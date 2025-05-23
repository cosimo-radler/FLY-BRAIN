#!/usr/bin/env python3
"""
09_Laplacian_utils.py

Utility functions for working with the computed Laplacian matrices.
This script provides helper functions to:
1. Load saved Laplacian matrices
2. Visualize Laplacian eigenvalue distributions
3. Compare Laplacian properties across brain regions
4. Extract specific matrix properties for further analysis

Usage examples:
- Load a specific brain region's Laplacian: load_laplacian_matrices('AL')
- Plot eigenvalue distributions: plot_eigenvalue_distribution(['AL', 'FB'])
- Get algebraic connectivity for all regions: get_algebraic_connectivity_summary()
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import data_io

def find_latest_laplacian_files():
    """
    Find the most recent Laplacian matrix files for all model types.
    
    Returns:
        dict: Dictionary mapping (region, model_type) tuples to their latest matrix file paths
    """
    matrices_dir = data_io.RESULTS_DIR / "laplacian_matrices"
    if not matrices_dir.exists():
        print(f"No Laplacian matrices directory found at {matrices_dir}")
        return {}
    
    # Find all npz files (now includes model type in filename)
    npz_files = list(matrices_dir.glob("*_laplacian_matrices_*.npz"))
    
    if not npz_files:
        print("No Laplacian matrix files found")
        return {}
    
    # Group by (region, model_type) and find latest for each
    region_model_files = {}
    for file_path in npz_files:
        # Extract region and model type from filename
        # Format: {region}_{model_type}_laplacian_matrices_{timestamp}.npz
        name_parts = file_path.name.split('_')
        if len(name_parts) >= 5:
            region = name_parts[0].upper()
            model_type = '_'.join(name_parts[1:-3])  # Handle multi-word model types
            key = (region, model_type)
            
            if key not in region_model_files:
                region_model_files[key] = file_path
            else:
                # Compare timestamps to find the latest
                current_timestamp = file_path.name.split('_')[-1].replace('.npz', '')
                existing_timestamp = region_model_files[key].name.split('_')[-1].replace('.npz', '')
                if current_timestamp > existing_timestamp:
                    region_model_files[key] = file_path
    
    return region_model_files

def load_laplacian_matrices(brain_region=None, model_type=None):
    """
    Load Laplacian matrices for specific brain regions and/or model types.
    
    Args:
        brain_region (str, optional): Specific brain region (e.g., 'AL'). 
                                     If None, loads all available regions.
        model_type (str, optional): Specific model type (e.g., 'original', 'configuration_model'). 
                                   If None, loads all available model types.
    
    Returns:
        dict: Nested dictionary with structure {region: {model_type: data}}
    """
    region_model_files = find_latest_laplacian_files()
    
    if not region_model_files:
        return {}
    
    results = {}
    
    # Filter by brain_region if specified
    if brain_region is not None:
        brain_region = brain_region.upper()
        region_model_files = {k: v for k, v in region_model_files.items() if k[0] == brain_region}
    
    # Filter by model_type if specified
    if model_type is not None:
        region_model_files = {k: v for k, v in region_model_files.items() if k[1] == model_type}
    
    for (region, model), file_path in region_model_files.items():
        try:
            data = np.load(file_path)
            
            if region not in results:
                results[region] = {}
            
            results[region][model] = {
                'standard_laplacian': data['standard_laplacian'],
                'normalized_laplacian': data['normalized_laplacian'],
                'eigenvals_standard': data['eigenvals_standard'],
                'eigenvals_normalized': data['eigenvals_normalized'],
                'file_path': file_path
            }
            print(f"Loaded Laplacian matrices for {region} {model} from {file_path}")
        except Exception as e:
            print(f"Error loading matrices for {region} {model}: {e}")
    
    return results

def plot_eigenvalue_distribution(brain_regions=None, model_types=None, normalized=True, save_fig=True):
    """
    Plot eigenvalue distributions for specified brain regions and model types.
    
    Args:
        brain_regions (list, optional): List of brain regions to plot. If None, plots all.
        model_types (list, optional): List of model types to plot. If None, plots all.
        normalized (bool): Whether to plot normalized or standard Laplacian eigenvalues
        save_fig (bool): Whether to save the figure
    """
    # Load the matrices
    matrices_data = load_laplacian_matrices()
    
    if not matrices_data:
        print("No matrix data available for plotting")
        return
    
    # Set defaults
    if brain_regions is None:
        brain_regions = list(matrices_data.keys())
    
    if model_types is None:
        # Get all unique model types
        model_types = set()
        for region_data in matrices_data.values():
            model_types.update(region_data.keys())
        model_types = sorted(list(model_types))
    
    # Create subplots
    n_regions = len(brain_regions)
    n_models = len(model_types)
    
    fig, axes = plt.subplots(n_regions, n_models, figsize=(4*n_models, 4*n_regions))
    if n_regions == 1 and n_models == 1:
        axes = [[axes]]
    elif n_regions == 1:
        axes = [axes]
    elif n_models == 1:
        axes = [[ax] for ax in axes]
    
    eigenval_type = 'eigenvals_normalized' if normalized else 'eigenvals_standard'
    title_type = 'Normalized' if normalized else 'Standard'
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']  # Different colors for different models
    
    for i, region in enumerate(brain_regions):
        if region not in matrices_data:
            print(f"No data for region {region}")
            continue
        
        for j, model_type in enumerate(model_types):
            if model_type not in matrices_data[region]:
                print(f"No {model_type} data for region {region}")
                axes[i][j].text(0.5, 0.5, f'No {model_type}\ndata available', 
                               ha='center', va='center', transform=axes[i][j].transAxes)
                axes[i][j].set_title(f'{region} - {model_type}\n{title_type} Laplacian')
                continue
            
            eigenvals = matrices_data[region][model_type][eigenval_type]
            
            axes[i][j].hist(eigenvals, bins=50, alpha=0.7, density=True, color=colors[j % len(colors)])
            axes[i][j].set_title(f'{region} - {model_type}\n{title_type} Laplacian')
            axes[i][j].set_xlabel('Eigenvalue')
            axes[i][j].set_ylabel('Density')
            axes[i][j].grid(True, alpha=0.3)
            
            # Add some statistics to the plot
            axes[i][j].axvline(eigenvals[1], color='red', linestyle='--', 
                              label=f'λ₂ = {eigenvals[1]:.3f}')
            axes[i][j].axvline(eigenvals[-1], color='orange', linestyle='--', 
                              label=f'λₘₐₓ = {eigenvals[-1]:.3f}')
            axes[i][j].legend()
    
    plt.tight_layout()
    
    if save_fig:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        regions_str = "_".join(brain_regions) if brain_regions else "all"
        models_str = "_".join(model_types) if model_types else "all"
        fig_name = f"laplacian_eigenvalue_distributions_{regions_str}_{models_str}_{title_type.lower()}_{timestamp}.png"
        fig_path = data_io.RESULTS_FIGURES / fig_name
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {fig_path}")
    
    plt.show()

def get_algebraic_connectivity_summary():
    """
    Get a summary of algebraic connectivity (second smallest eigenvalue) for all regions and models.
    
    Returns:
        pandas.DataFrame: Summary of algebraic connectivity values
    """
    matrices_data = load_laplacian_matrices()
    
    if not matrices_data:
        print("No matrix data available")
        return pd.DataFrame()
    
    summary_data = []
    for region, models in matrices_data.items():
        for model_type, data in models.items():
            eigenvals = data['eigenvals_standard']
            eigenvals_norm = data['eigenvals_normalized']
            
            summary_data.append({
                'brain_region': region,
                'model_type': model_type,
                'num_nodes': len(eigenvals),
                'algebraic_connectivity': eigenvals[1] if len(eigenvals) > 1 else 0.0,
                'spectral_gap': eigenvals[1] - eigenvals[0] if len(eigenvals) > 1 else 0.0,
                'max_eigenvalue': eigenvals[-1],
                'normalized_max_eigenvalue': eigenvals_norm[-1],
                'normalized_algebraic_connectivity': eigenvals_norm[1] if len(eigenvals_norm) > 1 else 0.0
            })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['brain_region', 'model_type'])
    
    print("Algebraic Connectivity Summary:")
    print(df.to_string(index=False))
    
    return df

def compare_laplacian_properties():
    """
    Create a comparison plot of key Laplacian properties across brain regions and model types.
    """
    df = get_algebraic_connectivity_summary()
    
    if df.empty:
        return
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Set up the plotting style
    regions = df['brain_region'].unique()
    model_types = df['model_type'].unique()
    
    # Define colors and patterns for different model types
    colors = {'original': 'blue', 'configuration_model': 'red', 'coarsened': 'green'}
    patterns = {'original': '', 'configuration_model': '///', 'coarsened': '...'}
    
    x_positions = np.arange(len(regions))
    width = 0.25  # Width of each bar
    
    # Plot 1: Algebraic connectivity
    for i, model in enumerate(model_types):
        model_data = df[df['model_type'] == model]
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['algebraic_connectivity'].iloc[0] if not region_data.empty else 0)
        
        axes[0,0].bar(x_positions + i*width, values, width, 
                     label=model, color=colors.get(model, 'gray'), 
                     hatch=patterns.get(model, ''), alpha=0.8)
    
    axes[0,0].set_title('Algebraic Connectivity (λ₂)')
    axes[0,0].set_ylabel('Value')
    axes[0,0].set_xlabel('Brain Region')
    axes[0,0].set_xticks(x_positions + width)
    axes[0,0].set_xticklabels(regions)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Spectral gap
    for i, model in enumerate(model_types):
        model_data = df[df['model_type'] == model]
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['spectral_gap'].iloc[0] if not region_data.empty else 0)
        
        axes[0,1].bar(x_positions + i*width, values, width, 
                     label=model, color=colors.get(model, 'gray'),
                     hatch=patterns.get(model, ''), alpha=0.8)
    
    axes[0,1].set_title('Spectral Gap (λ₂ - λ₁)')
    axes[0,1].set_ylabel('Value')
    axes[0,1].set_xlabel('Brain Region')
    axes[0,1].set_xticks(x_positions + width)
    axes[0,1].set_xticklabels(regions)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Maximum eigenvalue
    for i, model in enumerate(model_types):
        model_data = df[df['model_type'] == model]
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['max_eigenvalue'].iloc[0] if not region_data.empty else 0)
        
        axes[1,0].bar(x_positions + i*width, values, width, 
                     label=model, color=colors.get(model, 'gray'),
                     hatch=patterns.get(model, ''), alpha=0.8)
    
    axes[1,0].set_title('Maximum Eigenvalue (Standard)')
    axes[1,0].set_ylabel('Value')
    axes[1,0].set_xlabel('Brain Region')
    axes[1,0].set_xticks(x_positions + width)
    axes[1,0].set_xticklabels(regions)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Normalized maximum eigenvalue
    for i, model in enumerate(model_types):
        model_data = df[df['model_type'] == model]
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['normalized_max_eigenvalue'].iloc[0] if not region_data.empty else 0)
        
        axes[1,1].bar(x_positions + i*width, values, width, 
                     label=model, color=colors.get(model, 'gray'),
                     hatch=patterns.get(model, ''), alpha=0.8)
    
    axes[1,1].set_title('Maximum Eigenvalue (Normalized)')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_xlabel('Brain Region')
    axes[1,1].set_xticks(x_positions + width)
    axes[1,1].set_xticklabels(regions)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = data_io.RESULTS_FIGURES / f"laplacian_properties_comparison_all_models_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {fig_path}")
    
    plt.show()

def main():
    """
    Example usage of the utility functions for multiple model types.
    """
    print("="*60)
    print("LAPLACIAN MATRIX UTILITIES DEMO - ALL MODEL TYPES")
    print("="*60)
    
    # Get summary for all models
    print("\n1. Getting algebraic connectivity summary for all models...")
    df_summary = get_algebraic_connectivity_summary()
    
    print("\n2. Creating comparison plots for all models...")
    compare_laplacian_properties()
    
    print("\n3. Plotting eigenvalue distributions for specific regions and models...")
    # Plot for AL and MB with all available models
    plot_eigenvalue_distribution(['AL', 'MB'], normalized=True)
    
    print("\n4. Example: Loading specific region and model matrices...")
    # Load original AL data
    al_original = load_laplacian_matrices('AL', 'original')
    if al_original:
        print(f"AL original Laplacian matrix shape: {al_original['AL']['original']['standard_laplacian'].shape}")
        print(f"AL original has {len(al_original['AL']['original']['eigenvals_standard'])} eigenvalues")
    
    # Load configuration model data if available
    al_cm = load_laplacian_matrices('AL', 'configuration_model')
    if al_cm:
        print(f"AL configuration model Laplacian matrix shape: {al_cm['AL']['configuration_model']['standard_laplacian'].shape}")
        print(f"AL configuration model has {len(al_cm['AL']['configuration_model']['eigenvals_standard'])} eigenvalues")
    
    print("\n5. Model comparison for specific region...")
    if df_summary is not None and not df_summary.empty:
        al_data = df_summary[df_summary['brain_region'] == 'AL']
        if not al_data.empty:
            print("\nAL Model Comparison:")
            print(al_data[['model_type', 'algebraic_connectivity', 'spectral_gap', 'max_eigenvalue']].to_string(index=False))
    
    print("\n" + "="*60)
    print("Demo complete! Check the results/figures/ directory for generated plots.")
    print("="*60)

if __name__ == "__main__":
    main() 