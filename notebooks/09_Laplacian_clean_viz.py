#!/usr/bin/env python3
"""
09_Laplacian_clean_viz.py

Clean visualization script for Laplacian matrix comparisons across model types.
This script:
1. Loads the latest results from the CSV summary (cleaner than parsing filenames)
2. Creates proper comparison visualizations with distinct colors
3. Shows all brain regions and three model types: original, configuration_model, coarsened
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import data_io

def load_latest_summary():
    """
    Load the most recent Laplacian summary CSV file.
    """
    tables_dir = data_io.RESULTS_TABLES
    summary_files = list(tables_dir.glob("laplacian_summary_all_models_*.csv"))
    
    if not summary_files:
        print("No summary files found")
        return None
    
    # Get the most recent file
    latest_file = max(summary_files, key=lambda x: x.name)
    print(f"Loading summary from: {latest_file}")
    
    df = pd.read_csv(latest_file)
    
    # Clean up model types to ensure consistency
    model_type_mapping = {
        'original': 'Original',
        'configuration_model': 'Configuration Model', 
        'coarsened': 'Coarsened'
    }
    
    df['model_type_clean'] = df['model_type'].map(model_type_mapping)
    
    # Filter to only the three main model types
    df = df[df['model_type_clean'].notna()]
    
    return df

def create_comparison_plots(df):
    """
    Create clean comparison plots with proper colors and all regions.
    """
    if df is None or df.empty:
        print("No data available for plotting")
        return
    
    # Set up the plot style
    plt.style.use('default')  # Use default matplotlib style for colors
    regions = sorted(df['brain_region'].unique())
    model_types = sorted(df['model_type_clean'].unique())
    
    print(f"Brain regions: {regions}")
    print(f"Model types: {model_types}")
    
    # Define distinct colors for different model types
    colors = {'Original': '#1f77b4', 'Configuration Model': '#ff7f0e', 'Coarsened': '#2ca02c'}
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    x_positions = np.arange(len(regions))
    width = 0.25  # Width of each bar
    
    # Plot 1: Algebraic connectivity
    for i, model in enumerate(model_types):
        model_data = df[df['model_type_clean'] == model].sort_values('brain_region')
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['algebraic_connectivity'].iloc[0] if not region_data.empty else 0)
        
        axes[0,0].bar(x_positions + i*width, values, width, 
                     label=model, color=colors[model], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0,0].set_title('Algebraic Connectivity (λ₂)', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Value', fontsize=12)
    axes[0,0].set_xlabel('Brain Region', fontsize=12)
    axes[0,0].set_xticks(x_positions + width)
    axes[0,0].set_xticklabels(regions, fontsize=10)
    axes[0,0].legend(fontsize=10)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Spectral gap
    for i, model in enumerate(model_types):
        model_data = df[df['model_type_clean'] == model].sort_values('brain_region')
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['spectral_gap'].iloc[0] if not region_data.empty else 0)
        
        axes[0,1].bar(x_positions + i*width, values, width, 
                     label=model, color=colors[model], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[0,1].set_title('Spectral Gap (λ₂ - λ₁)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Value', fontsize=12)
    axes[0,1].set_xlabel('Brain Region', fontsize=12)
    axes[0,1].set_xticks(x_positions + width)
    axes[0,1].set_xticklabels(regions, fontsize=10)
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Maximum eigenvalue (log scale due to large differences)
    for i, model in enumerate(model_types):
        model_data = df[df['model_type_clean'] == model].sort_values('brain_region')
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['laplacian_largest'].iloc[0] if not region_data.empty else 1)
        
        axes[1,0].bar(x_positions + i*width, values, width, 
                     label=model, color=colors[model], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[1,0].set_title('Maximum Eigenvalue (Standard)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Value (log scale)', fontsize=12)
    axes[1,0].set_xlabel('Brain Region', fontsize=12)
    axes[1,0].set_xticks(x_positions + width)
    axes[1,0].set_xticklabels(regions, fontsize=10)
    axes[1,0].legend(fontsize=10)
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Normalized maximum eigenvalue
    for i, model in enumerate(model_types):
        model_data = df[df['model_type_clean'] == model].sort_values('brain_region')
        values = []
        for region in regions:
            region_data = model_data[model_data['brain_region'] == region]
            values.append(region_data['normalized_laplacian_largest'].iloc[0] if not region_data.empty else 0)
        
        axes[1,1].bar(x_positions + i*width, values, width, 
                     label=model, color=colors[model], alpha=0.8, edgecolor='black', linewidth=0.5)
    
    axes[1,1].set_title('Maximum Eigenvalue (Normalized)', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('Value', fontsize=12)
    axes[1,1].set_xlabel('Brain Region', fontsize=12)
    axes[1,1].set_xticks(x_positions + width)
    axes[1,1].set_xticklabels(regions, fontsize=10)
    axes[1,1].legend(fontsize=10)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = data_io.RESULTS_FIGURES / f"laplacian_comparison_clean_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Clean comparison plot saved to {fig_path}")
    
    plt.show()

def create_eigenvalue_distributions():
    """
    Create eigenvalue distribution plots using the actual matrix files for all regions and clean model types.
    """
    # Load matrices using the existing utility function, but with better filtering
    from notebooks import laplacian_utils_09 as utils
    
    try:
        matrices_data = utils.load_laplacian_matrices()
    except:
        print("Could not load matrix data for eigenvalue distributions")
        return
    
    if not matrices_data:
        print("No matrix data available")
        return
    
    # Filter to get clean model types and all regions
    regions = sorted(matrices_data.keys())
    
    # Map model type names to clean versions
    model_mapping = {
        'original': 'Original',
        'configuration_model': 'Configuration Model',
        'coarsened': 'Coarsened'
    }
    
    # Find available model types across all regions
    all_models = set()
    for region_data in matrices_data.values():
        all_models.update(region_data.keys())
    
    # Filter to main model types
    clean_models = []
    for model in all_models:
        clean_model = None
        if 'original' in model and model != 'laplacian':
            clean_model = 'Original'
        elif 'configuration' in model:
            clean_model = 'Configuration Model'
        elif 'coarsened' in model:
            clean_model = 'Coarsened'
        
        if clean_model and clean_model not in clean_models:
            clean_models.append(clean_model)
    
    clean_models = sorted(clean_models)
    
    if not clean_models:
        print("No clean model types found")
        return
    
    print(f"Creating eigenvalue distributions for regions: {regions}")
    print(f"Model types: {clean_models}")
    
    # Create subplots
    n_regions = len(regions)
    n_models = len(clean_models)
    
    fig, axes = plt.subplots(n_regions, n_models, figsize=(5*n_models, 4*n_regions))
    if n_regions == 1:
        axes = [axes]
    if n_models == 1:
        axes = [[ax] for ax in axes]
    
    colors = {'Original': '#1f77b4', 'Configuration Model': '#ff7f0e', 'Coarsened': '#2ca02c'}
    
    for i, region in enumerate(regions):
        for j, clean_model in enumerate(clean_models):
            # Find the actual model key in the data
            model_key = None
            for key in matrices_data[region].keys():
                if (clean_model == 'Original' and 'original' in key and key != 'laplacian') or \
                   (clean_model == 'Configuration Model' and 'configuration' in key) or \
                   (clean_model == 'Coarsened' and 'coarsened' in key):
                    model_key = key
                    break
            
            if model_key is None:
                axes[i][j].text(0.5, 0.5, f'No {clean_model}\ndata available', 
                               ha='center', va='center', transform=axes[i][j].transAxes)
                axes[i][j].set_title(f'{region} - {clean_model}\nNormalized Laplacian')
                continue
            
            eigenvals = matrices_data[region][model_key]['eigenvals_normalized']
            
            axes[i][j].hist(eigenvals, bins=50, alpha=0.7, density=True, 
                           color=colors[clean_model], edgecolor='black', linewidth=0.3)
            axes[i][j].set_title(f'{region} - {clean_model}\nNormalized Laplacian', fontweight='bold')
            axes[i][j].set_xlabel('Eigenvalue')
            axes[i][j].set_ylabel('Density')
            axes[i][j].grid(True, alpha=0.3)
            
            # Add statistics
            if len(eigenvals) > 1:
                axes[i][j].axvline(eigenvals[1], color='red', linestyle='--', 
                                  label=f'λ₂ = {eigenvals[1]:.3f}', linewidth=2)
            axes[i][j].axvline(eigenvals[-1], color='orange', linestyle='--', 
                              label=f'λₘₐₓ = {eigenvals[-1]:.3f}', linewidth=2)
            axes[i][j].legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = data_io.RESULTS_FIGURES / f"eigenvalue_distributions_all_regions_clean_{timestamp}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Clean eigenvalue distributions saved to {fig_path}")
    
    plt.show()

def main():
    """
    Main function to create clean visualizations.
    """
    print("="*60)
    print("CLEAN LAPLACIAN MATRIX VISUALIZATIONS")
    print("="*60)
    
    # Load summary data
    print("\n1. Loading latest summary data...")
    df = load_latest_summary()
    
    if df is not None:
        print(f"Loaded data for {len(df)} region-model combinations")
        print("\nData summary:")
        print(df.groupby(['brain_region', 'model_type_clean']).size().unstack(fill_value=0))
    
    # Create comparison plots
    print("\n2. Creating clean comparison plots...")
    create_comparison_plots(df)
    
    # Create eigenvalue distributions (this might not work due to import issues, but let's try)
    print("\n3. Creating eigenvalue distribution plots...")
    try:
        create_eigenvalue_distributions()
    except Exception as e:
        print(f"Could not create eigenvalue distributions: {e}")
        print("This is likely due to import path issues - you can run the eigenvalue part separately")
    
    print("\n" + "="*60)
    print("CLEAN VISUALIZATIONS COMPLETE!")
    print("Check results/figures/ for the new clean plots")
    print("="*60)

if __name__ == "__main__":
    main() 