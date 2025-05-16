"""
Percolation module for the Drosophila connectome analysis pipeline.

This module provides functions for performing bond percolation analysis on networks,
including random and degree-based edge removal strategies, and measuring the impact
on the largest connected component.
"""

import os
import random
import logging
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from matplotlib.ticker import MaxNLocator

from . import config

logger = logging.getLogger("fly_brain")

def set_seed(seed):
    """
    Set random seed for reproducibility.
    
    Parameters:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)

def random_percolation(G, fractions, seed=None):
    """
    Perform random edge removal percolation.
    
    Parameters:
        G (nx.Graph): Input graph
        fractions (list): List of fractions of edges to remove
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary mapping edge removal fractions to largest component sizes
    """
    if seed is not None:
        set_seed(seed)
    
    # Convert to undirected for percolation analysis if needed
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G.copy()
    
    # Get total number of edges
    total_edges = G_undirected.number_of_edges()
    total_nodes = G_undirected.number_of_nodes()
    
    # Store results
    results = {}
    
    # Original largest component size (normalized)
    original_lcc_size = len(max(nx.connected_components(G_undirected), key=len)) / total_nodes
    results[0.0] = original_lcc_size
    
    # Create a copy for edge removal
    G_work = G_undirected.copy()
    
    # Sort fractions in ascending order
    fractions = sorted(fractions)
    
    # Keep track of edges removed
    edges_removed = 0
    
    # Process each fraction
    for fraction in fractions:
        # Calculate how many edges to remove for this step
        target_edges_to_remove = int(total_edges * fraction)
        edges_to_remove_now = target_edges_to_remove - edges_removed
        
        if edges_to_remove_now <= 0:
            # We've already removed enough edges for this fraction
            continue
        
        # Get all remaining edges
        remaining_edges = list(G_work.edges())
        
        # Randomly sample edges to remove for this step
        if edges_to_remove_now < len(remaining_edges):
            edges_to_remove = random.sample(remaining_edges, edges_to_remove_now)
        else:
            edges_to_remove = remaining_edges
        
        # Remove edges
        G_work.remove_edges_from(edges_to_remove)
        edges_removed += len(edges_to_remove)
        
        # Calculate largest connected component size (normalized)
        if G_work.number_of_edges() > 0:
            largest_cc = max(nx.connected_components(G_work), key=len)
            lcc_size = len(largest_cc) / total_nodes
        else:
            lcc_size = 1 / total_nodes  # Only one node left in the largest component
        
        results[fraction] = lcc_size
    
    return results

def degree_based_percolation(G, fractions, method='high', seed=None):
    """
    Perform degree-based edge removal percolation.
    
    Parameters:
        G (nx.Graph): Input graph
        fractions (list): List of fractions of edges to remove
        method (str): 'high' to remove edges connected to high degree nodes first,
                     'low' to remove edges connected to low degree nodes first
        seed (int): Random seed for reproducibility
        
    Returns:
        dict: Dictionary mapping edge removal fractions to largest component sizes
    """
    if seed is not None:
        set_seed(seed)
    
    # Convert to undirected for percolation analysis if needed
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G.copy()
    
    # Get total number of edges
    total_edges = G_undirected.number_of_edges()
    total_nodes = G_undirected.number_of_nodes()
    
    # Store results
    results = {}
    
    # Original largest component size (normalized)
    original_lcc_size = len(max(nx.connected_components(G_undirected), key=len)) / total_nodes
    results[0.0] = original_lcc_size
    
    # Create a copy for edge removal
    G_work = G_undirected.copy()
    
    # Sort fractions in ascending order
    fractions = sorted(fractions)
    
    # Calculate edge importance based on degrees
    edge_importance = {}
    for u, v in G_work.edges():
        # Sum of degrees of incident nodes
        importance = G_work.degree(u) + G_work.degree(v)
        edge_importance[(u, v)] = importance
    
    # Sort edges by importance
    if method == 'high':
        # Remove edges connected to high degree nodes first
        sorted_edges = sorted(edge_importance.keys(), key=lambda e: edge_importance[e], reverse=True)
    else:
        # Remove edges connected to low degree nodes first
        sorted_edges = sorted(edge_importance.keys(), key=lambda e: edge_importance[e])
    
    # Keep track of edges removed
    edges_removed = 0
    
    # Process each fraction
    for fraction in fractions:
        # Calculate how many edges to remove for this step
        target_edges_to_remove = int(total_edges * fraction)
        edges_to_remove_now = target_edges_to_remove - edges_removed
        
        if edges_to_remove_now <= 0:
            # We've already removed enough edges for this fraction
            continue
        
        # Get edges to remove for this step
        edges_to_remove = sorted_edges[edges_removed:edges_removed + edges_to_remove_now]
        
        # Remove edges
        G_work.remove_edges_from(edges_to_remove)
        edges_removed += len(edges_to_remove)
        
        # Calculate largest connected component size (normalized)
        if G_work.number_of_edges() > 0:
            largest_cc = max(nx.connected_components(G_work), key=len)
            lcc_size = len(largest_cc) / total_nodes
        else:
            lcc_size = 1 / total_nodes  # Only one node left in the largest component
        
        results[fraction] = lcc_size
    
    return results

def compute_percolation_threshold(G):
    """
    Compute analytic percolation threshold for a network.
    
    For random networks, the percolation threshold is p_c = <k>/((<k^2> - <k>))
    where <k> is the average degree and <k^2> is the average squared degree.
    
    Parameters:
        G (nx.Graph): Input graph
        
    Returns:
        float: Percolation threshold p_c
    """
    # Convert to undirected if needed
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Get all degrees
    deg = np.array([d for _, d in G_undirected.degree()])
    
    # Calculate first moment (mean degree)
    k1 = deg.mean()
    
    # Calculate second moment (mean squared degree)
    k2 = (deg ** 2).mean()
    
    # Percolation threshold formula: <k> / (<k^2> - <k>)
    return k1 / (k2 - k1) if k2 > k1 else 1.0

def run_percolation_experiments(networks, fractions=None, seed=42, num_trials=1):
    """
    Run percolation experiments on multiple networks.
    
    Parameters:
        networks (dict): Dictionary of networks
        fractions (list): List of fractions of edges to remove
        seed (int): Random seed for reproducibility
        num_trials (int): Number of trials for random percolation
        
    Returns:
        tuple: (results, thresholds) - Dictionaries with experiment results and thresholds
    """
    if fractions is None:
        # Default fractions to remove, including f=1.0 (complete removal)
        fractions = np.linspace(0, 1.0, 21)
    
    # Store results
    results = {}
    thresholds = {}
    
    # Process each network
    for region, G in networks.items():
        results[region] = {}
        
        # Compute analytic percolation threshold
        p_c = compute_percolation_threshold(G)
        thresholds[region] = p_c
        logger.info(f"Region {region} analytic percolation threshold: p_c = {p_c:.4f}")
        
        logger.info(f"Running percolation experiments for {region}")
        
        # 1. Random percolation
        # For random percolation we might want to average over multiple trials
        random_results = []
        for trial in tqdm(range(num_trials), desc=f"Random percolation {region}"):
            trial_seed = seed + trial
            trial_result = random_percolation(G, fractions, seed=trial_seed)
            random_results.append(trial_result)
        
        # Average results across trials
        if num_trials > 1:
            avg_random = {}
            for fraction in fractions:
                values = [result.get(fraction, 0) for result in random_results if fraction in result]
                if values:
                    avg_random[fraction] = sum(values) / len(values)
            results[region]['random'] = avg_random
        else:
            results[region]['random'] = random_results[0]
        
        # 2. Degree-based percolation (high degree first)
        high_deg_result = degree_based_percolation(G, fractions, method='high', seed=seed)
        results[region]['high_degree'] = high_deg_result
        
        # 3. Degree-based percolation (low degree first)
        low_deg_result = degree_based_percolation(G, fractions, method='low', seed=seed)
        results[region]['low_degree'] = low_deg_result
    
    return results, thresholds

def plot_percolation_results(results, thresholds=None, plot_type="region"):
    """
    Plot percolation results.
    
    Parameters:
        results (dict): Results from percolation experiments
        thresholds (dict): Dictionary of percolation thresholds for each region
        plot_type (str): Type of plot to create ('region' or 'strategy')
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    if plot_type == "region":
        # One subplot per region, comparing different strategies
        fig, axs = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
        axs = axs.flatten()
        
        for i, (region, region_results) in enumerate(results.items()):
            # Skip configuration models in this plot (they're handled in another plot)
            if '_cm_' in region or 'scaled_cm' in region:
                continue
                
            if i < len(axs):
                ax = axs[i]
                
                for strategy, strategy_results in region_results.items():
                    fractions = sorted(strategy_results.keys())
                    lcc_sizes = [strategy_results[f] for f in fractions]
                    
                    if strategy == 'random':
                        label = 'Random Edge Removal'
                        linestyle = '-'
                    elif strategy == 'high_degree':
                        label = 'High Degree First'
                        linestyle = '--'
                    elif strategy == 'low_degree':
                        label = 'Low Degree First'
                        linestyle = ':'
                    else:
                        label = strategy
                        linestyle = '-'
                    
                    ax.plot(fractions, lcc_sizes, label=label, linestyle=linestyle)
                
                # Add percolation threshold vertical line if available
                if thresholds and region in thresholds:
                    p_c = thresholds[region]
                    f_c = 1 - p_c  # Convert to fraction of edges removed
                    ax.axvline(x=f_c, color='grey', linestyle='-', alpha=0.5, 
                               label=f'Analytic f_c = {f_c:.3f}')
                
                ax.set_xlabel('Fraction of Edges Removed')
                ax.set_ylabel('Relative Size of Largest Component')
                ax.set_title(f"{region} ({config.BRAIN_REGIONS.get(region, 'Unknown')})")
                ax.grid(True, alpha=0.3)
                ax.legend()
                ax.set_ylim(0, 1.05)
        
        # Remove any unused subplots
        i = 0
        for region in results.keys():
            if '_cm_' not in region and 'scaled_cm' not in region:
                i += 1
        for j in range(i, len(axs)):
            fig.delaxes(axs[j])
            
        plt.tight_layout()
        plt.suptitle("Percolation Analysis by Brain Region", fontsize=16)
        plt.subplots_adjust(top=0.93)
        
    else:  # plot_type == "strategy"
        # One subplot per strategy, comparing different regions
        strategies = ['random', 'high_degree', 'low_degree']
        
        fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
        
        for i, strategy in enumerate(strategies):
            ax = axs[i]
            
            for region, region_results in results.items():
                # Skip configuration models in this plot (they're handled in another plot)
                if '_cm_' in region or 'scaled_cm' in region:
                    continue
                    
                if strategy in region_results:
                    strategy_results = region_results[strategy]
                    fractions = sorted(strategy_results.keys())
                    lcc_sizes = [strategy_results[f] for f in fractions]
                    
                    line, = ax.plot(fractions, lcc_sizes, label=f"{region} ({config.BRAIN_REGIONS.get(region, 'Unknown')})")
                    
                    # Add percolation threshold marker if this is the random strategy
                    if strategy == 'random' and thresholds and region in thresholds:
                        p_c = thresholds[region]
                        f_c = 1 - p_c  # Convert to fraction of edges removed
                        ax.axvline(x=f_c, color=line.get_color(), linestyle=':', alpha=0.7)
            
            if strategy == 'random':
                title = 'Random Edge Removal'
                # Add a note about percolation thresholds
                if thresholds:
                    ax.text(0.95, 0.05, 'Vertical lines indicate analytic f_c', 
                           transform=ax.transAxes, ha='right', va='bottom', 
                           fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            elif strategy == 'high_degree':
                title = 'High Degree First'
            elif strategy == 'low_degree':
                title = 'Low Degree First'
            else:
                title = strategy
                
            ax.set_xlabel('Fraction of Edges Removed')
            ax.set_ylabel('Relative Size of Largest Component')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(0, 1.05)
            
        plt.tight_layout()
        plt.suptitle("Percolation Analysis by Attack Strategy", fontsize=16)
        plt.subplots_adjust(top=0.85)
    
    return fig

def plot_model_comparison(results, thresholds=None, strategy='random'):
    """
    Plot comparison between original networks and their configuration models.
    
    Parameters:
        results (dict): Results from percolation experiments
        thresholds (dict): Dictionary of percolation thresholds for each region
        strategy (str): Attack strategy to compare ('random', 'high_degree', or 'low_degree')
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Identify original regions
    original_regions = [r for r in results.keys() if '_cm_' not in r and 'scaled_cm' not in r]
    
    # Number of rows and columns for subplots (one row per region)
    n_rows = len(original_regions)
    
    if n_rows == 0:
        logger.warning("No original regions found for model comparison plot")
        return None
    
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 5*n_rows), dpi=100)
    
    # Handle case of only one region (axs not 2D)
    if n_rows == 1:
        axs = np.array([axs])
    
    # Process each original region
    for i, region in enumerate(original_regions):
        # Skip if region is not in results
        if region not in results:
            continue
            
        # Left plot: Unscaled CM comparison
        ax_unscaled = axs[i, 0]
        
        # Plot original network
        if strategy in results[region]:
            strategy_results = results[region][strategy]
            fractions = sorted(strategy_results.keys())
            lcc_sizes = [strategy_results[f] for f in fractions]
            
            ax_unscaled.plot(fractions, lcc_sizes, label=f"Original {region}", 
                          linestyle='-', linewidth=2)
            
            # Add percolation threshold
            if thresholds and region in thresholds:
                p_c = thresholds[region]
                f_c = 1 - p_c  # Convert to fraction of edges removed
                ax_unscaled.axvline(x=f_c, color='black', linestyle=':', alpha=0.7,
                                  label=f'Original f_c = {f_c:.3f}')
        
        # Plot unscaled configuration models
        for key in results.keys():
            # Look for unscaled CM for this region
            if key.startswith(f"{region}_cm_") and strategy in results[key]:
                # Extract seed from model key
                parts = key.split('_')
                seed = parts[-1]  # Last part is the seed
                
                strategy_results = results[key][strategy]
                fractions = sorted(strategy_results.keys())
                lcc_sizes = [strategy_results[f] for f in fractions]
                
                ax_unscaled.plot(fractions, lcc_sizes, label=f"CM (seed {seed})",
                              linestyle='--', alpha=0.7)
                
                # Add percolation threshold
                if thresholds and key in thresholds:
                    p_c = thresholds[key]
                    f_c = 1 - p_c  # Convert to fraction of edges removed
                    ax_unscaled.axvline(x=f_c, color='grey', linestyle='--', alpha=0.5,
                                     label=f'CM f_c = {f_c:.3f}')
        
        ax_unscaled.set_xlabel('Fraction of Edges Removed')
        ax_unscaled.set_ylabel('Relative Size of Largest Component')
        ax_unscaled.set_title(f"{region} vs Unscaled Configuration Models - {strategy}")
        ax_unscaled.grid(True, alpha=0.3)
        ax_unscaled.legend()
        ax_unscaled.set_ylim(0, 1.05)
        
        # Right plot: Scaled CM comparison
        ax_scaled = axs[i, 1]
        
        # Plot original network again
        if strategy in results[region]:
            strategy_results = results[region][strategy]
            fractions = sorted(strategy_results.keys())
            lcc_sizes = [strategy_results[f] for f in fractions]
            
            ax_scaled.plot(fractions, lcc_sizes, label=f"Original {region}", 
                         linestyle='-', linewidth=2)
            
            # Add percolation threshold
            if thresholds and region in thresholds:
                p_c = thresholds[region]
                f_c = 1 - p_c  # Convert to fraction of edges removed
                ax_scaled.axvline(x=f_c, color='black', linestyle=':', alpha=0.7,
                                label=f'Original f_c = {f_c:.3f}')
        
        # Plot scaled configuration models
        for key in results.keys():
            # Look for scaled CM for this region
            if key.startswith(f"{region}_scaled_cm_") and strategy in results[key]:
                # Extract seed from model key
                parts = key.split('_')
                seed = parts[-1]  # Last part is the seed
                
                strategy_results = results[key][strategy]
                fractions = sorted(strategy_results.keys())
                lcc_sizes = [strategy_results[f] for f in fractions]
                
                ax_scaled.plot(fractions, lcc_sizes, label=f"Scaled CM (seed {seed})",
                             linestyle='--', alpha=0.7)
                
                # Add percolation threshold
                if thresholds and key in thresholds:
                    p_c = thresholds[key]
                    f_c = 1 - p_c  # Convert to fraction of edges removed
                    ax_scaled.axvline(x=f_c, color='grey', linestyle='--', alpha=0.5,
                                   label=f'CM f_c = {f_c:.3f}')
        
        ax_scaled.set_xlabel('Fraction of Edges Removed')
        ax_scaled.set_ylabel('Relative Size of Largest Component')
        ax_scaled.set_title(f"{region} vs Scaled Configuration Models - {strategy}")
        ax_scaled.grid(True, alpha=0.3)
        ax_scaled.legend()
        ax_scaled.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.suptitle(f"Original Networks vs Configuration Models - {strategy} Attack", fontsize=16)
    plt.subplots_adjust(top=0.95)
    
    return fig

def plot_comprehensive_comparison(results, thresholds=None, timestamp=None):
    """
    Create a comprehensive visualization with panels for each brain region and attack type,
    comparing original networks with their configuration models.
    
    Parameters:
        results (dict): Results from percolation experiments
        thresholds (dict): Dictionary of percolation thresholds for each region
        timestamp (str): Timestamp for filename
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Set seaborn style for better aesthetics
    sns.set_style("whitegrid")
    
    # Identify original regions (exclude configuration models)
    original_regions = [r for r in results.keys() if '_cm_' not in r and 'scaled_cm' not in r]
    
    if len(original_regions) == 0:
        logger.warning("No original regions found for comprehensive comparison plot")
        return None
    
    # Attack strategies to visualize
    strategies = ['random', 'high_degree', 'low_degree']
    
    # Number of columns = number of attack strategies
    # Number of rows = number of brain regions
    n_cols = len(strategies)
    n_rows = len(original_regions)
    
    # Create figure with appropriate size
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*6, n_rows*5), dpi=100, 
                            sharex=True, sharey=True)
    
    # Handle case with only one region (axs not 2D)
    if n_rows == 1:
        axs = np.array([axs])
    
    # Color palette for consistent colors across plots
    region_palette = sns.color_palette("tab10", len(original_regions))
    model_types = ["Original", "Unscaled CM", "Scaled CM"]
    linestyles = ['-', '--', '-.']
    
    # Create a mapping from region name to color
    region_colors = dict(zip(original_regions, region_palette))
    
    # Iterate through regions and strategies to create subplots
    for row_idx, region in enumerate(original_regions):
        for col_idx, strategy in enumerate(strategies):
            ax = axs[row_idx, col_idx]
            
            # Add strategy name and region name as subplot title
            if strategy == 'random':
                strategy_name = 'Random Edge Removal'
            elif strategy == 'high_degree':
                strategy_name = 'High Degree First'
            else:  # low_degree
                strategy_name = 'Low Degree First'
                
            ax.set_title(f"{config.BRAIN_REGIONS.get(region, 'Unknown')} - {strategy_name}")
            
            # 1. Plot original network
            if region in results and strategy in results[region]:
                orig_results = results[region][strategy]
                fractions = sorted(orig_results.keys())
                lcc_sizes = [orig_results[f] for f in fractions]
                
                ax.plot(fractions, lcc_sizes, label=f"Original", 
                       color=region_colors[region], linestyle=linestyles[0], linewidth=2.5)
                
                # Add percolation threshold for original network
                if thresholds and region in thresholds:
                    p_c = thresholds[region]
                    f_c = 1 - p_c  # Convert to fraction of edges removed
                    ax.axvline(x=f_c, color=region_colors[region], linestyle=':', alpha=0.7,
                              label=f'Original f_c = {f_c:.3f}')
            
            # 2. Plot unscaled configuration models
            unscaled_models = [k for k in results.keys() if k.startswith(f"{region}_cm_")]
            for i, model_key in enumerate(unscaled_models):
                if strategy in results[model_key]:
                    # Extract seed from model key
                    seed = model_key.split('_')[-1]
                    
                    model_results = results[model_key][strategy]
                    fractions = sorted(model_results.keys())
                    lcc_sizes = [model_results[f] for f in fractions]
                    
                    # Lighter shade of the original color
                    color = region_colors[region]
                    ax.plot(fractions, lcc_sizes, 
                           label=f"Unscaled CM (seed {seed})",
                           color=color, linestyle=linestyles[1], alpha=0.7, linewidth=1.5)
                    
                    # Add percolation threshold if available
                    if thresholds and model_key in thresholds:
                        p_c = thresholds[model_key]
                        f_c = 1 - p_c  # Convert to fraction of edges removed
                        ax.axvline(x=f_c, color=color, linestyle='--', alpha=0.5,
                                  label=f'Unscaled CM f_c = {f_c:.3f}')
            
            # 3. Plot scaled configuration models
            scaled_models = [k for k in results.keys() if k.startswith(f"{region}_scaled_cm_")]
            for i, model_key in enumerate(scaled_models):
                if strategy in results[model_key]:
                    # Extract seed from model key
                    seed = model_key.split('_')[-1]
                    
                    model_results = results[model_key][strategy]
                    fractions = sorted(model_results.keys())
                    lcc_sizes = [model_results[f] for f in fractions]
                    
                    # Even lighter shade of the original color
                    color = region_colors[region]
                    ax.plot(fractions, lcc_sizes, 
                           label=f"Scaled CM (seed {seed})",
                           color=color, linestyle=linestyles[2], alpha=0.7, linewidth=1.5)
                    
                    # Add percolation threshold if available
                    if thresholds and model_key in thresholds:
                        p_c = thresholds[model_key]
                        f_c = 1 - p_c  # Convert to fraction of edges removed
                        ax.axvline(x=f_c, color=color, linestyle='-.', alpha=0.5,
                                  label=f'Scaled CM f_c = {f_c:.3f}')
            
            # Set axis labels
            ax.set_xlabel('Fraction of Edges Removed')
            ax.set_ylabel('Relative Size of Largest Component')
            
            # Set y-axis limits
            ax.set_ylim(0, 1.05)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend only to the rightmost column
            if col_idx == n_cols - 1:
                ax.legend(loc='upper right', fontsize='small')
            
            # Force integer ticks on x-axis
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a main title
    plt.suptitle("Comprehensive Percolation Analysis: Regions, Attack Strategies, and Models", 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    return fig

def save_percolation_results(results, thresholds=None, timestamp=None, output_dir=None):
    """
    Save percolation results to CSV and plots to figures.
    
    Parameters:
        results (dict): Results from percolation experiments
        thresholds (dict): Dictionary of percolation thresholds for each region
        timestamp (str): Timestamp for file naming
        output_dir (tuple): Tuple with (tables_dir, figures_dir) paths
        
    Returns:
        dict: Paths to saved files
    """
    import os
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Use provided output directories or default to config
    tables_dir = output_dir[0] if output_dir else config.TABLES_DIR
    figures_dir = output_dir[1] if output_dir else config.FIGURES_DIR
    
    # Ensure directories exist
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    saved_files = {}
    
    # Save results as CSV
    # First, convert nested dictionary to DataFrame
    rows = []
    for region, region_results in results.items():
        # Add threshold information
        p_c = thresholds.get(region) if thresholds else None
        
        # Determine region type and metadata
        if '_scaled_cm_' in region:
            # Scaled configuration model
            parts = region.split('_')
            original_region = parts[0]
            seed = parts[-1]
            region_type = 'Scaled CM'
            region_name = config.BRAIN_REGIONS.get(original_region, 'Unknown')
            region_seed = seed
        elif '_cm_' in region:
            # Unscaled configuration model
            parts = region.split('_')
            original_region = parts[0]
            seed = parts[-1]
            region_type = 'Unscaled CM'
            region_name = config.BRAIN_REGIONS.get(original_region, 'Unknown')
            region_seed = seed
        else:
            # Original network
            region_type = 'Original'
            region_name = config.BRAIN_REGIONS.get(region, 'Unknown')
            region_seed = 'N/A'
            original_region = region
        
        for strategy, strategy_results in region_results.items():
            for fraction, lcc_size in strategy_results.items():
                rows.append({
                    'Region': region,
                    'Region_Name': region_name,
                    'Region_Type': region_type,
                    'Seed': region_seed,
                    'Strategy': strategy,
                    'Fraction_Removed': fraction,
                    'LCC_Size': lcc_size,
                    'Percolation_Threshold': p_c
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    csv_path = os.path.join(tables_dir, f"percolation_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    saved_files['csv'] = csv_path
    logger.info(f"Saved results to {csv_path}")
    
    # Save threshold information separately for easier access
    if thresholds:
        threshold_rows = []
        for region, p_c in thresholds.items():
            # Determine region type and metadata
            if '_scaled_cm_' in region:
                # Scaled configuration model
                parts = region.split('_')
                original_region = parts[0]
                seed = parts[-1]
                region_type = 'Scaled CM'
                region_name = config.BRAIN_REGIONS.get(original_region, 'Unknown')
                region_seed = seed
            elif '_cm_' in region:
                # Unscaled configuration model
                parts = region.split('_')
                original_region = parts[0]
                seed = parts[-1]
                region_type = 'Unscaled CM'
                region_name = config.BRAIN_REGIONS.get(original_region, 'Unknown')
                region_seed = seed
            else:
                # Original network
                region_type = 'Original'
                region_name = config.BRAIN_REGIONS.get(region, 'Unknown')
                region_seed = 'N/A'
                original_region = region
                
            threshold_rows.append({
                'Region': region,
                'Original_Region': original_region,
                'Region_Name': region_name,
                'Region_Type': region_type,
                'Seed': region_seed,
                'Percolation_Threshold': p_c
            })
            
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_path = os.path.join(tables_dir, f"percolation_thresholds_{timestamp}.csv")
        threshold_df.to_csv(threshold_path, index=False)
        saved_files['thresholds'] = threshold_path
        logger.info(f"Saved percolation thresholds to {threshold_path}")
    
    # Only save the strategy plot and comprehensive comparison plot
    
    # 1. By strategy (original networks only)
    fig_strategy = plot_percolation_results(results, thresholds, plot_type="strategy")
    strategy_path = os.path.join(figures_dir, f"percolation_by_strategy_{timestamp}.png")
    fig_strategy.savefig(strategy_path, dpi=300, bbox_inches='tight')
    saved_files['percolation_by_strategy'] = strategy_path
    logger.info(f"Saved strategy plot to {strategy_path}")
    
    # 2. Comprehensive comparison plot
    fig_comprehensive = plot_comprehensive_comparison(results, thresholds, timestamp)
    if fig_comprehensive:
        comprehensive_path = os.path.join(figures_dir, f"comprehensive_comparison_{timestamp}.png")
        fig_comprehensive.savefig(comprehensive_path, dpi=300, bbox_inches='tight')
        saved_files['comprehensive_comparison'] = comprehensive_path
        logger.info(f"Saved comprehensive comparison plot to {comprehensive_path}")
        plt.close(fig_comprehensive)
    
    # Close figures to free memory
    plt.close(fig_strategy)
    
    return saved_files 