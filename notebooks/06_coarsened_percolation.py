#!/usr/bin/env python3
"""
Script to compare bond percolation between original networks and coarsened networks.

This script runs bond percolation analysis on both the original cleaned networks and
their coarsened versions, comparing how they respond to random and targeted attacks.
It generates a multipanel figure showing the results across different brain regions.
"""

import os
import sys
import time
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
from src import config
from src import percolation
from src import data_io
import src.utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/coarsened_percolation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("coarsened_percolation")

def load_coarsened_networks(regions=None):
    """
    Load coarsened networks from the Coarsened Networks directory.
    
    Parameters:
        regions (list): List of region codes to load. If None, load all regions.
        
    Returns:
        dict: Dictionary with region codes as keys and coarsened networks as values
    """
    # Get regions from config if not specified
    if regions is None:
        regions = config.BRAIN_REGIONS.keys()
    
    networks = {}
    
    for region in regions:
        try:
            # Construct path to coarsened network file
            network_path = os.path.join("data", "Coarsened Networks (0.5)", 
                                      f"{region.lower()}_cleaned_coarsened.gexf")
            
            if os.path.exists(network_path):
                network = nx.read_gexf(network_path)
                networks[f"{region}_coarsened"] = network
                logger.info(f"Loaded {region} coarsened network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            else:
                logger.warning(f"Coarsened network file not found for region {region}: {network_path}")
                
        except Exception as e:
            logger.error(f"Error loading coarsened network for region {region}: {str(e)}")
    
    return networks

def load_original_networks(regions=None):
    """
    Load all cleaned original networks from the processed directory.
    
    Parameters:
        regions (list): List of region codes to load. If None, load all regions.
        
    Returns:
        dict: Dictionary with region codes as keys and networks as values
    """
    # Get regions from config if not specified
    if regions is None:
        regions = config.BRAIN_REGIONS.keys()
    
    networks = {}
    
    for region in regions:
        try:
            # Construct path to cleaned network file
            network_path = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_cleaned.gexf")
            
            if os.path.exists(network_path):
                network = nx.read_gexf(network_path)
                networks[region] = network
                logger.info(f"Loaded {region} original network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            else:
                logger.warning(f"Network file not found for region {region}: {network_path}")
                
        except Exception as e:
            logger.error(f"Error loading network for region {region}: {str(e)}")
    
    return networks

def run_percolation_for_comparison(original_networks, coarsened_networks, fractions=None, seed=42, num_trials=3):
    """
    Run percolation experiments on both original and coarsened networks.
    
    Parameters:
        original_networks (dict): Dictionary of original networks
        coarsened_networks (dict): Dictionary of coarsened networks
        fractions (list): List of fractions of edges to remove
        seed (int): Random seed for reproducibility
        num_trials (int): Number of trials for random percolation
        
    Returns:
        tuple: (results, thresholds) dictionaries
    """
    if fractions is None:
        fractions = np.linspace(0.0, 1.0, 21)  # 21 points including 0.0 and 1.0
    
    # Combine networks
    all_networks = {}
    all_networks.update(original_networks)
    all_networks.update(coarsened_networks)
    
    # Run percolation experiments
    results = {}
    thresholds = {}
    
    # Calculate analytical thresholds
    for name, network in all_networks.items():
        thresholds[name] = percolation.compute_percolation_threshold(network)
    
    # Run experiments for each network and strategy
    for name, network in all_networks.items():
        results[name] = {}
        
        # Random percolation (average over multiple trials)
        random_results = {f: 0.0 for f in fractions}
        for trial in range(num_trials):
            trial_results = percolation.random_percolation(network, fractions, seed=seed+trial)
            for f, size in trial_results.items():
                random_results[f] += size
        
        # Average the results
        results[name]['random'] = {f: size/num_trials for f, size in random_results.items()}
        
        # Targeted percolation - high degree
        results[name]['degree_high'] = percolation.degree_based_percolation(network, fractions, method='high', seed=seed)
        
        # Targeted percolation - low degree
        results[name]['degree_low'] = percolation.degree_based_percolation(network, fractions, method='low', seed=seed)
        
        logger.info(f"Completed percolation analysis for {name}")
    
    return results, thresholds

def plot_comparison_figure(results, thresholds, regions=None, timestamp=None):
    """
    Create a multipanel figure comparing original vs coarsened networks.
    
    Parameters:
        results (dict): Percolation results dictionary
        thresholds (dict): Percolation thresholds dictionary
        regions (list): List of brain regions to include
        timestamp (str): Timestamp for file naming
        
    Returns:
        str: Path to saved figure
    """
    if regions is None:
        regions = config.BRAIN_REGIONS.keys()
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Define attack strategies
    strategies = ['random', 'degree_high', 'degree_low']
    strategy_titles = {
        'random': 'Random Edge Removal',
        'degree_high': 'High Degree First',
        'degree_low': 'Low Degree First'
    }
    
    # Define region-specific colors (similar to the comprehensive comparison plot)
    region_colors = {
        'MB': 'green',   # Mushroom Body
        'FB': 'orange',  # Fan-shaped Body
        'EB': 'blue',    # Ellipsoid Body
        'LH': 'red',     # Lateral Horn
        'AL': 'purple'   # Antennal Lobe
    }
    
    # Line styles and opacities
    line_styles = {
        'original': '-',     # Solid line for original
        'coarsened': '--'    # Dashed line for coarsened
    }
    line_widths = {
        'original': 2.0,
        'coarsened': 1.8
    }
    
    # Create a grid of subplots - regions as rows, strategies as columns
    fig, axes = plt.subplots(len(regions), len(strategies), 
                            figsize=(15, 4 * len(regions)), 
                            sharex=True, sharey=True)
    
    # If only one region, make axes 2D
    if len(regions) == 1:
        axes = np.array([axes])
    
    # Plot each combination of region and strategy
    for i, region in enumerate(regions):
        region_full_name = config.BRAIN_REGIONS.get(region, region)
        
        # Set the region color (default to black if not defined)
        region_color = region_colors.get(region, 'black')
        
        for j, strategy in enumerate(strategies):
            ax = axes[i, j]
            
            # Get network keys
            original_key = region
            coarsened_key = f"{region}_coarsened"
            
            # Skip if either network is missing
            if original_key not in results or coarsened_key not in results:
                ax.text(0.5, 0.5, f"Missing data for {region}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot the original network
            if strategy in results[original_key]:
                x_values = sorted(list(results[original_key][strategy].keys()))
                y_values = [results[original_key][strategy][x] for x in x_values]
                ax.plot(x_values, y_values, 
                        color=region_color, 
                        linestyle=line_styles['original'],
                        linewidth=line_widths['original'],
                        label=f"Original")
            
            # Plot the coarsened network
            if strategy in results[coarsened_key]:
                x_values = sorted(list(results[coarsened_key][strategy].keys()))
                y_values = [results[coarsened_key][strategy][x] for x in x_values]
                ax.plot(x_values, y_values, 
                        color=region_color, 
                        linestyle=line_styles['coarsened'],
                        linewidth=line_widths['coarsened'],
                        label=f"Coarsened")
            
            # Plot percolation thresholds
            if original_key in thresholds:
                ax.axvline(x=thresholds[original_key], color=region_color, linestyle=':', 
                          alpha=0.7, label=f"Orig. pc: {thresholds[original_key]:.3f}")
            
            if coarsened_key in thresholds:
                ax.axvline(x=thresholds[coarsened_key], color=region_color, linestyle='-.', 
                          alpha=0.7, label=f"Coars. pc: {thresholds[coarsened_key]:.3f}")
            
            # Set titles and labels
            if i == 0:  # Only show strategy titles on the top row
                ax.set_title(f"{strategy_titles[strategy]}", fontsize=12)
            
            if j == 0:  # Only show region names on the leftmost column
                ax.set_ylabel(f"{region_full_name}\nGiant Component Size", fontsize=12)
            
            # Add grid and set limits
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            
            # Set x-axis label only for bottom row
            if i == len(regions) - 1:
                ax.set_xlabel("Fraction of Edges Removed", fontsize=10)
            
            # Add legend only for the first plot
            if i == 0 and j == 0:
                ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    
    # Add an overall title
    plt.suptitle("Comparison of Bond Percolation: Original vs. Coarsened Networks", 
                fontsize=16, y=0.995)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save figure
    os.makedirs(os.path.join('results', 'figures'), exist_ok=True)
    output_path = os.path.join('results', 'figures', f'coarsened_percolation_comparison_{timestamp}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    
    return output_path

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Compare percolation between original and coarsened networks')
    parser.add_argument('--regions', nargs='+', help='Brain regions to process (e.g., MB FB EB)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials for random percolation')
    parser.add_argument('--steps', type=int, default=20, help='Number of percolation steps')
    args = parser.parse_args()
    
    logger.info("Starting percolation comparison analysis")
    logger.info(f"Regions: {args.regions}")
    logger.info(f"Seed: {args.seed}, Trials: {args.trials}, Steps: {args.steps}")
    
    try:
        # Define fractions to remove, include 1.0 for complete edge removal
        fractions = np.linspace(0.0, 1.0, args.steps + 1)
        
        # If no regions specified, use all available regions
        if args.regions is None:
            args.regions = list(config.BRAIN_REGIONS.keys())
            logger.info(f"No regions specified, using all regions: {args.regions}")
        
        # Load original networks
        original_networks = load_original_networks(regions=args.regions)
        
        # Load coarsened networks
        coarsened_networks = load_coarsened_networks(regions=args.regions)
        
        if not original_networks or not coarsened_networks:
            logger.error("No networks found. Please check data directories.")
            return
        
        # Filter to only regions that have both original and coarsened versions
        common_regions = []
        for region in args.regions:
            if region in original_networks and f"{region}_coarsened" in coarsened_networks:
                common_regions.append(region)
            else:
                logger.warning(f"Skipping {region} - missing original or coarsened network")
        
        if not common_regions:
            logger.error("No matching pairs of original and coarsened networks found.")
            return
        
        logger.info(f"Analyzing {len(common_regions)} regions: {common_regions}")
        
        # Run percolation experiments
        start_time = time.time()
        results, thresholds = run_percolation_for_comparison(
            original_networks, 
            coarsened_networks,
            fractions=fractions, 
            seed=args.seed, 
            num_trials=args.trials
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Percolation experiments completed in {total_time:.2f} seconds")
        
        # Generate and save the comparative figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        figure_path = plot_comparison_figure(
            results, 
            thresholds, 
            regions=common_regions,
            timestamp=timestamp
        )
        
        # Print summary
        print("\nPercolation Comparison Summary:")
        
        # Print percolation thresholds
        print("\nPercolation Thresholds:")
        for network_name, threshold in thresholds.items():
            print(f"  {network_name}: p_c = {threshold:.4f}")
        
        # Summarize robustness comparison
        print("\nRobustness Comparison (LCC size after 50% edge removal):")
        for region in common_regions:
            coarsened_region = f"{region}_coarsened"
            if region in results and coarsened_region in results:
                for strategy in ['random', 'degree_high', 'degree_low']:
                    orig_lcc = results[region][strategy].get(0.5, None)
                    coars_lcc = results[coarsened_region][strategy].get(0.5, None)
                    if orig_lcc is not None and coars_lcc is not None:
                        diff = coars_lcc - orig_lcc
                        better = "more" if diff > 0 else "less"
                        print(f"  {region} - {strategy}: Coarsened is {better} robust by {abs(diff):.3f}")
        
        print(f"\nSaved comparison figure to: {figure_path}")
        
    except Exception as e:
        logger.exception(f"Error in percolation comparison analysis: {str(e)}")
        
if __name__ == "__main__":
    main() 