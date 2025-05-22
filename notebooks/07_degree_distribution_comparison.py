#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Degree distribution comparison script for fly brain network models.
Compares original, unscaled config model, downscaled config model, and coarsened networks.
"""

import os
import sys
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
import argparse

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src import config
from src import utils
from src import data_io

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(config.LOGS_DIR, f"degree_distribution_comparison_{timestamp}.log")
logger = utils.setup_logging(log_file)

# Define paths
COARSENED_PATH = os.path.join(config.DATA_DIR, "Coarsened Networks (0.5)")
NULL_MODELS_PATH = os.path.join(config.DATA_PROCESSED_DIR, "null_models")

def compute_normalized_degree_dist(G):
    """
    Compute normalized degree distribution for a graph.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph
        
    Returns:
    --------
    tuple
        (degrees, frequencies) where frequencies are normalized by network size
    """
    if G is None or len(G) == 0:
        logger.warning("Empty graph provided to compute_normalized_degree_dist")
        return [], []
    
    degrees = sorted([d for n, d in G.degree()])
    degree_count = {}
    
    for d in degrees:
        if d in degree_count:
            degree_count[d] += 1
        else:
            degree_count[d] = 1
    
    # Sort by degree
    unique_degrees = sorted(degree_count.keys())
    frequencies = [degree_count[d] / len(G) for d in unique_degrees]
    
    return unique_degrees, frequencies

def plot_degree_distributions(networks, labels, output_path, region_name):
    """
    Plot normalized degree distributions for multiple networks.
    
    Parameters:
    -----------
    networks : list of networkx.Graph
        List of networks to compare
    labels : list of str
        Labels for each network
    output_path : str
        Path to save the output figure
    region_name : str
        Name of the brain region being analyzed
    """
    plt.figure(figsize=(14, 10))
    
    line_styles = ['-', '--', '-.', ':', '-', '--']
    markers = ['o', 's', '^', 'D', '*', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, (G, label) in enumerate(zip(networks, labels)):
        if G is None or len(G) == 0:
            logger.warning(f"Skipping empty graph: {label}")
            continue
            
        degrees, frequencies = compute_normalized_degree_dist(G)
        
        if not degrees:
            logger.warning(f"No degree distribution data for: {label}")
            continue
            
        plt.plot(
            degrees, 
            frequencies, 
            label=f"{label} (nodes: {len(G)}, edges: {G.number_of_edges()})",
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markersize=8,
            markevery=max(1, len(degrees) // 10),  # Show fewer markers for readability
            linewidth=2.5,
            alpha=0.8
        )
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.xlabel('Degree (k)', fontsize=16, fontweight='bold')
    plt.ylabel('Normalized Frequency P(k)', fontsize=16, fontweight='bold')
    plt.title(f'Normalized Degree Distribution Comparison - {region_name}', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Customize legend
    legend = plt.legend(fontsize=14, loc='upper right', framealpha=0.9, 
                      title="Network Models", title_fontsize=16)
    legend.get_frame().set_linewidth(1.5)
    
    # Add annotations for interpretation
    plt.annotate(
        f"Timestamp: {timestamp}",
        xy=(0.01, 0.01), xycoords='figure fraction',
        fontsize=10, color='gray'
    )
    
    # Adjust axis properties for better readability
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    
    # Add a textbox with network information
    info_text = f"{region_name} Brain Region Comparison\n"
    info_text += f"Original network has {len(networks[0])} nodes\n"
    info_text += f"Nodes ratio (Coarsened/Original): {len(networks[3])/len(networks[0]):.2f}\n"
    for i, (G, label) in enumerate(zip(networks, labels)):
        if G is None:
            continue
        avg_deg = 2 * G.number_of_edges() / len(G)
        info_text += f"{label}: Avg. degree = {avg_deg:.2f}\n"
    
    plt.figtext(0.02, 0.02, info_text, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved figure to {output_path}")
    
    # Save a second copy with timestamp in the filename
    timestamped_path = os.path.join(
        os.path.dirname(output_path), 
        f"{region_name}_degree_distribution_{timestamp}.png"
    )
    plt.savefig(timestamped_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """Main function to load networks and generate degree distribution comparison."""
    parser = argparse.ArgumentParser(description='Compare degree distributions across network models.')
    parser.add_argument('--region', type=str, default="FB", 
                        help='Brain region to analyze (e.g., FB, MB, AL)')
    args = parser.parse_args()
    
    region_name = args.region.upper()
    
    logger.info(f"Comparing degree distributions for region: {region_name}")
    
    try:
        # Load original network
        original_network = data_io.load_network(region_name)
        logger.info(f"Loaded original network with {len(original_network)} nodes and {original_network.number_of_edges()} edges")
        
        # Load unscaled configuration model (using first seed from config file)
        unscaled_cm = data_io.load_null_model(region_name, 'configuration', seed=config.RANDOM_SEED)
        if unscaled_cm:
            logger.info(f"Loaded unscaled CM with {len(unscaled_cm)} nodes and {unscaled_cm.number_of_edges()} edges")
        else:
            logger.warning(f"Unscaled CM not found for {region_name}")
        
        # Load downscaled configuration model
        downscaled_cm = data_io.load_null_model(region_name, 'scaled_configuration', 
                                                seed=config.RANDOM_SEED, n_target=1500)
        if downscaled_cm:
            logger.info(f"Loaded downscaled CM with {len(downscaled_cm)} nodes and {downscaled_cm.number_of_edges()} edges")
        else:
            logger.warning(f"Downscaled CM not found for {region_name}")
        
        # Load coarsened network
        coarsened_network = data_io.load_coarsened_network(region_name)
        if coarsened_network:
            logger.info(f"Loaded coarsened network with {len(coarsened_network)} nodes and {coarsened_network.number_of_edges()} edges")
        else:
            # Try direct loading as a fallback
            coarsened_path = os.path.join(COARSENED_PATH, f"{region_name.lower()}_coarsened.gpickle")
            if os.path.exists(coarsened_path):
                coarsened_network = nx.read_gpickle(coarsened_path)
                logger.info(f"Loaded coarsened network from {coarsened_path} with {len(coarsened_network)} nodes")
            else:
                logger.warning(f"Coarsened network not found for {region_name}")
        
        # Prepare list of networks and labels
        networks = [original_network, unscaled_cm, downscaled_cm, coarsened_network]
        labels = ["Original Network", "Unscaled Configuration Model", 
                  "Downscaled Configuration Model", "Coarsened Network (0.5)"]
        
        # Filter out None values
        valid_networks = []
        valid_labels = []
        for network, label in zip(networks, labels):
            if network is not None:
                valid_networks.append(network)
                valid_labels.append(label)
        
        if not valid_networks:
            logger.error("No valid networks found for comparison")
            return
        
        # Generate output path
        output_filename = f"{region_name}_degree_distribution_comparison.png"
        output_path = os.path.join(config.FIGURES_DIR, output_filename)
        
        # Plot degree distributions
        plot_degree_distributions(valid_networks, valid_labels, output_path, region_name)
        
    except Exception as e:
        logger.exception(f"Error in degree distribution comparison: {e}")

if __name__ == "__main__":
    main() 