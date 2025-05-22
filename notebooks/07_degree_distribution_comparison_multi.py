#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-region degree distribution comparison script for fly brain network models.
Creates a panel figure with brain regions as rows and network models as columns.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pathlib import Path
import multiprocessing

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src import config
from src import utils
from src import data_io

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(config.LOGS_DIR, f"panel_degree_distribution_{timestamp}.log")
logger = utils.setup_logging(log_file)

def compute_degree_distribution(G):
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
        logger.warning("Empty graph provided to compute_degree_distribution")
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

def generate_power_law(xmin, xmax, alpha, num_points=100):
    """
    Generate a power law distribution with exponent alpha.
    
    Parameters:
    -----------
    xmin : float
        Minimum x value
    xmax : float
        Maximum x value
    alpha : float
        Power law exponent
    num_points : int
        Number of points to generate
        
    Returns:
    --------
    tuple
        (x_values, y_values) for power law distribution
    """
    # Generate x values on log scale
    x_values = np.logspace(np.log10(xmin), np.log10(xmax), num=num_points)
    
    # Generate power law y values: p(x) ∝ x^(-alpha)
    y_values = x_values**(-alpha)
    
    # Normalize to have the same starting point as real data
    # We'll scale during plotting to match the actual data
    
    return x_values, y_values

def get_region_network_data(region_name):
    """
    Get degree distribution data for a brain region's networks without plotting.
    
    Parameters:
    -----------
    region_name : str
        Name of the brain region to analyze
        
    Returns:
    --------
    dict
        Dictionary with network data for the region
    """
    logger.info(f"Processing region: {region_name}")
    
    try:
        result = {
            'region': region_name,
            'networks': {},
            'stats': {}
        }
        
        # Load original network
        original_network = data_io.load_network(region_name)
        if original_network:
            logger.info(f"Loaded original network with {len(original_network)} nodes and {original_network.number_of_edges()} edges")
            degrees, freqs = compute_degree_distribution(original_network)
            result['networks']['original'] = {
                'graph': original_network,
                'degrees': degrees,
                'frequencies': freqs,
                'nodes': len(original_network),
                'edges': original_network.number_of_edges(),
                'avg_degree': 2 * original_network.number_of_edges() / len(original_network)
            }
        
        # Load unscaled configuration model
        unscaled_cm = data_io.load_null_model(region_name, 'configuration', seed=config.RANDOM_SEED)
        if unscaled_cm:
            logger.info(f"Loaded unscaled CM with {len(unscaled_cm)} nodes and {unscaled_cm.number_of_edges()} edges")
            degrees, freqs = compute_degree_distribution(unscaled_cm)
            result['networks']['config_model'] = {
                'graph': unscaled_cm,
                'degrees': degrees,
                'frequencies': freqs,
                'nodes': len(unscaled_cm),
                'edges': unscaled_cm.number_of_edges(),
                'avg_degree': 2 * unscaled_cm.number_of_edges() / len(unscaled_cm)
            }
        
        # Load coarsened network
        coarsened_network = data_io.load_coarsened_network(region_name)
        if not coarsened_network:
            # Try direct loading as a fallback
            coarsened_path = os.path.join(config.DATA_DIR, "Coarsened Networks (0.5)", 
                                       f"{region_name.lower()}_coarsened.gpickle")
            if os.path.exists(coarsened_path):
                coarsened_network = nx.read_gpickle(coarsened_path)
                
        if coarsened_network:
            logger.info(f"Loaded coarsened network with {len(coarsened_network)} nodes and {coarsened_network.number_of_edges()} edges")
            degrees, freqs = compute_degree_distribution(coarsened_network)
            result['networks']['coarsened'] = {
                'graph': coarsened_network,
                'degrees': degrees,
                'frequencies': freqs,
                'nodes': len(coarsened_network),
                'edges': coarsened_network.number_of_edges(),
                'avg_degree': 2 * coarsened_network.number_of_edges() / len(coarsened_network)
            }
                
        # Calculate statistics
        if 'original' in result['networks'] and 'coarsened' in result['networks']:
            result['stats']['coarsened_ratio'] = len(result['networks']['coarsened']['graph']) / len(result['networks']['original']['graph'])
            
        return result
    
    except Exception as e:
        logger.exception(f"Error processing region {region_name}: {e}")
        return {'region': region_name, 'error': str(e)}

def create_panel_figure(all_region_data):
    """
    Create a panel figure with subplots for all brain regions and network models.
    
    Parameters:
    -----------
    all_region_data : list
        List of dictionaries with network data for each region
        
    Returns:
    --------
    str
        Path to the generated panel figure
    """
    # Filter out regions with errors
    valid_region_data = [data for data in all_region_data if 'error' not in data]
    
    if not valid_region_data:
        logger.error("No valid region data available for panel figure")
        return None
    
    # Count number of valid regions for subplot layout
    n_regions = len(valid_region_data)
    
    # Define column models to show (always in this order if available)
    column_models = ['original', 'config_model', 'coarsened']
    n_models = len(column_models)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_regions, n_models, figsize=(n_models*5, n_regions*4), 
                             squeeze=False, sharex='col', sharey='row')
    
    # Define styling
    colors = {
        'original': '#1f77b4',
        'config_model': '#ff7f0e',
        'coarsened': '#2ca02c'
    }
    
    model_labels = {
        'original': 'Original Network',
        'config_model': 'Configuration Model',
        'coarsened': 'Coarsened Network (0.5)'
    }
    
    # Define power law exponents to show
    power_laws = [
        {'alpha': 2.1, 'color': '#FF9999', 'label': 'Power law α=2.1'},
        {'alpha': 2.5, 'color': '#99FF99', 'label': 'Power law α=2.5'},
        {'alpha': 2.9, 'color': '#9999FF', 'label': 'Power law α=2.9'}
    ]
    
    # Add plots for each region and model
    for i, region_data in enumerate(valid_region_data):
        region_name = region_data['region']
        networks = region_data['networks']
        
        # Add plots for each model (column)
        for j, model in enumerate(column_models):
            ax = axes[i, j]
            
            if model in networks:
                # Get data
                degrees = networks[model]['degrees']
                freqs = networks[model]['frequencies']
                nodes = networks[model]['nodes']
                edges = networks[model]['edges']
                avg_deg = networks[model]['avg_degree']
                
                if degrees and freqs:
                    # Plot degree distribution
                    ax.plot(degrees, freqs, 
                            label=f"{model_labels[model]}\n(n={nodes:,}, e={edges:,})",
                            color=colors[model], 
                            marker='o', 
                            markersize=3,
                            markevery=max(1, len(degrees) // 10),
                            linewidth=2)
                    
                    # Add power law overlays
                    if len(degrees) > 1:
                        xmin, xmax = min(degrees), max(degrees)
                        ymin, ymax = min(freqs), max(freqs)
                        
                        # Find a suitable scaling point - use the median degree value as reference
                        mid_idx = len(degrees) // 3  # Use the 1/3 point instead of median for better visibility
                        ref_x, ref_y = degrees[mid_idx], freqs[mid_idx]
                        
                        for power_law in power_laws:
                            # Generate power law points
                            x_values, y_values = generate_power_law(xmin, xmax, power_law['alpha'])
                            
                            # Scale to match at the reference point
                            scale_factor = ref_y / (ref_x**(-power_law['alpha']))
                            y_scaled = y_values * scale_factor
                            
                            # Plot power law overlay with translucency
                            if i == 0 and j == 0:  # Only add to legend once
                                ax.plot(x_values, y_scaled, '--', color=power_law['color'], 
                                        alpha=0.4, linewidth=1.5, label=power_law['label'])
                            else:
                                ax.plot(x_values, y_scaled, '--', color=power_law['color'], 
                                        alpha=0.4, linewidth=1.5)
                    
                    # Set log scales
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    
                    # Add model info
                    info_text = f"Avg degree: {avg_deg:.2f}"
                    ax.text(0.05, 0.05, info_text, transform=ax.transAxes, 
                            fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
                
                # Set title for first row only
                if i == 0:
                    ax.set_title(model_labels[model], fontsize=14, fontweight='bold')
                
                # Add region label on y-axis for first column only
                if j == 0:
                    ax.set_ylabel(f"{region_name}", fontsize=14, fontweight='bold')
                    
                # Add legend - only for the first panel to avoid clutter
                if i == 0 and j == 0:
                    ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
                else:
                    # For other panels, only show the network model in legend
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend([handles[0]], [labels[0]], loc='upper right', fontsize=9, framealpha=0.7)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
            else:
                # Model not available for this region
                ax.text(0.5, 0.5, f"No {model_labels[model]} available", 
                        ha='center', va='center', fontsize=12,
                        transform=ax.transAxes)
                ax.set_axis_off()
    
    # Add power law legend in a separate box at the bottom
    power_law_legend_elements = [
        plt.Line2D([0], [0], color=pl['color'], linestyle='--', alpha=0.7, label=pl['label'])
        for pl in power_laws
    ]
    
    fig.legend(handles=power_law_legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.01), ncol=len(power_laws), fontsize=12, 
               framealpha=0.7, title="Theoretical Power Laws")
    
    # Add common x and y labels
    fig.text(0.5, 0.02, 'Degree (k)', fontsize=16, fontweight='bold', ha='center')
    fig.text(0.01, 0.5, 'Normalized Frequency P(k)', fontsize=16, fontweight='bold', 
             va='center', rotation='vertical')
    
    # Add main title
    fig.suptitle('Degree Distribution Comparison Across Brain Regions and Models', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Add timestamp annotation
    fig.text(0.01, 0.01, f"Generated: {timestamp}", fontsize=8, color='gray')
    
    # Tight layout with padding for the bottom legend
    plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])
    
    # Save figure
    output_filename = f"panel_degree_distribution_{timestamp}.png"
    output_path = os.path.join(config.FIGURES_DIR, output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved panel figure to {output_path}")
    
    return output_path

def main():
    """Main function to create panel figure of degree distributions."""
    parser = argparse.ArgumentParser(description='Create panel figure of degree distributions for brain regions and models.')
    parser.add_argument('--regions', type=str, nargs='+', 
                      help='Brain regions to analyze (e.g., FB MB AL). If not specified, all available regions will be used.')
    parser.add_argument('--parallel', action='store_true',
                      help='Run data collection in parallel using multiple processes')
    
    args = parser.parse_args()
    
    # Determine regions to process
    if args.regions:
        regions = [r.upper() for r in args.regions]
    else:
        # Use all available regions from config
        regions = list(config.BRAIN_REGIONS.keys())
    
    logger.info(f"Analyzing degree distributions for regions: {', '.join(regions)}")
    
    # Collect data for all regions (either in parallel or sequentially)
    if args.parallel and len(regions) > 1:
        logger.info(f"Processing {len(regions)} regions in parallel")
        pool = multiprocessing.Pool()
        all_region_data = pool.map(get_region_network_data, regions)
        pool.close()
        pool.join()
    else:
        logger.info(f"Processing {len(regions)} regions sequentially")
        all_region_data = [get_region_network_data(region) for region in regions]
    
    # Create panel figure with all data
    logger.info("Creating panel figure")
    panel_path = create_panel_figure(all_region_data)
    
    if panel_path:
        logger.info(f"Panel figure created at: {panel_path}")
    else:
        logger.error("Failed to create panel figure")
    
    logger.info("Analysis complete")

if __name__ == "__main__":
    main() 