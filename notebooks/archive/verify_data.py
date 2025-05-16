#!/usr/bin/env python3
"""
Script to verify the correctness of fetched neuPrint data.

This script loads the fetched data and computes basic statistics
to verify that the data is valid and reasonable.
"""

import os
import sys
import json
import logging
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src import config, utils, metrics

# Set up logging
logger = utils.setup_logging(None, logging.INFO)

def verify_neurons_file(region):
    """Verify the neurons JSON file for a brain region."""
    file_path = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_neurons.json")
    
    if not os.path.exists(file_path):
        logger.error(f"Neurons file not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r') as f:
            neurons = json.load(f)
        
        if not neurons:
            logger.error(f"Neurons file is empty: {file_path}")
            return False
        
        # Check if we have the expected fields in each neuron
        sample_neuron = neurons[0]
        required_fields = ['n.bodyId', 'n.type']
        missing_fields = [field for field in required_fields if field not in sample_neuron]
        
        if missing_fields:
            logger.error(f"Missing required fields in neurons file: {missing_fields}")
            return False
        
        logger.info(f"Verified neurons file for {region}: {len(neurons)} neurons")
        logger.info(f"Sample neuron: {sample_neuron}")
        
        # Report some statistics
        types = [n.get('n.type', 'unknown') for n in neurons]
        type_counts = {}
        for t in types:
            if t is not None:
                type_counts[t] = type_counts.get(t, 0) + 1
        
        logger.info(f"Found {len(type_counts)} unique neuron types")
        if type_counts:
            top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"Top 5 neuron types: {top_types}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying neurons file: {str(e)}")
        return False

def verify_connectivity_file(region):
    """Verify the connectivity CSV file for a brain region."""
    file_path = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    
    if not os.path.exists(file_path):
        logger.error(f"Connectivity file not found: {file_path}")
        return False
    
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            logger.error(f"Connectivity file is empty: {file_path}")
            return False
        
        # Check if we have the expected columns
        required_columns = ['source', 'target', 'weight']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in connectivity file: {missing_columns}")
            return False
        
        logger.info(f"Verified connectivity file for {region}: {len(df)} connections")
        logger.info(f"Sample connections:\n{df.head(3)}")
        
        # Report some statistics
        unique_sources = df['source'].nunique()
        unique_targets = df['target'].nunique()
        avg_weight = df['weight'].mean()
        max_weight = df['weight'].max()
        min_weight = df['weight'].min()
        
        logger.info(f"Unique source neurons: {unique_sources}")
        logger.info(f"Unique target neurons: {unique_targets}")
        logger.info(f"Average connection weight: {avg_weight:.2f}")
        logger.info(f"Min/Max connection weight: {min_weight}/{max_weight}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying connectivity file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_network_file(region):
    """Verify the network GEXF file for a brain region."""
    file_path = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_network.gexf")
    
    if not os.path.exists(file_path):
        logger.error(f"Network file not found: {file_path}")
        return False
    
    try:
        G = nx.read_gexf(file_path)
        
        if not G:
            logger.error(f"Network graph is empty: {file_path}")
            return False
        
        logger.info(f"Verified network file for {region}: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Compute basic metrics
        is_directed = nx.is_directed(G)
        is_connected = nx.is_strongly_connected(G) if is_directed else nx.is_connected(G)
        density = nx.density(G)
        avg_degree = sum(dict(G.degree()).values()) / len(G)
        
        logger.info(f"Is directed: {is_directed}")
        logger.info(f"Is {'strongly ' if is_directed else ''}connected: {is_connected}")
        logger.info(f"Network density: {density:.6f}")
        logger.info(f"Average degree: {avg_degree:.2f}")
        
        # Compute more advanced metrics if the graph is not too large
        if len(G) <= 1000:  # Avoid expensive computations for very large graphs
            avg_clustering = nx.average_clustering(G)
            logger.info(f"Average clustering coefficient: {avg_clustering:.6f}")
            
            # Compute average shortest path length for connected graphs
            if is_connected or len(list(nx.strongly_connected_components(G))[0]) > len(G) * 0.9:
                try:
                    avg_path_length = nx.average_shortest_path_length(G)
                    logger.info(f"Average shortest path length: {avg_path_length:.2f}")
                except:
                    logger.warning("Couldn't compute average shortest path length (graph may not be connected)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error verifying network file: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def verify_region(region):
    """Verify all data files for a brain region."""
    region_name = config.BRAIN_REGIONS.get(region, region)
    logger.info(f"\n=== Verifying data for {region_name} ({region}) ===")
    
    neuron_ok = verify_neurons_file(region)
    connectivity_ok = verify_connectivity_file(region)
    network_ok = verify_network_file(region)
    
    return neuron_ok and connectivity_ok and network_ok

def main():
    """Verify data for all brain regions."""
    import logging
    global logger
    logger = utils.setup_logging(None, logging.INFO)
    
    logger.info("Starting data verification")
    
    # Verify each region
    all_ok = True
    for region in config.BRAIN_REGIONS:
        ok = verify_region(region)
        if not ok:
            all_ok = False
    
    if all_ok:
        logger.info("\nAll data verified successfully!")
    else:
        logger.error("\nSome data verification failed. Check logs for details.")
    
    return all_ok

if __name__ == "__main__":
    import logging
    success = main()
    sys.exit(0 if success else 1) 