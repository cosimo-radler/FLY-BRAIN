"""
Preprocessing module for the Drosophila connectome analysis pipeline.

This module provides functions for cleaning and preprocessing the raw
connectome data, including removing self-loops, extracting the largest
connected component, and normalizing edge weights.
"""

import os
import logging
import networkx as nx
import pandas as pd
import numpy as np
from tqdm import tqdm

from . import config

logger = logging.getLogger("fly_brain")

def remove_self_loops(G):
    """
    Remove self-loops from a graph.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        nx.Graph: Graph with self-loops removed
    """
    # Count self-loops
    self_loops = list(nx.selfloop_edges(G))
    logger.info(f"Removing {len(self_loops)} self-loops from graph")
    
    # Create a new graph without self-loops
    G_no_loops = G.copy()
    G_no_loops.remove_edges_from(self_loops)
    
    return G_no_loops

def extract_largest_connected_component(G):
    """
    Extract the largest connected component from a graph.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        nx.Graph: Largest connected component subgraph
    """
    if nx.is_directed(G):
        # For directed graphs, use weakly connected components
        components = list(nx.weakly_connected_components(G))
    else:
        # For undirected graphs, use connected components
        components = list(nx.connected_components(G))
    
    # Sort components by size (largest first)
    components.sort(key=len, reverse=True)
    
    if len(components) > 1:
        largest_component = components[0]
        logger.info(f"Extracted largest connected component with {len(largest_component)} nodes "
                   f"from graph with {len(components)} components")
        return G.subgraph(largest_component).copy()
    else:
        logger.info("Graph is already connected")
        return G.copy()

def normalize_edge_weights(G, method='log'):
    """
    Normalize edge weights in a graph.
    
    Parameters:
        G (nx.Graph): Input graph
        method (str): Normalization method ('linear', 'log', or 'binary')
    
    Returns:
        nx.Graph: Graph with normalized edge weights
    """
    G_norm = G.copy()
    
    if method == 'binary':
        # Convert to binary weights (1 for all edges)
        for u, v, d in G_norm.edges(data=True):
            d['weight_original'] = d['weight']
            d['weight'] = 1
        logger.info("Converted edge weights to binary (1)")
        
    elif method == 'log':
        # Log-transform weights
        for u, v, d in G_norm.edges(data=True):
            d['weight_original'] = d['weight']
            d['weight'] = np.log1p(d['weight'])  # log(1+x) to handle weights of 0
        logger.info("Applied log-transform to edge weights")
        
    elif method == 'linear':
        # Min-max normalization to [0,1]
        weights = [d['weight'] for _, _, d in G_norm.edges(data=True)]
        if not weights:
            return G_norm
            
        min_weight = min(weights)
        max_weight = max(weights)
        
        if max_weight == min_weight:
            logger.warning("All weights are identical, normalization not applied")
            return G_norm
            
        for u, v, d in G_norm.edges(data=True):
            d['weight_original'] = d['weight']
            d['weight'] = (d['weight'] - min_weight) / (max_weight - min_weight)
        logger.info(f"Normalized edge weights to [0,1] range (min={min_weight}, max={max_weight})")
        
    else:
        logger.warning(f"Unknown normalization method: {method}. No normalization applied.")
        
    return G_norm

def threshold_weak_connections(G, threshold=config.CONNECTIVITY_THRESHOLD):
    """
    Remove edges with weights below a threshold.
    
    Parameters:
        G (nx.Graph): Input graph
        threshold (float): Minimum weight to keep
    
    Returns:
        nx.Graph: Graph with weak connections removed
    """
    G_thresholded = G.copy()
    edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] < threshold]
    
    G_thresholded.remove_edges_from(edges_to_remove)
    logger.info(f"Removed {len(edges_to_remove)} edges with weight < {threshold}")
    
    return G_thresholded

def relabel_nodes_sequential(G):
    """
    Relabel nodes with sequential integers.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        nx.Graph: Graph with relabeled nodes
        dict: Mapping from new labels to original labels
    """
    # Create mapping from original to sequential IDs
    nodes = list(G.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    reverse_mapping = {i: node for i, node in enumerate(nodes)}
    
    # Apply relabeling
    G_relabeled = nx.relabel_nodes(G, mapping)
    logger.info(f"Relabeled {len(nodes)} nodes with sequential integers")
    
    return G_relabeled, reverse_mapping

def downsample_network(G, target_size=None, sampling_fraction=None, preserve_largest_connected=True):
    """
    Downsample a network by randomly selecting a subset of nodes.
    
    Parameters:
        G (nx.Graph): Input graph
        target_size (int, optional): Target number of nodes
        sampling_fraction (float, optional): Fraction of nodes to keep (0.0-1.0)
        preserve_largest_connected (bool): Whether to extract the largest connected component after downsampling
    
    Returns:
        nx.Graph: Downsampled graph
    """
    if target_size is None and sampling_fraction is None:
        raise ValueError("Either target_size or sampling_fraction must be specified")
    
    original_size = G.number_of_nodes()
    
    if target_size is not None:
        if target_size >= original_size:
            logger.warning(f"Target size {target_size} >= original size {original_size}, no downsampling performed")
            return G.copy()
        sampling_fraction = target_size / original_size
    
    # Randomly select nodes to keep
    nodes = list(G.nodes())
    nodes_to_keep = np.random.choice(nodes, int(original_size * sampling_fraction), replace=False)
    
    # Create a subgraph with only the selected nodes
    G_downsampled = G.subgraph(nodes_to_keep).copy()
    
    logger.info(f"Downsampled graph from {original_size} to {G_downsampled.number_of_nodes()} nodes "
               f"({sampling_fraction:.2%} of original)")
    
    # Extract largest connected component if requested
    if preserve_largest_connected:
        G_downsampled = extract_largest_connected_component(G_downsampled)
    
    return G_downsampled

def clean_network(G, remove_loops=True, extract_lcc=True, normalize=None, threshold=None):
    """
    Apply a series of cleaning operations to a network.
    
    Parameters:
        G (nx.Graph): Input graph
        remove_loops (bool): Whether to remove self-loops
        extract_lcc (bool): Whether to extract the largest connected component
        normalize (str, optional): Normalization method ('linear', 'log', or 'binary')
        threshold (float, optional): Minimum weight threshold
    
    Returns:
        nx.Graph: Cleaned network
    """
    G_clean = G.copy()
    
    # Apply cleaning operations in a sensible order
    if remove_loops:
        G_clean = remove_self_loops(G_clean)
    
    if threshold is not None:
        G_clean = threshold_weak_connections(G_clean, threshold)
    
    if normalize is not None:
        G_clean = normalize_edge_weights(G_clean, method=normalize)
    
    if extract_lcc:
        G_clean = extract_largest_connected_component(G_clean)
    
    logger.info(f"Cleaned graph: {G.number_of_nodes()} nodes -> {G_clean.number_of_nodes()} nodes, "
               f"{G.number_of_edges()} edges -> {G_clean.number_of_edges()} edges")
    
    return G_clean

def save_cleaned_network(G, region, output_dir=config.DATA_PROCESSED_DIR):
    """
    Save a cleaned network to the processed data directory.
    
    Parameters:
        G (nx.Graph): Cleaned network
        region (str): Brain region code
        output_dir (Path): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{region.lower()}_cleaned.gexf")
    
    # Save the network
    nx.write_gexf(G, output_file)
    
    logger.info(f"Saved cleaned network to {output_file}")
    return output_file 