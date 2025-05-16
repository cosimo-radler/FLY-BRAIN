# This file has been archived.
# Please use neuprint_client.py for data retrieval.

"""
Data input/output operations for the Drosophila connectome analysis pipeline.

This module handles loading and saving network data, including original graphs,
cleaned graphs, and generated null models.
"""

import os
import logging
import networkx as nx
from datetime import datetime
import json
from pathlib import Path

import src.config as config

logger = logging.getLogger("fly_brain")

def load_network(region, processed=True):
    """
    Load a network for a given brain region.
    
    Parameters:
        region (str): Brain region code (e.g., 'MB', 'AL')
        processed (bool): Whether to load the processed version (default) or raw
        
    Returns:
        nx.Graph: The loaded network
        
    Raises:
        FileNotFoundError: If the network file does not exist
    """
    if processed:
        file_path = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_cleaned.gexf")
        network_type = "processed"
    else:
        file_path = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_network.gexf")
        network_type = "raw"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No {network_type} network file found for region {region}: {file_path}")
    
    G = nx.read_gexf(file_path)
    logger.info(f"Loaded {network_type} {region} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G

def save_null_model(G, region, model_type, seed=None, n_target=None):
    """
    Save a null model to the appropriate location.
    
    Parameters:
        G (nx.Graph): The null model graph to save
        region (str): Brain region code
        model_type (str): Type of null model (e.g., 'configuration', 'scaled')
        seed (int, optional): Random seed used to generate the model
        n_target (int, optional): Target node count for scaled models
        
    Returns:
        str: Path to the saved file
    """
    # Create directory for null models if it doesn't exist
    null_model_dir = os.path.join(config.DATA_PROCESSED_DIR, "null_models")
    os.makedirs(null_model_dir, exist_ok=True)
    
    # Create model-specific filename
    filename_parts = [region.lower(), model_type]
    
    if n_target:
        filename_parts.append(f"n{n_target}")
    
    if seed is not None:
        filename_parts.append(f"seed{seed}")
    
    filename = "_".join(filename_parts) + ".gexf"
    file_path = os.path.join(null_model_dir, filename)
    
    # Save the network
    nx.write_gexf(G, file_path)
    logger.info(f"Saved {model_type} model for {region} to {file_path}")
    
    # Update model metadata
    update_model_metadata(region, {
        'model_type': model_type,
        'seed': seed,
        'n_target': n_target,
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'timestamp': datetime.now().isoformat(),
        'file_path': file_path
    })
    
    return file_path

def update_model_metadata(region, model_info):
    """
    Update the metadata file with information about generated models.
    
    Parameters:
        region (str): Brain region code
        model_info (dict): Information about the generated model
    """
    metadata_file = os.path.join(config.DATA_PROCESSED_DIR, "null_models", "metadata.json")
    
    # Load existing metadata if available
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse metadata file {metadata_file}, creating new one")
    
    # Initialize region entry if needed
    if region not in metadata:
        metadata[region] = {'models': []}
    
    # Add new model info
    metadata[region]['models'].append(model_info)
    
    # Save updated metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

def load_null_model(region, model_type, seed=None, n_target=None):
    """
    Load a previously generated null model.
    
    Parameters:
        region (str): Brain region code
        model_type (str): Type of null model
        seed (int, optional): Random seed used to generate the model
        n_target (int, optional): Target node count for scaled models
        
    Returns:
        nx.Graph or None: The loaded null model, or None if not found
    """
    # Construct the expected filename
    filename_parts = [region.lower(), model_type]
    
    if n_target:
        filename_parts.append(f"n{n_target}")
    
    if seed is not None:
        filename_parts.append(f"seed{seed}")
    
    filename = "_".join(filename_parts) + ".gexf"
    file_path = os.path.join(config.DATA_PROCESSED_DIR, "null_models", filename)
    
    if not os.path.exists(file_path):
        logger.warning(f"No {model_type} model found for {region} with the specified parameters")
        return None
    
    G = nx.read_gexf(file_path)
    logger.info(f"Loaded {model_type} model for {region}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    return G
