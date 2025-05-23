# This file has been archived.
# Please use neuprint_client.py for data retrieval.

"""
Data I/O Module

This module handles all data input/output operations for the brain network analysis:
- Loading raw brain network data
- Loading processed graph data
- Saving results to appropriate locations
"""

import os
import logging
import networkx as nx
import pandas as pd
import numpy as np
import glob
from datetime import datetime
import json
from pathlib import Path
import matplotlib.pyplot as plt

import src.config as config

logger = logging.getLogger("fly_brain")

# Define project paths
PROJECT_ROOT = Path(__file__).parents[1].resolve()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_COARSENED = PROJECT_ROOT / "data" / "Coarsened Networks (0.5)"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES = RESULTS_DIR / "tables"
RESULTS_FIGURES = RESULTS_DIR / "figures"

def get_brain_regions():
    """
    Get list of available brain regions from processed data directory.
    
    Returns:
        list: Names of available brain regions
    """
    # Look for cleaned GEXF files in processed directory
    gexf_files = list(DATA_PROCESSED.glob("*_cleaned.gexf"))
    region_names = [f.stem.split('_')[0].upper() for f in gexf_files]
    
    # If no GEXF files, look for other formats
    if not region_names:
        region_files = list(DATA_PROCESSED.glob("*.graphml")) + list(DATA_PROCESSED.glob("*.gml"))
        region_names = [f.stem.split('_')[0].upper() for f in region_files]
    
    # Also look for directories that might contain region data
    region_dirs = [d for d in DATA_PROCESSED.glob("*") if d.is_dir() and d.name != "null_models"]
    for d in region_dirs:
        if d.name.upper() not in region_names:
            region_names.append(d.name.upper())
    
    return sorted(region_names)

def get_available_models(brain_region):
    """
    Get list of available network models for a given brain region.
    
    Args:
        brain_region (str): Name of the brain region
        
    Returns:
        list: Names of available models (original, configuration models, coarsened, etc.)
    """
    brain_region = brain_region.lower()  # Convert to lowercase for file matching
    
    # First check for original cleaned files directly in data/processed
    model_names = []
    
    # GEXF files have format like "al_cleaned.gexf"
    cleaned_path = DATA_PROCESSED / f"{brain_region}_cleaned.gexf"
    if cleaned_path.exists():
        model_names.append("original")
        logger.info(f"Found original model for {brain_region} at {cleaned_path}")
    
    # Check for coarsened models in data/Coarsened Networks (0.5)
    coarsened_path = DATA_COARSENED / f"{brain_region}_cleaned_coarsened.gexf"
    if coarsened_path.exists():
        model_names.append("coarsened")
        logger.info(f"Found coarsened model for {brain_region} at {coarsened_path}")
    
    # Check for other model types in the data/processed/null_models directory
    null_model_dir = DATA_PROCESSED / "null_models"
    if null_model_dir.is_dir():
        # CM files
        cm_files = list(null_model_dir.glob(f"{brain_region}_cm*.gexf")) + \
                  list(null_model_dir.glob(f"{brain_region}_configuration*.gexf"))
        if cm_files:
            model_names.append("configuration_model")
            logger.info(f"Found configuration models for {brain_region} in null_models directory")
            
        # SSM files
        ssm_files = list(null_model_dir.glob(f"{brain_region}_ssm*.gexf")) + \
                   list(null_model_dir.glob(f"{brain_region}_spectral*.gexf"))
        if ssm_files:
            model_names.append("spectral_sparsifier")
            logger.info(f"Found spectral sparsifier models for {brain_region} in null_models directory")
    
    # Ensure every region has at least the original model if the GEXF file exists
    if cleaned_path.exists() and "original" not in model_names:
        model_names.append("original")
    
    return sorted(model_names)

def load_graph(brain_region, model_type="original"):
    """
    Load a specific graph for a brain region and model type.
    
    Args:
        brain_region (str): Name of the brain region (e.g., 'AL', 'MB')
        model_type (str): Type of model (original, configuration_model, spectral_sparsifier, coarsened)
        
    Returns:
        networkx.Graph: The loaded graph
    """
    brain_region = brain_region.lower()  # Convert to lowercase for file matching
    
    # Build potential file paths based on conventions
    potential_paths = []
    
    # For original models, first try the standardized cleaned GEXF files
    if model_type == "original":
        potential_paths.append(DATA_PROCESSED / f"{brain_region}_cleaned.gexf")
    
    # For coarsened models
    elif model_type == "coarsened":
        potential_paths.append(DATA_COARSENED / f"{brain_region}_cleaned_coarsened.gexf")
    
    # Check in region-specific subdirectory
    region_dir = DATA_PROCESSED / brain_region
    if region_dir.is_dir():
        if model_type == "original":
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_original*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_original*.gml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_original*.gexf")))
            potential_paths.extend(list(region_dir.glob(f"original*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"original*.gml")))
            potential_paths.extend(list(region_dir.glob(f"original*.gexf")))
        elif model_type == "configuration_model":
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_cm*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_cm*.gml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_cm*.gexf")))
            potential_paths.extend(list(region_dir.glob(f"cm*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"cm*.gml")))
            potential_paths.extend(list(region_dir.glob(f"cm*.gexf")))
        elif model_type == "spectral_sparsifier":
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_ssm*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_ssm*.gml")))
            potential_paths.extend(list(region_dir.glob(f"{brain_region}_ssm*.gexf")))
            potential_paths.extend(list(region_dir.glob(f"ssm*.graphml")))
            potential_paths.extend(list(region_dir.glob(f"ssm*.gml")))
            potential_paths.extend(list(region_dir.glob(f"ssm*.gexf")))
    
    # Also check in main processed directory and null_models directory
    if model_type == "configuration_model":
        null_models_dir = DATA_PROCESSED / "null_models"
        if null_models_dir.is_dir():
            potential_paths.extend(list(null_models_dir.glob(f"{brain_region}_configuration*.gexf")))
            potential_paths.extend(list(null_models_dir.glob(f"{brain_region}_cm*.gexf")))
    
    # Try to load from any of the potential paths
    for path in potential_paths:
        try:
            logger.info(f"Trying to load graph from {path}")
            if path.suffix == ".graphml":
                G = nx.read_graphml(path)
            elif path.suffix == ".gml":
                G = nx.read_gml(path)
            elif path.suffix == ".gexf":
                G = nx.read_gexf(path)
            else:
                continue
                
            logger.info(f"Successfully loaded graph with {len(G)} nodes and {G.number_of_edges()} edges")
            return G
        except Exception as e:
            logger.warning(f"Failed to load graph from {path}: {str(e)}")
    
    # Handle the case when no graph is found
    logger.error(f"No graph found for region {brain_region} and model {model_type}")
    return None

def load_coarsened_network(brain_region):
    """
    Load a coarsened network for a given brain region.
    
    Args:
        brain_region (str): Name of the brain region (e.g., 'AL', 'MB')
        
    Returns:
        networkx.Graph or None: The loaded coarsened network, or None if not found
    """
    brain_region = brain_region.lower()  # Convert to lowercase for file matching
    
    # Define path to coarsened network file
    coarsened_path = DATA_COARSENED / f"{brain_region}_cleaned_coarsened.gexf"
    
    # Try to load the coarsened network
    try:
        if coarsened_path.exists():
            logger.info(f"Loading coarsened network for {brain_region} from {coarsened_path}")
            G = nx.read_gexf(coarsened_path)
            logger.info(f"Successfully loaded coarsened network with {len(G)} nodes and {G.number_of_edges()} edges")
            return G
        else:
            logger.warning(f"Coarsened network file not found for {brain_region}")
            return None
    except Exception as e:
        logger.error(f"Failed to load coarsened network for {brain_region}: {str(e)}")
        return None

def load_null_models(brain_region, model_type="configuration_model", max_models=10):
    """
    Load multiple instances of null models for a given brain region.
    
    Args:
        brain_region (str): Name of the brain region
        model_type (str): Type of model (configuration_model, spectral_sparsifier)
        max_models (int): Maximum number of model instances to load
        
    Returns:
        list: List of networkx.Graph objects
    """
    # Build search pattern based on model type
    if model_type == "configuration_model":
        pattern = f"{brain_region}_cm_*"
    elif model_type == "spectral_sparsifier":
        pattern = f"{brain_region}_ssm_*"
    else:
        logger.error(f"Invalid model type: {model_type}")
        return []
    
    # Check in region-specific subdirectory first
    region_dir = DATA_PROCESSED / brain_region / "null_models"
    if not region_dir.is_dir():
        region_dir = DATA_PROCESSED / "null_models"
    
    # Fall back to main processed directory if needed
    if not region_dir.is_dir():
        region_dir = DATA_PROCESSED
    
    # Search for model files
    model_files = list(region_dir.glob(f"{pattern}.graphml")) + list(region_dir.glob(f"{pattern}.gml"))
    
    # Load up to max_models
    graphs = []
    for i, path in enumerate(model_files[:max_models]):
        try:
            if path.suffix == ".graphml":
                G = nx.read_graphml(path)
            else:
                G = nx.read_gml(path)
            graphs.append(G)
            logger.info(f"Loaded {model_type} #{i+1} for {brain_region} with {len(G)} nodes")
        except Exception as e:
            logger.warning(f"Failed to load {path}: {str(e)}")
    
    return graphs

def save_metrics_results(metrics_df, name, brain_region=None, timestamp=None):
    """
    Save metrics results to the appropriate location.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing the metrics
        name (str): Base name for the saved file
        brain_region (str, optional): Name of brain region if applicable
        timestamp (str, optional): Timestamp to include in filename
    """
    # Create directory if it doesn't exist
    RESULTS_TABLES.mkdir(parents=True, exist_ok=True)
    
    # Build filename
    if brain_region:
        filename = f"{brain_region}_{name}"
    else:
        filename = name
    
    if timestamp:
        filename = f"{filename}_{timestamp}"
    
    # Save as CSV
    output_path = RESULTS_TABLES / f"{filename}.csv"
    metrics_df.to_csv(output_path, index=True)
    logger.info(f"Saved metrics to {output_path}")
    
    return output_path

def load_all_brain_regions():
    """
    Attempt to load all available brain regions.
    
    Returns:
        dict: Dictionary mapping region names to their original graphs
    """
    regions = get_brain_regions()
    graphs = {}
    
    for region in regions:
        G = load_graph(region, "original")
        if G is not None:
            graphs[region] = G
    
    return graphs

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

def save_figure(fig, filepath, dpi=300, bbox_inches='tight', format=None):
    """
    Save a matplotlib figure to file with standard parameters.
    
    Args:
        fig (matplotlib.figure.Figure): Figure to save
        filepath (str or Path): Path where to save the figure
        dpi (int): Resolution (dots per inch)
        bbox_inches (str): Bounding box parameter
        format (str): File format override (if None, inferred from extension)
    
    Returns:
        str: Path to the saved file
    """
    # Ensure the directory exists
    output_dir = os.path.dirname(filepath)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the figure
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, format=format)
        logger.info(f"Figure saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save figure to {filepath}: {e}")
        return None
