#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization of fly brain structures using neuprint-python API and navis.

This script demonstrates how to:
1. Connect to neuPrint server
2. Fetch neurons by brain region
3. Fetch skeleton data
4. Visualize neurons in 3D using navis
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path

# Add the src directory to the path so we can import modules
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'visualization.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import standard packages first
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Import local modules
from src.utils import set_seed
from src.data_io import save_figure
import src.config as config

# Import neuprint - try with the correct imports per documentation
try:
    from neuprint import Client, fetch_neurons, NeuronCriteria
    from neuprint.skeleton import fetch_skeleton
    print("Successfully imported neuprint modules")
    HAS_NEUPRINT = True
except ImportError as e:
    logger.error(f"Error importing neuprint-python: {str(e)}")
    print(f"Error importing neuprint-python: {str(e)}")
    HAS_NEUPRINT = False

# Import navis after neuprint to avoid conflicts
try:
    import navis
    import plotly.graph_objects as go
    print("Successfully imported navis")
    HAS_NAVIS = True
except ImportError as e:
    logger.error(f"Error importing navis: {str(e)}")
    print(f"Error importing navis: {str(e)}")
    HAS_NAVIS = False

def connect_to_neuprint():
    """
    Connect to neuPrint server.
    
    Returns:
        Client: A neuprint Client object
    """
    # Use token from config file if available
    try:
        # Get token from config file
        token = config.NEUPRINT_TOKEN
        server = config.NEUPRINT_SERVER
        dataset = config.NEUPRINT_DATASET
        
        if not token:
            logger.warning("NEUPRINT_TOKEN not set in config.py. Using guest mode with limited access.")
            token = 'neuprint-readonly'
        
        logger.info(f"Connecting to neuPrint server: {server}, dataset: {dataset}")
        # Create the client
        c = Client(server, dataset=dataset, token=token)
        logger.info(f"Connected to neuPrint server: {c.server}")
        return c
    except Exception as e:
        logger.error(f"Failed to connect to neuPrint server: {e}")
        logger.error(traceback.format_exc())
        raise

def fetch_brain_regions():
    """
    Fetch available brain regions from neuPrint.
    
    Returns:
        list: List of brain region names
    """
    c = connect_to_neuprint()
    
    # Fetch all available regions
    cypher_query = """
    MATCH (n:Meta)
    RETURN n.superLevelRois AS regions
    """
    results = c.fetch_custom(cypher_query)
    
    # Extract regions
    regions = results.iloc[0]['regions']
    logger.info(f"Found {len(regions)} brain regions")
    return regions

def fetch_neurons_by_region(region, limit=50):
    """
    Fetch neurons from a specific brain region.
    
    Args:
        region (str): Name of the brain region
        limit (int): Maximum number of neurons to fetch
    
    Returns:
        pd.DataFrame: DataFrame containing neuron data
    """
    c = connect_to_neuprint()
    
    logger.info(f"Fetching neurons from region: {region} (limit: {limit})")
    
    # Use the correct parameter name for ROI in NeuronCriteria
    # According to documentation, rois should be a list
    criteria = NeuronCriteria(rois=[region])
    logger.info(f"Created NeuronCriteria with rois=[{region}]")
    
    try:
        logger.info("Calling fetch_neurons...")
        # Remove the limit parameter - we'll handle it after fetching
        neurons_df, roi_counts_df = fetch_neurons(criteria, client=c)
        
        # Apply limit after fetching
        if limit and len(neurons_df) > limit:
            logger.info(f"Limiting results from {len(neurons_df)} to {limit} neurons")
            neurons_df = neurons_df.head(limit)
        
        logger.info(f"Fetched {len(neurons_df)} neurons from {region}")
        return neurons_df
    except Exception as e:
        logger.error(f"Error fetching neurons from {region}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def fetch_neuron_skeletons(body_ids):
    """
    Fetch skeleton data for a list of neuron body IDs.
    
    Args:
        body_ids (list): List of neuron body IDs
    
    Returns:
        dict: Dictionary of skeleton data by body ID
    """
    c = connect_to_neuprint()
    
    logger.info(f"Fetching skeletons for {len(body_ids)} neurons")
    skeletons = {}
    
    try:
        # Fetch each skeleton individually
        for body_id in tqdm(body_ids, desc="Fetching skeletons"):
            try:
                # Use the correct fetch_skeleton function from the skeleton module
                skeleton = fetch_skeleton(body_id, client=c)
                if skeleton is not None and not skeleton.empty:
                    skeletons[body_id] = skeleton
                    logger.info(f"Successfully fetched skeleton for body ID {body_id}")
                else:
                    logger.warning(f"Empty skeleton returned for body ID {body_id}")
            except Exception as e:
                logger.warning(f"Could not fetch skeleton for body ID {body_id}: {e}")
                # Don't retry - just log and continue to the next neuron
        
        if not skeletons:
            logger.warning("No skeletons were successfully fetched. The visualization will be empty.")
            print("No skeletons were successfully fetched. Try a different region or increase the limit.")
        else:
            logger.info(f"Successfully fetched {len(skeletons)} skeletons")
        
        return skeletons
    except Exception as e:
        logger.error(f"Error fetching skeletons: {e}")
        logger.error(traceback.format_exc())
        return {}

def parse_skeleton(skeleton):
    """
    Parse a skeleton into vertices and edges for navis visualization.
    
    Args:
        skeleton: Skeleton data from neuPrint API
        
    Returns:
        tuple: (vertices, edges) where vertices is a numpy array of shape (N, 3)
               and edges is a numpy array of shape (M, 2)
    """
    try:
        # Check if it's a pandas DataFrame (SWC format)
        if isinstance(skeleton, pd.DataFrame):
            logger.info("Converting SWC DataFrame to vertices and edges")
            
            # Prepare data for navis TreeNeuron
            if 'x' in skeleton.columns and 'y' in skeleton.columns and 'z' in skeleton.columns:
                vertices = skeleton[['x', 'y', 'z']].values
            else:
                # If no x,y,z columns, try sample_number and position_x,y,z columns
                vertices = skeleton[['position_x', 'position_y', 'position_z']].values
            
            # Build edges from parent information
            if 'parent_id' in skeleton.columns and 'sample_number' in skeleton.columns:
                # SWC format typically has parent_id and sample_number
                edges = []
                for i, row in skeleton.iterrows():
                    if row['parent_id'] >= 0:  # Skip root nodes (-1)
                        # Find indices in the DataFrame
                        child_idx = i
                        parent_idx = skeleton[skeleton['sample_number'] == row['parent_id']].index
                        if len(parent_idx) > 0:
                            edges.append([child_idx, parent_idx[0]])
                edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
            elif 'rowId' in skeleton.columns and 'link' in skeleton.columns:
                # Generic format with rowId and link
                edges = []
                for i, row in skeleton.iterrows():
                    if row['link'] >= 0:  # Skip root nodes (-1)
                        parent_idx = skeleton[skeleton['rowId'] == row['link']].index
                        if len(parent_idx) > 0:
                            edges.append([i, parent_idx[0]])
                edges = np.array(edges) if edges else np.empty((0, 2), dtype=int)
            else:
                logger.warning("Cannot determine edge structure from DataFrame")
                edges = np.empty((0, 2), dtype=int)
        
        # Check if it's a dict or has vertices/edges attributes
        elif isinstance(skeleton, dict) and 'vertices' in skeleton and 'edges' in skeleton:
            vertices = np.array(skeleton['vertices'])
            edges = np.array(skeleton['edges'])
        elif hasattr(skeleton, 'vertices') and hasattr(skeleton, 'edges'):
            vertices = np.array(skeleton.vertices)
            edges = np.array(skeleton.edges)
        else:
            logger.warning(f"Unknown skeleton format: {type(skeleton)}")
            return None, None
        
        return vertices, edges
    except Exception as e:
        logger.error(f"Error parsing skeleton: {e}")
        logger.error(traceback.format_exc())
        return None, None

def visualize_neurons(neurons_df, skeletons, region_name, save=True):
    """
    Visualize neurons using direct plotly visualization.
    
    Args:
        neurons_df (pd.DataFrame): DataFrame containing neuron data
        skeletons (dict): Dictionary of skeleton data by body ID
        region_name (str): Name of the brain region
        save (bool): Whether to save the figure
    
    Returns:
        None
    """
    if not skeletons:
        logger.warning("No skeletons to visualize")
        print("No skeletons to visualize. Try a different region or increase the limit.")
        return
    
    logger.info(f"Visualizing {len(skeletons)} neurons from {region_name}")
    
    # Use a direct plotly approach for visualization
    fig = go.Figure()
    
    # Create a body_id to row lookup for faster access
    body_id_to_row = {}
    if 'bodyId' in neurons_df.columns:
        for i, row in neurons_df.iterrows():
            body_id_to_row[row['bodyId']] = i
    
    # For each skeleton, add a scatter3d trace
    for bodyid, skeleton in skeletons.items():
        try:
            # Get neuron metadata
            if bodyid in body_id_to_row:
                row = neurons_df.iloc[body_id_to_row[bodyid]]
                neuron_type = row['type'] if 'type' in neurons_df.columns else 'unknown'
                neuron_name = row['instance'] if 'instance' in neurons_df.columns else str(bodyid)
            else:
                neuron_type = 'unknown'
                neuron_name = str(bodyid)
            
            # Get vertices
            vertices, edges = parse_skeleton(skeleton)
            
            if vertices is None or len(vertices) == 0:
                logger.warning(f"No valid vertices for body ID {bodyid}")
                continue
            
            # Generate a random color for this neuron
            color = f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 0.8)'
            
            # Add points for vertices
            fig.add_trace(go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=color,
                    opacity=0.5
                ),
                name=f"{neuron_name} ({neuron_type})"
            ))
            
            # Add lines for edges if available
            if edges is not None and len(edges) > 0:
                # Create line segments for each edge
                x_edges = []
                y_edges = []
                z_edges = []
                
                # Instead of individual traces for each edge, create a single trace with line breaks
                for edge in edges:
                    if edge[0] < len(vertices) and edge[1] < len(vertices):
                        # Add the vertices for this edge
                        x_edges.extend([vertices[edge[0], 0], vertices[edge[1], 0], None])
                        y_edges.extend([vertices[edge[0], 1], vertices[edge[1], 1], None])
                        z_edges.extend([vertices[edge[0], 2], vertices[edge[1], 2], None])
                
                # Add a single trace for all edges
                if x_edges:
                    fig.add_trace(go.Scatter3d(
                        x=x_edges,
                        y=y_edges,
                        z=z_edges,
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f"{neuron_name} edges",
                        showlegend=False
                    ))
            
            logger.info(f"Added neuron {neuron_name} to visualization")
        except Exception as e:
            logger.error(f"Error creating visualization for body ID {bodyid}: {e}")
            logger.error(traceback.format_exc())
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)',
            zaxis_title='Z (nm)',
        ),
        title=f"Neurons in {region_name}",
        width=1000,
        height=800,
        showlegend=True
    )
    
    # Display the figure
    fig.show()
    
    # Optionally save the figure
    if save:
        try:
            output_path = os.path.join(parent_dir, 'results', 'figures', f'neurons_{region_name}.html')
            fig.write_html(output_path)
            logger.info(f"Interactive figure saved to {output_path}")
            
            # Also save as PNG
            png_path = os.path.join(parent_dir, 'results', 'figures', f'neurons_{region_name}.png')
            fig.write_image(png_path)
            logger.info(f"Static figure saved to {png_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            logger.error(traceback.format_exc())
    
    return fig

def run_visualization(region=None, neuron_limit=50, random_seed=42):
    """
    Run the visualization pipeline for a brain region.
    
    Args:
        region (str): Name of the brain region to visualize (if None, show available regions)
        neuron_limit (int): Maximum number of neurons to fetch
        random_seed (int): Random seed for reproducibility
    
    Returns:
        None
    """
    # Set seed for reproducibility
    set_seed(random_seed)
    
    # Check if required packages are installed
    if not HAS_NEUPRINT:
        print("ERROR: neuprint-python is not properly installed or imported. Please check the logs.")
        return
    
    if not HAS_NAVIS and region is not None:
        print("WARNING: navis is not properly installed. You can list regions but not visualize neurons.")
    
    # Connect to neuPrint
    try:
        client = connect_to_neuprint()
    except Exception as e:
        print(f"Could not connect to neuPrint server: {e}")
        return
    
    # Get available regions if none specified
    if region is None:
        try:
            regions = fetch_brain_regions()
            print("Available brain regions:")
            for i, r in enumerate(sorted(regions)):
                print(f"{i+1}. {r}")
        except Exception as e:
            print(f"Could not fetch brain regions: {e}")
        return
    
    # Fetch neurons from specified region
    print(f"Fetching neurons for region: {region}")
    neurons_df = fetch_neurons_by_region(region, limit=neuron_limit)
    
    if isinstance(neurons_df, pd.DataFrame) and len(neurons_df) == 0:
        print(f"No neurons found in region {region}")
        return
    
    # Print the columns of the dataframe for debugging
    print(f"Neurons DataFrame columns: {neurons_df.columns.tolist()}")
    
    # Get the actual bodyIds from the DataFrame
    if 'bodyId' in neurons_df.columns:
        # If 'bodyId' is in columns, use those values
        real_body_ids = neurons_df['bodyId'].tolist()
    else:
        # Fall back to index if needed, but this might not work correctly
        real_body_ids = neurons_df.index.tolist()
    
    print(f"Using {len(real_body_ids)} real neuron IDs for skeleton fetching: {real_body_ids}")
    
    # Display some information about the neurons we're fetching
    if 'type' in neurons_df.columns and 'instance' in neurons_df.columns:
        for i, body_id in enumerate(real_body_ids):
            if i < len(neurons_df):
                row = neurons_df.iloc[i]
                neuron_type = row['type'] if 'type' in neurons_df.columns else 'unknown'
                neuron_name = row['instance'] if 'instance' in neurons_df.columns else str(body_id)
                print(f"Neuron {body_id}: {neuron_name} ({neuron_type})")
    
    skeletons = fetch_neuron_skeletons(real_body_ids)
    
    # Visualize neurons
    fig = visualize_neurons(neurons_df, skeletons, region)
    
    # Output a summary of the visualization
    if skeletons:
        output_html = os.path.join(parent_dir, 'results', 'figures', f'neurons_{region}.html')
        output_png = os.path.join(parent_dir, 'results', 'figures', f'neurons_{region}.png')
        print(f"\nVisualization complete! Fetched {len(neurons_df)} neurons and visualized {len(skeletons)} skeletons.")
        print(f"  HTML interactive visualization: {output_html}")
        print(f"  PNG static visualization: {output_png}")
    
    logger.info("Visualization complete")

def parse_arguments():
    """
    Parse command-line arguments for the visualization script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Visualize 3D neuron structures from the fly brain connectome.')
    
    parser.add_argument('--region', type=str, help='Brain region to visualize (e.g., FB, EB, MB)')
    parser.add_argument('--limit', type=int, default=50, help='Maximum number of neurons to fetch (default: 50)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    if args.region:
        # Run visualization with specified region
        run_visualization(region=args.region, neuron_limit=args.limit, random_seed=args.seed)
    else:
        # If run directly without args, show available regions
        run_visualization() 