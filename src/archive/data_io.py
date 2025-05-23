"""
Data I/O module for the Drosophila connectome analysis pipeline.

This module provides functions for fetching data from the neuPrint API
and handling file input/output operations. It abstracts away the details
of API communication and file storage.
"""

import os
import json
import requests
import pandas as pd
import networkx as nx
from pathlib import Path
import logging
from tqdm import tqdm

from . import config

logger = logging.getLogger("fly_brain")

def fetch_neurons_by_region(region, dataset=config.NEUPRINT_DATASET):
    """
    Fetch neurons from a specific brain region using the neuPrint API.
    
    Parameters:
        region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
        dataset (str): neuPrint dataset identifier
    
    Returns:
        list: List of dictionaries containing neuron metadata
    """
    endpoint = f"{config.NEUPRINT_SERVER}/api/custom/custom"
    
    # Construct query based on region
    if region == "EB":
        # Ellipsoid Body neurons - expanded query to catch more EB neurons
        query = """
        MATCH (n:Neuron)
        WHERE n.bodyId IS NOT NULL AND (
          n.instance CONTAINS 'EB' OR 
          n.type CONTAINS 'EB' OR
          n.type STARTS WITH 'E' OR
          n.instance CONTAINS 'ellipsoid'
        )
        RETURN n.bodyId, n.type, n.instance
        ORDER BY n.bodyId
        """
    elif region == "FB":
        # Fan-shaped Body neurons
        query = """
        MATCH (n:Neuron)
        WHERE n.bodyId IS NOT NULL AND n.instance CONTAINS 'FB'
        RETURN n.bodyId, n.type, n.instance
        ORDER BY n.bodyId
        """
    elif region == "MB":
        # Mushroom Body - Kenyon Cells
        query = """
        MATCH (n:Neuron)
        WHERE n.bodyId IS NOT NULL AND n.type STARTS WITH 'KC'
        RETURN n.bodyId, n.type, n.instance
        ORDER BY n.bodyId
        """
    elif region == "LH":
        # Lateral Horn neurons
        query = """
        MATCH (n:Neuron)
        WHERE n.bodyId IS NOT NULL AND n.instance CONTAINS 'LH'
        RETURN n.bodyId, n.type, n.instance
        ORDER BY n.bodyId
        """
    elif region == "AL":
        # Antennal Lobe neurons
        query = """
        MATCH (n:Neuron)
        WHERE n.bodyId IS NOT NULL AND n.instance CONTAINS 'AL'
        RETURN n.bodyId, n.type, n.instance
        ORDER BY n.bodyId
        """
    else:
        raise ValueError(f"Unsupported brain region: {region}")
    
    logger.info(f"Fetching neurons for region {region}")
    
    # Send request with authentication
    response = requests.post(
        endpoint,
        headers={"Authorization": f"Bearer {config.NEUPRINT_TOKEN}"},
        json={"cypher": query, "dataset": dataset}
    )
    response.raise_for_status()
    
    # Parse response
    result = response.json()
    
    # Handle the specific response format from neuPrint API
    if 'data' in result and isinstance(result['data'], list):
        data = result['data']
        columns = result.get('columns', [])
        
        # If data is a list of lists, convert to list of dicts using column names
        if data and isinstance(data[0], list) and columns:
            neurons = []
            for row in data:
                neuron_dict = {}
                for i, value in enumerate(row):
                    if i < len(columns):
                        neuron_dict[columns[i]] = value
                neurons.append(neuron_dict)
        else:
            neurons = data
    else:
        neurons = []
    
    logger.info(f"Found {len(neurons)} neurons in region {region}")
    return neurons

def fetch_connectivity(neuron_ids, dataset=config.NEUPRINT_DATASET, chunk_size=50, min_weight=config.CONNECTIVITY_THRESHOLD):
    """
    Fetch connectivity data between neurons from the neuPrint API.
    
    Parameters:
        neuron_ids (list): List of neuron bodyIds to query
        dataset (str): neuPrint dataset identifier
        chunk_size (int): Number of neurons to query at once to avoid API limits
        min_weight (int): Minimum synaptic weight to include
    
    Returns:
        pd.DataFrame: DataFrame with source, target, weight, and roiInfo columns
    """
    endpoint = f"{config.NEUPRINT_SERVER}/api/custom/custom"
    all_results = []
    
    # Process in chunks to avoid hitting API limits
    logger.info(f"Fetching connectivity for {len(neuron_ids)} neurons in chunks of {chunk_size}")
    
    for i in tqdm(range(0, len(neuron_ids), chunk_size), desc="Fetching connectivity"):
        chunk = neuron_ids[i:i+chunk_size]
        
        # Query both pre and post-synaptic connections
        query = f"""
        MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
        WHERE a.bodyId IN {str(chunk)} 
            AND b.bodyId IN {str(neuron_ids)}
            AND c.weight >= {min_weight}
        RETURN a.bodyId AS source, b.bodyId AS target, 
               c.weight AS weight, c.roiInfo AS roiInfo
        ORDER BY source, target
        """
        
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {config.NEUPRINT_TOKEN}"},
                json={"cypher": query, "dataset": dataset}
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Handle the specific response format from neuPrint API
            if 'data' in result and isinstance(result['data'], list):
                data = result['data']
                columns = result.get('columns', [])
                
                # If data is a list of lists, convert to list of dicts using column names
                if data and isinstance(data[0], list) and columns:
                    formatted_results = []
                    for row in data:
                        row_dict = {}
                        for i, value in enumerate(row):
                            if i < len(columns):
                                row_dict[columns[i]] = value
                        formatted_results.append(row_dict)
                    all_results.extend(formatted_results)
                else:
                    all_results.extend(data)
            
        except Exception as e:
            logger.error(f"Error fetching chunk {i//chunk_size + 1}: {str(e)}")
            continue
    
    # Convert to DataFrame
    if not all_results:
        logger.warning("No connectivity data returned")
        return pd.DataFrame(columns=['source', 'target', 'weight', 'roiInfo'])
    
    df = pd.DataFrame(all_results)
    logger.info(f"Retrieved {len(df)} connections between neurons")
    return df

def save_neurons_to_json(neurons, region, output_dir=config.DATA_RAW_DIR):
    """
    Save neuron metadata to a JSON file.
    
    Parameters:
        neurons (list): List of neuron dictionaries
        region (str): Brain region code
        output_dir (Path): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{region.lower()}_neurons.json")
    
    # Save the data
    with open(output_file, 'w') as f:
        json.dump(neurons, f, indent=2)
    
    logger.info(f"Saved {len(neurons)} neurons to {output_file}")
    return output_file

def save_connectivity_to_csv(connectivity_df, region, output_dir=config.DATA_RAW_DIR):
    """
    Save connectivity data to a CSV file.
    
    Parameters:
        connectivity_df (pd.DataFrame): DataFrame with connectivity data
        region (str): Brain region code
        output_dir (Path): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{region.lower()}_connectivity.csv")
    
    # Save the data
    connectivity_df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(connectivity_df)} connections to {output_file}")
    return output_file

def build_network_from_connectivity(connectivity_df):
    """
    Create a NetworkX directed graph from connectivity data.
    
    Parameters:
        connectivity_df (pd.DataFrame): DataFrame with source, target, and weight columns
    
    Returns:
        nx.DiGraph: Directed graph representation of the neural network
    """
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges from connectivity dataframe
    for _, row in connectivity_df.iterrows():
        G.add_edge(
            row['source'],
            row['target'],
            weight=row['weight'],
            roiInfo=row.get('roiInfo', {})
        )
    
    logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G

def save_network_to_gexf(network, region, output_dir=config.DATA_PROCESSED_DIR):
    """
    Save a NetworkX graph to a GEXF file for visualization in Gephi.
    
    Parameters:
        network (nx.Graph): NetworkX graph to save
        region (str): Brain region code
        output_dir (Path): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, f"{region.lower()}_network.gexf")
    
    # Save the network
    nx.write_gexf(network, output_file)
    
    logger.info(f"Saved network to {output_file}")
    return output_file

def load_network_from_gexf(region, input_dir=config.DATA_PROCESSED_DIR):
    """
    Load a NetworkX graph from a GEXF file.
    
    Parameters:
        region (str): Brain region code
        input_dir (Path): Directory containing the file
    
    Returns:
        nx.Graph: Loaded NetworkX graph
    """
    # Define input file path
    input_file = os.path.join(input_dir, f"{region.lower()}_network.gexf")
    
    # Check if file exists
    if not os.path.exists(input_file):
        logger.error(f"File not found: {input_file}")
        return None
    
    # Load the network
    network = nx.read_gexf(input_file)
    
    logger.info(f"Loaded network from {input_file} with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
    return network
