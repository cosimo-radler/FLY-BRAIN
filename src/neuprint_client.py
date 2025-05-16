"""
neuPrint client module for the Drosophila connectome analysis pipeline.

This module provides a more efficient interface to the neuPrint API using
the official neuprint-python client library.
"""

import os
import json
import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from neuprint import Client, fetch_neurons, fetch_adjacencies, NeuronCriteria
from neuprint.utils import connection_table_to_matrix

import src.config as config

logger = logging.getLogger("fly_brain")

class NeuPrintInterface:
    """Interface to the neuPrint API using the official client library."""
    
    def __init__(self, server=config.NEUPRINT_SERVER, token=config.NEUPRINT_TOKEN,
                 dataset=config.NEUPRINT_DATASET):
        """
        Initialize the neuPrint client.
        
        Parameters:
            server (str): neuPrint server URL
            token (str): Authentication token
            dataset (str): Dataset identifier
        """
        # Make sure we have the https:// prefix
        if not server.startswith("http"):
            server = f"https://{server}"
        
        logger.info(f"Initializing neuPrint client for {server}, dataset: {dataset}")
        
        # Initialize client
        self.client = Client(server, token=token, dataset=dataset)
        self.dataset = dataset
        
        # Test connection
        try:
            version = self.client.fetch_version()
            logger.info(f"Connected to neuPrint server version: {version}")
        except Exception as e:
            logger.warning(f"Could not fetch server version: {e}")
            logger.warning("Continuing with limited functionality...")
    
    def fetch_neurons_by_region(self, region):
        """
        Fetch neurons from a specific brain region.
        
        Parameters:
            region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
            
        Returns:
            list: List of dictionaries containing neuron metadata
        """
        logger.info(f"Fetching neurons for region {region}")
        
        # Map our simplified region codes to actual neuPrint ROI names
        # Based on the ROI hierarchy from the documentation
        roi_mapping = {
            'EB': ['EB'],
            'FB': ['FB'],
            'MB': ['MB(L)', 'MB(R)'],  # Both left and right MB
            'LH': ['LH(R)'],           # Only right LH is available
            'AL': ['AL(L)', 'AL(R)']   # Both left and right AL
        }
        
        # Get the appropriate ROI names for the region
        if region not in roi_mapping:
            logger.error(f"Unknown region code: {region}. Available regions: {list(roi_mapping.keys())}")
            return []
        
        roi_names = roi_mapping[region]
        logger.info(f"Mapping region {region} to ROIs: {roi_names}")
        
        try:
            # Create a proper NeuronCriteria object with the mapped ROI names
            criteria = NeuronCriteria(rois=roi_names)
            
            # Fetch neurons using the criteria
            neurons_df, roi_counts_df = fetch_neurons(criteria, client=self.client)
            
            logger.info(f"Found {len(neurons_df)} neurons in region {region}")
            
            # Convert DataFrame to list of dictionaries for consistency with previous implementation
            neurons = neurons_df.to_dict('records')
            return neurons
        except Exception as e:
            logger.error(f"Error fetching neurons for {region}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return []
    
    def fetch_connectivity(self, neuron_ids, min_weight=config.CONNECTIVITY_THRESHOLD):
        """
        Fetch connectivity data between neurons.
        
        Parameters:
            neuron_ids (list): List of neuron bodyIds
            min_weight (int): Minimum synaptic weight to include
            
        Returns:
            pd.DataFrame: DataFrame with source, target, weight columns
        """
        logger.info(f"Fetching connectivity for {len(neuron_ids)} neurons")
        
        if not neuron_ids:
            logger.warning("No neuron IDs provided")
            return pd.DataFrame(columns=['source', 'target', 'weight'])
        
        try:
            # Create NeuronCriteria for the neuron IDs
            criteria = NeuronCriteria(bodyId=neuron_ids)
            
            # Use the fetch_adjacencies function to get connectivity
            neuron_df, conn_df = fetch_adjacencies(criteria, criteria, 
                                                  min_total_weight=min_weight,
                                                  client=self.client)
            
            if conn_df.empty:
                logger.warning("No connectivity data returned")
                return pd.DataFrame(columns=['source', 'target', 'weight'])
            
            # Prepare the output in the format expected by the rest of the code
            # Group by source and target to get total weights
            connectivity = conn_df.groupby(['bodyId_pre', 'bodyId_post'], as_index=False)['weight'].sum()
            
            # Rename columns to match our expected format
            connectivity_df = connectivity.rename(columns={
                'bodyId_pre': 'source',
                'bodyId_post': 'target',
                'weight': 'weight'
            })
            
            logger.info(f"Retrieved {len(connectivity_df)} connections between neurons")
            return connectivity_df
            
        except Exception as e:
            logger.error(f"Error fetching connectivity: {e}")
            return pd.DataFrame(columns=['source', 'target', 'weight'])
    
    def connectivity_to_matrix(self, conn_df, group_by='bodyId', weight_col='weight', make_square=True):
        """
        Convert connectivity DataFrame to adjacency matrix using neuprint-python utility.
        
        Parameters:
            conn_df (pd.DataFrame): DataFrame with connectivity data ('bodyId_pre', 'bodyId_post', 'weight')
            group_by (str): Base name for grouping columns (will be appended with _pre and _post)
            weight_col (str): Name of column with connection weights
            make_square (bool): If True, ensure matrix is square with same IDs in rows and columns
            
        Returns:
            pd.DataFrame: Adjacency matrix where rows are source neurons and columns are target neurons
        """
        logger.info("Converting connectivity table to adjacency matrix")
        
        if conn_df.empty:
            logger.warning("Empty connectivity DataFrame, returning empty matrix")
            return pd.DataFrame()
        
        try:
            # Check if the DataFrame has the expected column format for neuprint.utils
            # If our dataframe uses 'source' and 'target', rename to bodyId_pre and bodyId_post
            if 'source' in conn_df.columns and 'target' in conn_df.columns:
                conn_df = conn_df.rename(columns={
                    'source': 'bodyId_pre',
                    'target': 'bodyId_post'
                })
            
            # Use the neuprint utility to convert to matrix
            matrix = connection_table_to_matrix(
                conn_df=conn_df,
                group_cols=group_by,  # Will use bodyId_pre and bodyId_post
                weight_col=weight_col,
                make_square=make_square
            )
            
            logger.info(f"Created adjacency matrix with shape {matrix.shape}")
            return matrix
            
        except Exception as e:
            logger.error(f"Error converting connectivity to matrix: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def save_matrix_to_csv(self, matrix, region, output_dir=config.DATA_PROCESSED_DIR):
        """
        Save adjacency matrix to a CSV file.
        
        Parameters:
            matrix (pd.DataFrame): Adjacency matrix to save
            region (str): Brain region code
            output_dir (Path): Directory to save the file
        
        Returns:
            str: Path to the saved file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Define output file path
        output_file = os.path.join(output_dir, f"{region.lower()}_matrix.csv")
        
        # Save the matrix
        matrix.to_csv(output_file)
        
        logger.info(f"Saved adjacency matrix with shape {matrix.shape} to {output_file}")
        return output_file
    
    def fetch_network(self, neuron_ids, min_weight=config.CONNECTIVITY_THRESHOLD):
        """
        Fetch a network graph directly.
        
        Parameters:
            neuron_ids (list): List of neuron bodyIds
            min_weight (int): Minimum synaptic weight to include
            
        Returns:
            nx.DiGraph: Directed graph representation of the neural network
        """
        logger.info(f"Fetching network for {len(neuron_ids)} neurons")
        
        if not neuron_ids:
            logger.warning("No neuron IDs provided")
            return nx.DiGraph()
        
        try:
            # Get connectivity data
            connectivity_df = self.fetch_connectivity(neuron_ids, min_weight)
            
            if connectivity_df.empty:
                logger.warning("No connectivity data returned")
                return nx.DiGraph()
            
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add nodes
            for neuron_id in neuron_ids:
                G.add_node(neuron_id)
            
            # Add edges from connectivity data
            for _, row in connectivity_df.iterrows():
                G.add_edge(row['source'], row['target'], weight=row['weight'])
            
            logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except Exception as e:
            logger.error(f"Error fetching network: {e}")
            return nx.DiGraph()

    def save_neurons_to_json(self, neurons, region, output_dir=config.DATA_RAW_DIR):
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

    def save_connectivity_to_csv(self, connectivity_df, region, output_dir=config.DATA_RAW_DIR):
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

    def save_network_to_gexf(self, network, region, output_dir=config.DATA_PROCESSED_DIR):
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
        
        # Save the graph
        nx.write_gexf(network, output_file)
        
        logger.info(f"Saved network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges to {output_file}")
        return output_file
    
    def process_region(self, region, force=False):
        """
        Process a brain region: fetch neurons, connectivity, and build a network.
        
        Parameters:
            region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
            force (bool): If True, process even if files already exist
            
        Returns:
            tuple: (neurons, connectivity_df, network)
        """
        # Define expected output files
        neurons_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_neurons.json")
        connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
        network_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_network.gexf")
        matrix_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_matrix.csv")
        
        # Check if files already exist
        files_exist = all(os.path.exists(f) for f in [neurons_file, connectivity_file, network_file, matrix_file])
        
        if files_exist and not force:
            logger.info(f"Files for region {region} already exist, skipping. Use force=True to reprocess.")
            
            # Load existing data
            with open(neurons_file, 'r') as f:
                neurons = json.load(f)
            
            connectivity_df = pd.read_csv(connectivity_file)
            network = nx.read_gexf(network_file)
            
            return neurons, connectivity_df, network
        
        # Fetch neurons for the region
        neurons = self.fetch_neurons_by_region(region)
        
        if not neurons:
            logger.warning(f"No neurons found for region {region}")
            return [], pd.DataFrame(), nx.DiGraph()
        
        # Extract neuron IDs
        neuron_ids = [n.get('bodyId') for n in neurons if 'bodyId' in n]
        
        # Fetch connectivity data
        connectivity_df = self.fetch_connectivity(neuron_ids)
        
        # Build network
        network = self.fetch_network(neuron_ids)
        
        # Create and save adjacency matrix
        matrix = self.connectivity_to_matrix(connectivity_df)
        self.save_matrix_to_csv(matrix, region)
        
        # Save data
        self.save_neurons_to_json(neurons, region)
        self.save_connectivity_to_csv(connectivity_df, region)
        self.save_network_to_gexf(network, region)
        
        return neurons, connectivity_df, network 