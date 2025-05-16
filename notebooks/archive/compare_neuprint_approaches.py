#!/usr/bin/env python3
"""
Script to compare the previously fetched networks with the new neuprint_client implementation.
This script analyzes both approaches in terms of number of neurons, connections, and network properties.
"""

import os
import sys
import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import logging
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our neuprint client and config
from src.neuprint_client import NeuPrintInterface
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("network_comparison")

def load_previous_data(region, data_raw_dir, data_processed_dir):
    """Load data from the previous approach."""
    logger.info(f"Loading previous data for {region}")
    
    try:
        # Load neurons
        neurons_file = os.path.join(data_raw_dir, f"{region.lower()}_neurons.json")
        with open(neurons_file, 'r') as f:
            neurons = json.load(f)
        
        # Load connectivity
        connectivity_file = os.path.join(data_raw_dir, f"{region.lower()}_connectivity.csv")
        connectivity = pd.read_csv(connectivity_file)
        
        # Load network
        network_file = os.path.join(data_processed_dir, f"{region.lower()}_network.gexf")
        network = nx.read_gexf(network_file)
        
        return {
            "neurons": neurons,
            "connectivity": connectivity,
            "network": network
        }
    except Exception as e:
        logger.error(f"Error loading previous data for {region}: {e}")
        return {
            "neurons": [],
            "connectivity": pd.DataFrame(columns=['source', 'target', 'weight']),
            "network": nx.DiGraph(),
            "error": str(e)
        }

def fetch_new_data(region, npi):
    """
    Fetch new data using the neuprint_client.
    """
    logger.info(f"Fetching new data for {region} using neuprint_client")
    
    try:
        # Process the region with the new implementation
        neurons, connectivity, network = npi.process_region(region, force=True)
        
        return {
            "neurons": neurons,
            "connectivity": connectivity,
            "network": network,
            "is_mock": False
        }
    
    except Exception as e:
        logger.error(f"Failed to fetch live data for {region}: {str(e)}")
        logger.warning("Using empty placeholder data for comparison")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Return empty placeholder data
        return {
            "neurons": [],
            "connectivity": pd.DataFrame(columns=['source', 'target', 'weight']),
            "network": nx.DiGraph(),
            "is_mock": True,
            "error": str(e)
        }

def compare_networks(previous, new, region):
    """Compare network properties between previous and new approach."""
    logger.info(f"Comparing {region} networks")
    
    # Basic comparison metrics
    comparison = {
        "Region": region,
        "Previous_Neurons": len(previous["neurons"]),
        "New_Neurons": len(new["neurons"]),
        "Previous_Connections": len(previous["connectivity"]),
        "New_Connections": len(new["connectivity"]),
        "Previous_Graph_Nodes": previous["network"].number_of_nodes(),
        "New_Graph_Nodes": new["network"].number_of_nodes(),
        "Previous_Graph_Edges": previous["network"].number_of_edges(),
        "New_Graph_Edges": new["network"].number_of_edges(),
        "Is_Mock_Data": new.get("is_mock", False)
    }
    
    if "error" in new:
        comparison["Error"] = new["error"]
    
    # Additional network metrics if networks are not empty
    if previous["network"].number_of_nodes() > 0:
        try:
            # For directed graphs, we use the undirected version for clustering coefficient
            undirected_graph = previous["network"].to_undirected()
            comparison["Previous_Density"] = nx.density(previous["network"])
            comparison["Previous_Avg_Clustering"] = nx.average_clustering(undirected_graph)
            comparison["Previous_Avg_Degree"] = sum(dict(previous["network"].degree()).values()) / previous["network"].number_of_nodes()
        except Exception as e:
            logger.warning(f"Couldn't compute some metrics for previous {region} network: {e}")
    
    if new["network"].number_of_nodes() > 0 and not new.get("is_mock", False):
        try:
            # For directed graphs, we use the undirected version for clustering coefficient
            undirected_graph = new["network"].to_undirected()
            comparison["New_Density"] = nx.density(new["network"])
            comparison["New_Avg_Clustering"] = nx.average_clustering(undirected_graph)
            comparison["New_Avg_Degree"] = sum(dict(new["network"].degree()).values()) / new["network"].number_of_nodes()
        except Exception as e:
            logger.warning(f"Couldn't compute some metrics for new {region} network: {e}")
    
    # Common neurons - only if we have real data
    if not new.get("is_mock", False) and len(new["neurons"]) > 0:
        try:
            previous_ids = {n.get('bodyId') for n in previous["neurons"] if 'bodyId' in n}
            new_ids = {n.get('bodyId') for n in new["neurons"] if 'bodyId' in n}
            common_ids = previous_ids.intersection(new_ids)
            
            comparison["Common_Neurons"] = len(common_ids)
            comparison["Only_Previous"] = len(previous_ids - new_ids)
            comparison["Only_New"] = len(new_ids - previous_ids)
        except Exception as e:
            logger.warning(f"Couldn't compare neuron sets: {e}")
    
    return comparison

def create_comparison_plots(df, regions, output_dir):
    """Create visualizations comparing the two approaches."""
    # Check if we have mock data
    if 'Is_Mock_Data' in df.columns and df['Is_Mock_Data'].any():
        logger.warning("Some data is mocked - visualizations may not be meaningful")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot neuron counts
    plt.figure(figsize=(12, 6))
    x = range(len(df))
    width = 0.35
    
    # Extract data for plotting, handling string values
    prev_neurons = []
    new_neurons = []
    for _, row in df.iterrows():
        prev_neurons.append(row["Previous_Neurons"] if isinstance(row["Previous_Neurons"], (int, float)) else 0)
        new_neurons.append(row["New_Neurons"] if isinstance(row["New_Neurons"], (int, float)) else 0)
    
    plt.bar(x, prev_neurons, width, label='Previous Approach')
    plt.bar([i + width for i in x], new_neurons, width, label='New Approach')
    
    plt.xlabel('Brain Region')
    plt.ylabel('Number of Neurons')
    plt.title('Comparison of Neuron Counts')
    plt.xticks([i + width/2 for i in x], df['Region'])
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, "neuron_count_comparison.png"), dpi=300, bbox_inches='tight')
    
    # Plot connection counts if available
    if "Previous_Connections" in df.columns and "New_Connections" in df.columns:
        plt.figure(figsize=(12, 6))
        
        prev_connections = []
        new_connections = []
        for _, row in df.iterrows():
            prev_connections.append(row["Previous_Connections"] if isinstance(row["Previous_Connections"], (int, float)) else 0)
            new_connections.append(row["New_Connections"] if isinstance(row["New_Connections"], (int, float)) else 0)
        
        plt.bar(x, prev_connections, width, label='Previous Approach')
        plt.bar([i + width for i in x], new_connections, width, label='New Approach')
        
        plt.xlabel('Brain Region')
        plt.ylabel('Number of Connections')
        plt.title('Comparison of Connection Counts')
        plt.xticks([i + width/2 for i in x], df['Region'])
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, "connection_count_comparison.png"), dpi=300, bbox_inches='tight')
    
    logger.info("Comparison plots created")

def main():
    # Regions to compare
    regions = ["EB", "FB", "MB", "LH", "AL"]
    
    # Get paths from config
    data_raw_dir = config.DATA_RAW_DIR
    data_processed_dir = config.DATA_PROCESSED_DIR
    figures_dir = config.FIGURES_DIR
    tables_dir = config.TABLES_DIR
    
    try:
        # Initialize neuPrint client
        npi = NeuPrintInterface()
        
        # Store comparison results
        comparisons = []
        
        for region in regions:
            try:
                # Load previous data
                previous_data = load_previous_data(region, data_raw_dir, data_processed_dir)
                logger.info(f"Successfully loaded previous data for {region}")
                
                # Fetch new data
                new_data = fetch_new_data(region, npi)
                if new_data.get("is_mock", False):
                    logger.warning(f"Using mock data for {region}")
                
                # Compare networks
                comparison = compare_networks(previous_data, new_data, region)
                comparisons.append(comparison)
                
                logger.info(f"Comparison for {region} complete")
                
            except Exception as e:
                logger.error(f"Error comparing {region}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                # Add error record
                comparisons.append({
                    "Region": region,
                    "Previous_Neurons": "ERROR",
                    "New_Neurons": "ERROR",
                    "Error": str(e)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparisons)
        
        # Save results to CSV
        os.makedirs(tables_dir, exist_ok=True)
        output_file = os.path.join(tables_dir, "neuprint_comparison.csv")
        comparison_df.to_csv(output_file, index=False)
        
        logger.info(f"Comparison results saved to {output_file}")
        
        # Print summary
        print("\nNeuPrint Comparison Summary:")
        print(comparison_df.to_string(index=False))
        
        # Create a visualization
        create_comparison_plots(comparison_df, regions, figures_dir)
            
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print("\nERROR: Could not complete comparison due to:", str(e))
        print("Check your API token and network connection.")

if __name__ == "__main__":
    main() 