#!/usr/bin/env python3
"""
Test script for the adjacency matrix conversion functionality.

This script tests the conversion of connectivity data to adjacency matrices
using the neuprint-python utility.
"""

import os
import sys
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our modules
from src.neuprint_client import NeuPrintInterface
import src.config as config

def main():
    """Test the matrix conversion functionality."""
    # Initialize the neuPrint client
    print("Initializing neuPrint client...")
    npi = NeuPrintInterface()
    
    # Test with a specific region
    region = "EB"  # Ellipsoid Body is typically smaller and faster to process
    
    print(f"Processing region: {region} ({config.BRAIN_REGIONS[region]})")
    
    # Check if connectivity data already exists
    connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    
    if os.path.exists(connectivity_file):
        print(f"Loading existing connectivity data from {connectivity_file}")
        connectivity_df = pd.read_csv(connectivity_file)
    else:
        # If not, get neurons and connectivity
        print("Fetching neurons and connectivity...")
        neurons = npi.fetch_neurons_by_region(region)
        neuron_ids = [n.get('bodyId') for n in neurons if 'bodyId' in n]
        connectivity_df = npi.fetch_connectivity(neuron_ids)
    
    if connectivity_df.empty:
        print("No connectivity data available.")
        return
    
    print(f"Connectivity data shape: {connectivity_df.shape}")
    print("Sample connectivity data:")
    print(connectivity_df.head())
    
    # Convert to matrix
    print("\nConverting to adjacency matrix...")
    matrix = npi.connectivity_to_matrix(connectivity_df)
    
    print(f"Matrix shape: {matrix.shape}")
    print("Sample matrix (first 5x5):")
    if matrix.shape[0] > 5 and matrix.shape[1] > 5:
        print(matrix.iloc[:5, :5])
    else:
        print(matrix)
    
    # Save the matrix
    output_file = npi.save_matrix_to_csv(matrix, region)
    print(f"Matrix saved to: {output_file}")
    
    # Basic validation
    print("\nValidating matrix...")
    total_connections = connectivity_df['weight'].sum()
    total_matrix_weight = matrix.values.sum()
    
    print(f"Total weight in connectivity data: {total_connections}")
    print(f"Total weight in matrix: {total_matrix_weight}")
    
    if abs(total_connections - total_matrix_weight) < 0.01:
        print("✅ Matrix conversion looks correct (total weights match)")
    else:
        print("❌ Matrix weights don't match connectivity data!")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main() 