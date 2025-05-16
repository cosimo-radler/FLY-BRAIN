#!/usr/bin/env python3
"""
Script to compare the adjacency matrix with the connectivity data.

This script verifies that the adjacency matrix created from the connectivity data
accurately represents the same information.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our config
import src.config as config

def verify_matrix_equivalence(region):
    """
    Verify that the adjacency matrix matches the connectivity data.
    
    Parameters:
        region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
    
    Returns:
        bool: True if data matches, False otherwise
    """
    print(f"Comparing matrix and connectivity data for region: {region}")
    
    # Load connectivity data
    connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    connectivity_df = pd.read_csv(connectivity_file)
    
    # Load adjacency matrix
    matrix_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_matrix.csv")
    matrix_df = pd.read_csv(matrix_file, index_col=0)
    
    print(f"Connectivity data shape: {connectivity_df.shape}")
    print(f"Matrix shape: {matrix_df.shape}")
    
    # Check total weight
    total_connectivity_weight = connectivity_df['weight'].sum()
    total_matrix_weight = matrix_df.values.sum()
    
    print(f"\nTotal weight in connectivity data: {total_connectivity_weight}")
    print(f"Total weight in matrix: {total_matrix_weight}")
    
    weights_match = abs(total_connectivity_weight - total_matrix_weight) < 0.01
    print(f"Total weights match: {weights_match}")
    
    # Sample specific connections to verify
    sample_size = min(10, len(connectivity_df))
    sampled_connections = connectivity_df.sample(sample_size)
    
    print(f"\nChecking {sample_size} random connections:")
    all_match = True
    
    for _, row in sampled_connections.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        # Find the corresponding value in the matrix
        if source in matrix_df.index and target in matrix_df.columns:
            matrix_weight = matrix_df.loc[source, target]
            match = abs(weight - matrix_weight) < 0.01
            all_match = all_match and match
            
            print(f"Connection {source} -> {target}: Connectivity weight = {weight}, Matrix weight = {matrix_weight}, Match = {match}")
        else:
            print(f"Connection {source} -> {target} not found in matrix!")
            all_match = False
    
    print(f"\nAll sampled connections match: {all_match}")
    
    # Check if any neurons in connectivity data are missing from matrix
    unique_sources = set(connectivity_df['source'].unique())
    unique_targets = set(connectivity_df['target'].unique())
    unique_neurons = unique_sources.union(unique_targets)
    
    matrix_neurons = set(matrix_df.index)
    
    missing_neurons = unique_neurons - matrix_neurons
    extra_neurons = matrix_neurons - unique_neurons
    
    if missing_neurons:
        print(f"\nWarning: {len(missing_neurons)} neurons in connectivity data not found in matrix")
        print(f"Example missing neurons: {list(missing_neurons)[:5]}")
    
    if extra_neurons:
        print(f"\nNote: {len(extra_neurons)} neurons in matrix not found in connectivity data")
        print(f"Example extra neurons: {list(extra_neurons)[:5]}")
    
    return weights_match and all_match

def main():
    """Compare matrix and connectivity data for all regions."""
    regions = list(config.BRAIN_REGIONS.keys())
    
    results = {}
    for region in regions:
        print(f"\n{'='*50}")
        print(f"Analyzing region: {region} ({config.BRAIN_REGIONS[region]})")
        print(f"{'='*50}")
        
        try:
            match = verify_matrix_equivalence(region)
            results[region] = match
        except Exception as e:
            print(f"Error analyzing region {region}: {e}")
            results[region] = False
    
    # Print summary
    print("\nSummary:")
    all_match = True
    for region, match in results.items():
        print(f"{region}: {'✅ MATCH' if match else '❌ MISMATCH'}")
        all_match = all_match and match
    
    if all_match:
        print("\n✅ All matrices match their connectivity data!")
    else:
        print("\n❌ Some matrices don't match their connectivity data!")

if __name__ == "__main__":
    main() 