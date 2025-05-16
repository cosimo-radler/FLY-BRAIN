#!/usr/bin/env python3
"""
Script to correctly verify matrix connections against connectivity data.

This script checks if the connections in the connectivity data are properly
represented in the adjacency matrix, accounting for how the matrix labels are stored.
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

def check_matrix_connections(region):
    """
    Correctly check the connections in the matrix against connectivity data.
    
    Parameters:
        region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
    """
    print(f"Checking matrix connections for region: {region}")
    
    # Load connectivity data
    connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    connectivity_df = pd.read_csv(connectivity_file)
    
    # Load adjacency matrix
    matrix_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_matrix.csv")
    
    # Read matrix with the first column as index
    matrix_df = pd.read_csv(matrix_file, index_col=0)
    
    print(f"Connectivity data shape: {connectivity_df.shape}")
    print(f"Matrix shape: {matrix_df.shape}")
    
    # Show the first few rows/columns to understand the structure
    print("\nMatrix index name:", matrix_df.index.name)
    print("Matrix column names (first 5):", list(matrix_df.columns[:5]))
    
    # Check total weight sum to ensure they match
    total_connectivity_weight = connectivity_df['weight'].sum()
    total_matrix_weight = matrix_df.values.sum()
    
    print(f"\nTotal weight in connectivity data: {total_connectivity_weight}")
    print(f"Total weight in matrix: {total_matrix_weight}")
    
    weights_match = abs(total_connectivity_weight - total_matrix_weight) < 0.01
    print(f"Total weights match: {weights_match}")
    
    # Sample connections to verify
    sample_size = min(10, len(connectivity_df))
    sampled_connections = connectivity_df.sample(sample_size)
    
    print(f"\nChecking {sample_size} random connections (correctly handling matrix labels):")
    matching_found = 0
    
    for _, row in sampled_connections.iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        # Find the corresponding value in the matrix
        # The matrix columns are strings, so convert target to string
        target_str = str(target)
        
        if target_str in matrix_df.columns:
            try:
                # Use loc to access by label
                matrix_weight = matrix_df.loc[source, target_str]
                match = abs(weight - matrix_weight) < 0.01
                matching_found += 1
                print(f"Connection {source} -> {target}: Connectivity weight = {weight}, Matrix weight = {matrix_weight}, Match = {match}")
            except KeyError:
                print(f"Source {source} not found in matrix index")
        else:
            print(f"Target {target} not found in matrix columns")
    
    print(f"\nFound and matched {matching_found}/{sample_size} connections")
    
    return weights_match

def main():
    """Check matrix connections for all regions."""
    regions = list(config.BRAIN_REGIONS.keys())
    
    results = {}
    for region in regions:
        print(f"\n{'='*50}")
        print(f"Analyzing region: {region} ({config.BRAIN_REGIONS[region]})")
        print(f"{'='*50}")
        
        try:
            match = check_matrix_connections(region)
            results[region] = match
        except Exception as e:
            print(f"Error analyzing region {region}: {e}")
            import traceback
            print(traceback.format_exc())
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