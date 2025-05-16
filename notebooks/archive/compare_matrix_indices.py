#!/usr/bin/env python3
"""
Script to investigate how matrix indices relate to connectivity data.

This script examines the indices of the adjacency matrix and compares them to
the neuron IDs in the connectivity data to understand the relationship.
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our config
import src.config as config

def analyze_matrix_indices(region):
    """
    Analyze how matrix indices relate to connectivity data.
    
    Parameters:
        region (str): Brain region code (e.g., 'EB', 'FB', 'MB')
    """
    print(f"Analyzing matrix indices for region: {region}")
    
    # Load connectivity data
    connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    connectivity_df = pd.read_csv(connectivity_file)
    
    # Load adjacency matrix
    matrix_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_matrix.csv")
    matrix_df = pd.read_csv(matrix_file, index_col=0)
    
    print(f"Connectivity data shape: {connectivity_df.shape}")
    print(f"Matrix shape: {matrix_df.shape}")
    
    # Get unique neuron IDs from connectivity data
    unique_sources = set(connectivity_df['source'].unique())
    unique_targets = set(connectivity_df['target'].unique())
    unique_neurons = unique_sources.union(unique_targets)
    
    # Get matrix indices
    matrix_indices = set(matrix_df.index)
    
    # Compare counts
    print(f"\nUnique neurons in connectivity data: {len(unique_neurons)}")
    print(f"Indices in matrix: {len(matrix_indices)}")
    
    # Show sample data
    print("\nSample connectivity data:")
    print(connectivity_df.head())
    
    print("\nMatrix index and column names type:")
    print(f"Index type: {type(matrix_df.index[0])}")
    print(f"Column type: {type(matrix_df.columns[0])}")
    
    # Check index type conversion
    if matrix_df.index.dtype != 'object':
        print("\nMatrix indices are not strings, connectivity IDs may need conversion")
    
    # Check if connectivity IDs exist in matrix after conversion
    converted_found = 0
    total_checked = min(20, len(connectivity_df))
    
    print(f"\nChecking {total_checked} random connections with potential type conversion:")
    for i, row in connectivity_df.sample(total_checked).iterrows():
        source = row['source']
        target = row['target']
        weight = row['weight']
        
        # Try to convert to the same type as matrix indices
        source_converted = str(source)
        target_converted = str(target)
        
        if source_converted in matrix_df.index and target_converted in matrix_df.columns:
            matrix_weight = matrix_df.loc[source_converted, target_converted]
            match = abs(weight - matrix_weight) < 0.01
            converted_found += 1
            print(f"Found after conversion: {source} -> {target}, Weight: {weight}, Matrix: {matrix_weight}, Match: {match}")
        else:
            print(f"Not found even after conversion: {source} -> {target}")
    
    print(f"\nFound {converted_found}/{total_checked} connections after conversion")
    
    # Check for specific indices
    print("\nFirst 5 matrix indices:")
    for idx in list(matrix_indices)[:5]:
        print(f"  {idx} (type: {type(idx)})")
    
    print("\nFirst 5 connectivity sources:")
    for src in list(unique_sources)[:5]:
        print(f"  {src} (type: {type(src)})")

def main():
    """Analyze matrix indices for all regions."""
    regions = list(config.BRAIN_REGIONS.keys())
    
    for region in regions:
        print(f"\n{'='*50}")
        print(f"Analyzing region: {region} ({config.BRAIN_REGIONS[region]})")
        print(f"{'='*50}")
        
        try:
            analyze_matrix_indices(region)
        except Exception as e:
            print(f"Error analyzing region {region}: {e}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    main() 