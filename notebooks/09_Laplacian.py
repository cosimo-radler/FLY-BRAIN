#!/usr/bin/env python3
"""
09_Laplacian.py

Script to compute Laplacian matrices for the original brain networks.
This script:
1. Loads all original brain region networks from data/processed/
2. Computes both standard and normalized Laplacian matrices
3. Computes Laplacian eigenvalues and related metrics
4. Saves results to results/tables/ and matrices to results/

Following the project structure:
- Reads from data/processed/ (original cleaned networks)
- Saves results to results/tables/ and results/figures/
- Uses functions from src/ modules for all computations
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import our modules
import data_io
import metrics
import utils

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("laplacian_analysis")

def compute_and_save_laplacian_matrices():
    """
    Main function to compute Laplacian matrices for all brain regions and model types.
    """
    logger.info("Starting Laplacian matrix computation for all brain regions and model types")
    
    # Set random seed for reproducibility
    utils.set_seed(42)
    
    # Get available brain regions
    brain_regions = data_io.get_brain_regions()
    logger.info(f"Found brain regions: {brain_regions}")
    
    # Define model types to process
    model_types = ["original", "configuration_model", "coarsened"]
    
    # Initialize results storage
    all_results = []
    laplacian_matrices = {}
    
    # Process each brain region
    for region in brain_regions:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing brain region: {region}")
        logger.info(f"{'='*50}")
        
        # Get available models for this region
        available_models = data_io.get_available_models(region)
        logger.info(f"Available models for {region}: {available_models}")
        
        region_matrices = {}
        
        # Process each model type
        for model_type in model_types:
            if model_type not in available_models:
                logger.warning(f"Model type '{model_type}' not available for region {region}")
                continue
                
            logger.info(f"\n--- Processing {model_type} for {region} ---")
            
            try:
                # Load the network
                if model_type == "configuration_model":
                    # For CM, load the first available instance or generate one if needed
                    null_models = data_io.load_null_models(region, "configuration_model", max_models=1)
                    if null_models:
                        G = null_models[0]
                        logger.info(f"Loaded existing configuration model")
                    else:
                        # Load original and create CM if none exists
                        G_orig = data_io.load_graph(region, "original")
                        if G_orig is None:
                            logger.warning(f"Could not load original graph for {region}")
                            continue
                        
                        # Import null_models module and create CM
                        import null_models as nm
                        logger.info("Generating configuration model...")
                        G = nm.build_plain_cm(G_orig.to_undirected() if G_orig.is_directed() else G_orig)
                else:
                    # Load original or coarsened model
                    G = data_io.load_graph(region, model_type=model_type)
                
                if G is None:
                    logger.warning(f"Could not load {model_type} graph for region {region}")
                    continue
                
                logger.info(f"Loaded {region} {model_type}: {len(G)} nodes, {G.number_of_edges()} edges")
                
                # Convert to undirected if needed
                if G.is_directed():
                    logger.info("Converting directed graph to undirected")
                    G = G.to_undirected()
                
                # Check connectivity
                is_connected = nx.is_connected(G)
                logger.info(f"Network is connected: {is_connected}")
                
                if not is_connected:
                    # Get largest connected component
                    largest_cc = max(nx.connected_components(G), key=len)
                    G_lcc = G.subgraph(largest_cc).copy()
                    logger.info(f"Using largest connected component: {len(G_lcc)} nodes")
                    G_for_laplacian = G_lcc
                else:
                    G_for_laplacian = G
                
                # Compute basic graph metrics first
                basic_metrics = metrics.compute_all_metrics(G)
                
                # Compute Laplacian matrices
                logger.info("Computing standard Laplacian matrix...")
                L_standard = metrics.compute_laplacian_matrix(G_for_laplacian, normalized=False)
                
                logger.info("Computing normalized Laplacian matrix...")
                L_normalized = metrics.compute_laplacian_matrix(G_for_laplacian, normalized=True)
                
                # Compute eigenvalues and related metrics
                logger.info("Computing Laplacian eigenvalues and metrics...")
                laplacian_metrics = metrics.compute_laplacian_metrics(G_for_laplacian)
                
                # Store matrices for saving
                model_key = f"{region}_{model_type}"
                region_matrices[model_type] = {
                    'standard': L_standard,
                    'normalized': L_normalized,
                    'eigenvals_standard': laplacian_metrics.get('laplacian_eigenvals', 
                                                              laplacian_metrics.get('lcc_laplacian_eigenvals')),
                    'eigenvals_normalized': laplacian_metrics.get('normalized_laplacian_eigenvals',
                                                                laplacian_metrics.get('lcc_normalized_laplacian_eigenvals'))
                }
                
                # Combine all metrics
                result = {
                    'brain_region': region,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    **basic_metrics,
                    **laplacian_metrics
                }
                
                all_results.append(result)
                
                logger.info(f"Successfully computed Laplacian metrics for {region} {model_type}")
                algebraic_conn = result.get('algebraic_connectivity', result.get('lcc_algebraic_connectivity', 'N/A'))
                spectral_gap = result.get('spectral_gap', result.get('lcc_spectral_gap', 'N/A'))
                logger.info(f"Algebraic connectivity: {algebraic_conn}")
                logger.info(f"Spectral gap: {spectral_gap}")
                
            except Exception as e:
                logger.error(f"Error processing {model_type} for region {region}: {str(e)}")
                continue
        
        # Store all matrices for this region
        if region_matrices:
            laplacian_matrices[region] = region_matrices
    
    return all_results, laplacian_matrices

def save_results(all_results, laplacian_matrices):
    """
    Save the computed results and matrices to files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directories if they don't exist
    os.makedirs(data_io.RESULTS_TABLES, exist_ok=True)
    os.makedirs(data_io.RESULTS_FIGURES, exist_ok=True)
    
    # Save metrics as CSV
    df_results = pd.DataFrame(all_results)
    csv_path = data_io.RESULTS_TABLES / f"laplacian_metrics_all_models_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved metrics to {csv_path}")
    
    # Save Laplacian matrices as NPZ files
    matrices_dir = data_io.RESULTS_DIR / "laplacian_matrices"
    os.makedirs(matrices_dir, exist_ok=True)
    
    for region, matrices in laplacian_matrices.items():
        for model_type, matrix_data in matrices.items():
            npz_path = matrices_dir / f"{region.lower()}_{model_type}_laplacian_matrices_{timestamp}.npz"
            np.savez_compressed(
                npz_path,
                standard_laplacian=matrix_data['standard'],
                normalized_laplacian=matrix_data['normalized'],
                eigenvals_standard=matrix_data['eigenvals_standard'],
                eigenvals_normalized=matrix_data['eigenvals_normalized']
            )
            logger.info(f"Saved Laplacian matrices for {region} {model_type} to {npz_path}")
    
    # Save summary statistics - one row per region-model combination
    summary_cols = ['num_nodes', 'num_edges', 'algebraic_connectivity', 'spectral_gap', 
                   'laplacian_largest', 'normalized_laplacian_largest']
    
    # Handle LCC prefixed columns if they exist
    lcc_cols = ['lcc_algebraic_connectivity', 'lcc_spectral_gap', 'lcc_laplacian_largest', 'lcc_normalized_laplacian_largest']
    
    # Create a clean summary dataframe
    summary_data = []
    for _, row in df_results.iterrows():
        summary_row = {
            'brain_region': row['brain_region'],
            'model_type': row['model_type'],
            'num_nodes': row['num_nodes'],
            'num_edges': row['num_edges']
        }
        
        # Handle regular vs LCC columns
        for col in ['algebraic_connectivity', 'spectral_gap', 'laplacian_largest', 'normalized_laplacian_largest']:
            if col in row and pd.notna(row[col]):
                summary_row[col] = row[col]
            elif f'lcc_{col}' in row and pd.notna(row[f'lcc_{col}']):
                summary_row[col] = row[f'lcc_{col}']
            else:
                summary_row[col] = None
        
        summary_data.append(summary_row)
    
    summary_df = pd.DataFrame(summary_data)
    
    summary_path = data_io.RESULTS_TABLES / f"laplacian_summary_all_models_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary statistics to {summary_path}")
    
    return csv_path, summary_path

def main():
    """
    Main execution function.
    """
    logger.info("="*60)
    logger.info("LAPLACIAN MATRIX ANALYSIS FOR BRAIN NETWORKS")
    logger.info("="*60)
    
    try:
        # Compute Laplacian matrices and metrics
        all_results, laplacian_matrices = compute_and_save_laplacian_matrices()
        
        if not all_results:
            logger.error("No results computed. Check if brain region data is available.")
            return
        
        # Save results
        csv_path, summary_path = save_results(all_results, laplacian_matrices)
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to:")
        logger.info(f"  - Detailed metrics: {csv_path}")
        logger.info(f"  - Summary statistics: {summary_path}")
        logger.info(f"  - Laplacian matrices: {data_io.RESULTS_DIR / 'laplacian_matrices'}")
        
        # Display summary
        df_summary = pd.read_csv(summary_path)
        print("\nSUMMARY OF RESULTS:")
        print(df_summary.to_string(index=False))
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 