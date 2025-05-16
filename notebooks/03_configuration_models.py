#!/usr/bin/env python3
"""
03_configuration_models.py

This script generates configuration models for each brain region network.
It creates both unscaled and scaled versions of the configuration model.
"""

import os
import sys
import time
import logging
import argparse
import json
from datetime import datetime
import networkx as nx
import pandas as pd
import numpy as np

print("Script started")  # DEBUG

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Python path: {sys.path}")  # DEBUG

# Import project modules
print("Importing modules...")  # DEBUG
from src import config
from src import utils
from src import null_models
from src import data_io
print("Modules imported")  # DEBUG

# Setup logging
print("Setting up logging...")  # DEBUG
logger = utils.setup_logging(
    os.path.join(config.LOGS_DIR, f"configuration_models_{utils.generate_timestamp()}.log")
)
print(f"Log file: {os.path.join(config.LOGS_DIR, f'configuration_models_{utils.generate_timestamp()}.log')}")  # DEBUG

def generate_configuration_models(
    regions=None,
    seeds=None, 
    scaled_n_target=1500, 
    force=False,
    ensure_connected=True
):
    """
    Generate configuration models for specified brain regions.
    
    Parameters:
        regions (list): List of brain region codes to process
        seeds (list): List of random seeds for model generation
        scaled_n_target (int): Target number of nodes for scaled models
        force (bool): Whether to regenerate models even if they exist
        ensure_connected (bool): Whether to ensure the models are connected
        
    Returns:
        dict: Dictionary with results of model generation
    """
    start_time = datetime.now()
    
    # Use all brain regions if none specified
    if regions is None:
        regions = list(config.BRAIN_REGIONS.keys())
    
    # Use default seeds if none specified
    if seeds is None:
        seeds = [42, 123, 456, 789, 101112]
    
    logger.info(f"Generating configuration models for regions: {', '.join(regions)}")
    logger.info(f"Using seeds: {seeds}")
    logger.info(f"Scaled model target size: {scaled_n_target} nodes")
    
    results = {
        'unscaled': {},
        'scaled': {}
    }
    
    for region in regions:
        logger.info(f"Processing region: {region}")
        
        results['unscaled'][region] = []
        results['scaled'][region] = []
        
        # Load the processed network
        try:
            G_original = data_io.load_network(region, processed=True)
            logger.info(f"Loaded {region} network: {G_original.number_of_nodes()} nodes, {G_original.number_of_edges()} edges")
        except FileNotFoundError as e:
            logger.error(f"Could not load network for region {region}: {e}")
            continue
        
        # Generate unscaled configuration models with different seeds
        for seed in seeds:
            seed_start = time.time()
            
            # Check if unscaled model already exists
            existing_model = data_io.load_null_model(region, 'configuration', seed=seed)
            
            if existing_model is not None and not force:
                logger.info(f"Unscaled model for {region} with seed {seed} already exists, skipping")
                results['unscaled'][region].append({
                    'seed': seed,
                    'nodes': existing_model.number_of_nodes(),
                    'edges': existing_model.number_of_edges(),
                    'status': 'SKIPPED'
                })
                continue
            
            try:
                # Create unscaled configuration model
                utils.set_seed(seed)
                G_cm = null_models.build_plain_cm(G_original, seed=seed)
                
                # Ensure connectedness if requested
                if ensure_connected and not nx.is_connected(G_cm):
                    logger.warning(f"Unscaled model for {region} (seed {seed}) is not connected, fixing")
                    G_cm = null_models.ensure_connected(G_cm, original=G_original, seed=seed)
                
                # Save the model
                data_io.save_null_model(G_cm, region, 'configuration', seed=seed)
                
                seed_time = time.time() - seed_start
                logger.info(f"Generated unscaled model for {region} (seed {seed}) in {seed_time:.2f} seconds")
                
                results['unscaled'][region].append({
                    'seed': seed,
                    'nodes': G_cm.number_of_nodes(),
                    'edges': G_cm.number_of_edges(),
                    'time': seed_time,
                    'status': 'SUCCESS'
                })
                
            except Exception as e:
                logger.error(f"Error generating unscaled model for {region} (seed {seed}): {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                results['unscaled'][region].append({
                    'seed': seed,
                    'error': str(e),
                    'status': 'ERROR'
                })
        
        # Generate scaled configuration models
        for seed in seeds:
            seed_start = time.time()
            
            # Check if scaled model already exists
            existing_model = data_io.load_null_model(
                region, 'scaled_configuration', 
                seed=seed, n_target=scaled_n_target
            )
            
            if existing_model is not None and not force:
                logger.info(f"Scaled model for {region} with seed {seed} already exists, skipping")
                results['scaled'][region].append({
                    'seed': seed,
                    'n_target': scaled_n_target,
                    'nodes': existing_model.number_of_nodes(),
                    'edges': existing_model.number_of_edges(),
                    'status': 'SKIPPED'
                })
                continue
            
            try:
                # Create scaled configuration model
                utils.set_seed(seed)
                G_scaled = null_models.build_scaled_cm(G_original, scaled_n_target, seed=seed)
                
                # Ensure connectedness if requested
                if ensure_connected and not nx.is_connected(G_scaled):
                    logger.warning(f"Scaled model for {region} (seed {seed}) is not connected, fixing")
                    G_scaled = null_models.ensure_connected(G_scaled, original=G_original, seed=seed)
                
                # Save the model
                data_io.save_null_model(
                    G_scaled, region, 'scaled_configuration', 
                    seed=seed, n_target=scaled_n_target
                )
                
                seed_time = time.time() - seed_start
                logger.info(f"Generated scaled model for {region} (seed {seed}) in {seed_time:.2f} seconds")
                
                results['scaled'][region].append({
                    'seed': seed,
                    'n_target': scaled_n_target,
                    'nodes': G_scaled.number_of_nodes(),
                    'edges': G_scaled.number_of_edges(),
                    'time': seed_time,
                    'status': 'SUCCESS'
                })
                
            except Exception as e:
                logger.error(f"Error generating scaled model for {region} (seed {seed}): {e}")
                import traceback
                logger.debug(traceback.format_exc())
                
                results['scaled'][region].append({
                    'seed': seed,
                    'n_target': scaled_n_target,
                    'error': str(e),
                    'status': 'ERROR'
                })
    
    # Calculate total time
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    logger.info(f"Total time: {total_time:.2f} seconds")
    
    # Save results summary
    save_models_summary(results, total_time)
    
    return results

def save_models_summary(results, total_time):
    """
    Save a summary of the models generated.
    
    Parameters:
        results (dict): Results from model generation
        total_time (float): Total processing time in seconds
    """
    # Ensure tables directory exists
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create summary DataFrames
    unscaled_summary = []
    scaled_summary = []
    
    for region, models in results['unscaled'].items():
        for model in models:
            model_data = {
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Seed': model.get('seed', None),
                'Nodes': model.get('nodes', 0),
                'Edges': model.get('edges', 0),
                'Time': model.get('time', 0),
                'Status': model.get('status', 'UNKNOWN'),
                'Error': model.get('error', None)
            }
            unscaled_summary.append(model_data)
    
    for region, models in results['scaled'].items():
        for model in models:
            model_data = {
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Seed': model.get('seed', None),
                'Target_Nodes': model.get('n_target', 0),
                'Actual_Nodes': model.get('nodes', 0),
                'Edges': model.get('edges', 0),
                'Time': model.get('time', 0),
                'Status': model.get('status', 'UNKNOWN'),
                'Error': model.get('error', None)
            }
            scaled_summary.append(model_data)
    
    # Create DataFrames and save to CSV
    if unscaled_summary:
        unscaled_df = pd.DataFrame(unscaled_summary)
        unscaled_file = os.path.join(config.TABLES_DIR, f"unscaled_cm_summary_{timestamp}.csv")
        unscaled_df.to_csv(unscaled_file, index=False)
        logger.info(f"Unscaled models summary saved to {unscaled_file}")
    
    if scaled_summary:
        scaled_df = pd.DataFrame(scaled_summary)
        scaled_file = os.path.join(config.TABLES_DIR, f"scaled_cm_summary_{timestamp}.csv")
        scaled_df.to_csv(scaled_file, index=False)
        logger.info(f"Scaled models summary saved to {scaled_file}")
    
    # Create a more detailed report as JSON
    report = {
        'timestamp': timestamp,
        'total_time': total_time,
        'unscaled_models': results['unscaled'],
        'scaled_models': results['scaled']
    }
    
    # Save detailed report
    report_file = os.path.join(config.TABLES_DIR, f"configuration_models_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Detailed report saved to {report_file}")

def main():
    """Main function to run the script."""
    print("Entering main function")  # DEBUG
    parser = argparse.ArgumentParser(description='Generate configuration models for brain regions')
    parser.add_argument('--regions', nargs='+', help='Brain regions to process')
    parser.add_argument('--seeds', nargs='+', type=int, help='Random seeds for model generation')
    parser.add_argument('--n-target', type=int, default=1500, help='Target node count for scaled models')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if models exist')
    parser.add_argument('--no-connect', action='store_true', help='Skip ensuring models are connected')
    args = parser.parse_args()
    
    print(f"Args: {args}")  # DEBUG
    logger.info("Starting configuration model generation")
    logger.info(f"Arguments: {args}")
    
    try:
        print("Starting model generation...")  # DEBUG
        # Generate models
        results = generate_configuration_models(
            regions=args.regions,
            seeds=args.seeds,
            scaled_n_target=args.n_target,
            force=args.force,
            ensure_connected=not args.no_connect
        )
        
        # Print a simple summary
        print("\nConfiguration Models Summary:")
        
        print("\nUnscaled Models:")
        for region, models in results['unscaled'].items():
            success_count = sum(1 for m in models if m.get('status') == 'SUCCESS')
            skipped_count = sum(1 for m in models if m.get('status') == 'SKIPPED')
            error_count = sum(1 for m in models if m.get('status') == 'ERROR')
            
            print(f"  {region}: {success_count} generated, {skipped_count} skipped, {error_count} errors")
        
        print("\nScaled Models:")
        for region, models in results['scaled'].items():
            success_count = sum(1 for m in models if m.get('status') == 'SUCCESS')
            skipped_count = sum(1 for m in models if m.get('status') == 'SKIPPED')
            error_count = sum(1 for m in models if m.get('status') == 'ERROR')
            
            print(f"  {region}: {success_count} generated, {skipped_count} skipped, {error_count} errors")
        
        logger.info("Configuration model generation completed successfully")
        
    except Exception as e:
        print(f"ERROR: {e}")  # DEBUG
        logger.error(f"Fatal error in main: {e}")
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")  # DEBUG
        logger.error(traceback_str)
        print(f"\nERROR: Could not complete model generation due to: {str(e)}")
        print("Check logs for details.")
        sys.exit(1)

print("Checking if script is being run directly...")  # DEBUG
if __name__ == "__main__":
    print("Running main function")  # DEBUG
    main()
else:
    print("Script imported as module")  # DEBUG 