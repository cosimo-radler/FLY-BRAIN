#!/usr/bin/env python3
"""
Script to fetch data from neuPrint API for all configured brain regions.

This script will fetch neuron data and connectivity for each brain region
defined in the config file, and save it to the appropriate directories.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src import data_io, config, utils

def process_region(region, force=False):
    """
    Process a single brain region.
    
    Parameters:
        region (str): Brain region code (e.g., 'EB', 'FB')
        force (bool): Whether to force re-fetch even if files exist
        
    Returns:
        bool: Whether processing was successful
    """
    region_name = config.BRAIN_REGIONS[region]
    logger.info(f"Processing {region_name} ({region})")
    
    # Define output files
    neurons_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_neurons.json")
    connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
    network_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_network.gexf")
    
    # Check if files already exist
    if not force and all(os.path.exists(f) for f in [neurons_file, connectivity_file, network_file]):
        logger.info(f"Data for {region} already exists. Use --force to re-fetch.")
        return True
    
    # Step 1: Fetch neurons for the region
    try:
        logger.info(f"Fetching neurons for {region}")
        neurons = data_io.fetch_neurons_by_region(region)
        
        if not neurons:
            logger.warning(f"No neurons found for {region}")
            return False
        
        # Save neuron data
        data_io.save_neurons_to_json(neurons, region)
        
        # Extract neuron IDs
        neuron_ids = []
        for neuron in neurons:
            # Handle different data formats
            if isinstance(neuron, dict) and 'bodyId' in neuron:
                neuron_ids.append(neuron['bodyId'])
            elif isinstance(neuron, dict) and 'n.bodyId' in neuron:
                neuron_ids.append(neuron['n.bodyId'])
            
        logger.info(f"Found {len(neuron_ids)} neurons with IDs for {region}")
        
        if not neuron_ids:
            logger.warning(f"No valid neuron IDs found for {region}")
            return False
            
        # Step 2: Fetch connectivity data
        logger.info(f"Fetching connectivity for {region} neurons")
        connectivity_df = data_io.fetch_connectivity(neuron_ids)
        
        if connectivity_df.empty:
            logger.warning(f"No connectivity found for {region} neurons")
            return False
        
        # Save connectivity data
        data_io.save_connectivity_to_csv(connectivity_df, region)
        
        # Step 3: Build and save network
        logger.info(f"Building network for {region}")
        network = data_io.build_network_from_connectivity(connectivity_df)
        
        # Save network
        data_io.save_network_to_gexf(network, region)
        
        logger.info(f"Successfully processed {region}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {region}: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main():
    """Main function to process all brain regions."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch data for all brain regions from neuPrint API.')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Force re-fetch even if data already exists')
    parser.add_argument('--log-level', '-l', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    global logger
    log_file = os.path.join(project_root, 'logs', f'fetch_regions_{utils.generate_timestamp()}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = utils.setup_logging(log_file, getattr(logging, args.log_level))
    
    # Set random seed for reproducibility
    utils.set_seed(config.DEFAULT_SEED)
    
    logger.info(f"Starting data fetch for {len(config.BRAIN_REGIONS)} brain regions")
    logger.info(f"Using neuPrint server: {config.NEUPRINT_SERVER}")
    logger.info(f"Using dataset: {config.NEUPRINT_DATASET}")
    
    # Process each region
    success_count = 0
    
    for region in config.BRAIN_REGIONS:
        if process_region(region, args.force):
            success_count += 1
    
    # Report results
    logger.info(f"Completed processing {success_count}/{len(config.BRAIN_REGIONS)} regions successfully")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 