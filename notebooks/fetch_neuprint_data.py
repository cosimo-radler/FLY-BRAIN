#!/usr/bin/env python3
"""
Script to fetch connectome data from neuPrint API and save it locally.

This script demonstrates how to use the data_io module to fetch neuron data and 
connectivity information for brain regions defined in the config module.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.append(project_root)

from src import data_io, config, utils

def main():
    """Fetch data from neuPrint and save it locally."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Fetch connectome data from neuPrint API.')
    parser.add_argument('--region', '-r', choices=list(config.BRAIN_REGIONS.keys()), 
                        help='Brain region to fetch (default: all regions)')
    parser.add_argument('--force', '-f', action='store_true', 
                        help='Force fetch even if data already exists')
    parser.add_argument('--log-level', '-l', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    args = parser.parse_args()
    
    # Set up logging
    log_file = os.path.join(project_root, 'logs', f'fetch_data_{utils.generate_timestamp()}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = utils.setup_logging(log_file, getattr(logging, args.log_level))
    
    # Set the random seed for reproducibility
    utils.set_seed(config.DEFAULT_SEED)
    
    # Determine which regions to process
    regions_to_process = [args.region] if args.region else list(config.BRAIN_REGIONS.keys())
    
    # Process each region
    for region in regions_to_process:
        region_name = config.BRAIN_REGIONS[region]
        logger.info(f"Processing {region_name} ({region})")
        
        # Define output files
        neurons_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_neurons.json")
        connectivity_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_connectivity.csv")
        network_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_network.gexf")
        
        # Check if files already exist
        if not args.force and all(os.path.exists(f) for f in [neurons_file, connectivity_file, network_file]):
            logger.info(f"Data for {region} already exists. Use --force to re-fetch.")
            continue
        
        # Fetch neurons for the region
        logger.info(f"Fetching neurons for {region}")
        neurons = data_io.fetch_neurons_by_region(region)
        if not neurons:
            logger.warning(f"No neurons found for {region}")
            continue
        
        # Save neuron data
        data_io.save_neurons_to_json(neurons, region)
        
        # Extract neuron IDs
        neuron_ids = [n['bodyId'] for n in neurons]
        logger.info(f"Found {len(neuron_ids)} neurons for {region}")
        
        # Fetch connectivity data
        logger.info(f"Fetching connectivity for {region} neurons")
        connectivity_df = data_io.fetch_connectivity(neuron_ids)
        if connectivity_df.empty:
            logger.warning(f"No connectivity found for {region} neurons")
            continue
        
        # Save connectivity data
        data_io.save_connectivity_to_csv(connectivity_df, region)
        
        # Build network from connectivity data
        logger.info(f"Building network for {region}")
        network = data_io.build_network_from_connectivity(connectivity_df)
        
        # Save network
        data_io.save_network_to_gexf(network, region)
        
        logger.info(f"Completed processing for {region}")
    
    logger.info("All regions processed successfully")

if __name__ == "__main__":
    main() 