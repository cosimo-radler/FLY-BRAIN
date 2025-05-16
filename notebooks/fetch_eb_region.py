#!/usr/bin/env python3
"""
Script to fetch updated data for the Ellipsoid Body (EB) region.

This script uses the improved query to fetch more neurons in the EB region.
"""

import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src import data_io, config, utils

def main():
    """Fetch data for the EB region."""
    # Set up logging
    log_file = os.path.join(project_root, 'logs', f'fetch_eb_{utils.generate_timestamp()}.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = utils.setup_logging(log_file, logging.INFO)
    
    # Set random seed for reproducibility
    utils.set_seed(config.DEFAULT_SEED)
    
    region = "EB"
    region_name = config.BRAIN_REGIONS[region]
    
    logger.info(f"Re-fetching data for {region_name} ({region}) with improved query")
    
    # Step 1: Fetch neurons for the region
    try:
        logger.info(f"Fetching neurons for {region}")
        neurons = data_io.fetch_neurons_by_region(region)
        
        if not neurons:
            logger.warning(f"No neurons found for {region}")
            return False
        
        logger.info(f"Found {len(neurons)} neurons in {region_name}")
        
        # Save neuron data
        output_file = data_io.save_neurons_to_json(neurons, region)
        logger.info(f"Saved neurons to {output_file}")
        
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
        output_file = data_io.save_connectivity_to_csv(connectivity_df, region)
        logger.info(f"Saved connectivity to {output_file}")
        
        # Step 3: Build and save network
        logger.info(f"Building network for {region}")
        network = data_io.build_network_from_connectivity(connectivity_df)
        
        # Save network
        output_file = data_io.save_network_to_gexf(network, region)
        logger.info(f"Saved network to {output_file}")
        
        logger.info(f"Successfully processed {region}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {region}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 