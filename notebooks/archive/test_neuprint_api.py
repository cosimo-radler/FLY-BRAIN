#!/usr/bin/env python3
"""
Simple test script to verify neuPrint API connectivity.

This script tests basic connectivity to the neuPrint API and retrieves a small 
sample of data to confirm everything is working correctly.
"""

import os
import sys
import logging
import traceback
import requests

# Add the src directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.append(project_root)

from src import data_io, config, utils

# Set up logging to console
logger = utils.setup_logging(None, logging.INFO)

def test_api_connection():
    """Test basic connection to neuPrint server."""
    logger.info(f"Testing connection to neuPrint server: {config.NEUPRINT_SERVER}")
    
    try:
        response = requests.get(config.NEUPRINT_SERVER)
        response.raise_for_status()
        logger.info(f"Connection to neuPrint server successful. Status code: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to neuPrint server: {str(e)}")
        return False

def test_fetch_neurons(region):
    """Test fetching neurons from a specific region."""
    logger.info(f"Testing neuron fetching for region: {region}")
    
    try:
        # Fetch neurons from the region
        neurons = data_io.fetch_neurons_by_region(region)
        
        if neurons:
            logger.info(f"Successfully retrieved {len(neurons)} neurons from {region}")
            
            # Display first few neurons as a sample
            if len(neurons) > 0:
                logger.info("Sample of retrieved neurons:")
                for i, neuron in enumerate(neurons[:5]):  # Show first 5 neurons
                    logger.info(f"  Neuron {i+1}: Body ID={neuron['bodyId']}, Type={neuron.get('type', 'N/A')}")
                
                return neurons
        else:
            logger.warning(f"No neurons found in {region}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching neurons from {region}: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

def test_fetch_connectivity(neuron_ids, region_name):
    """Test fetching connectivity for specific neurons."""
    if not neuron_ids:
        logger.warning(f"No neuron IDs provided for {region_name}, skipping connectivity test")
        return False
    
    logger.info(f"Testing connectivity retrieval for {len(neuron_ids)} neurons from {region_name}")
    
    try:
        # Limit to just a few neurons for testing
        test_ids = neuron_ids[:min(2, len(neuron_ids))]
        logger.info(f"Using sample IDs: {test_ids}")
        
        connectivity = data_io.fetch_connectivity(test_ids)
        
        if not connectivity.empty:
            logger.info(f"Successfully retrieved {len(connectivity)} connections")
            logger.info("Sample of connections:")
            logger.info(connectivity.head(3).to_string())
            return True
        else:
            logger.warning(f"No connections retrieved for neurons in {region_name}")
            return False
            
    except Exception as e:
        logger.error(f"Error fetching connectivity: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

def main():
    """Test connection to neuPrint API and fetch a small sample of data."""
    logger.info("Starting neuPrint API connection tests...")
    
    # 1. Test basic server connection
    if not test_api_connection():
        logger.error("Failed to connect to neuPrint server. Aborting further tests.")
        return False
    
    # Print API configuration
    logger.info(f"Using neuPrint dataset: {config.NEUPRINT_DATASET}")
    logger.info(f"Using neuPrint token: {config.NEUPRINT_TOKEN[:10]}...")
    
    # 2. Test neuron fetching for all available regions
    success = False
    
    for region_code, region_name in config.BRAIN_REGIONS.items():
        logger.info(f"\n--- Testing {region_name} ({region_code}) ---")
        
        # Try to fetch neurons
        neurons = test_fetch_neurons(region_code)
        
        # If neurons were found, try fetching connectivity
        if neurons:
            neuron_ids = [n['bodyId'] for n in neurons]
            if test_fetch_connectivity(neuron_ids, region_name):
                success = True
                logger.info(f"Full test for {region_name} ({region_code}) completed successfully")
            else:
                logger.warning(f"Could fetch neurons but not connectivity for {region_name}")
        else:
            logger.warning(f"Could not fetch neurons for {region_name}")
    
    logger.info("\nneuPrint API connection test summary:")
    if success:
        logger.info("At least one region was successfully tested!")
    else:
        logger.error("Failed to retrieve data from any region")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 