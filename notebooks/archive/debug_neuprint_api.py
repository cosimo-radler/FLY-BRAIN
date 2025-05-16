#!/usr/bin/env python3
"""
Debug script for neuPrint API issues.

This script performs direct API queries to troubleshoot connectivity issues.
"""

import os
import sys
import json
import requests
import logging

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src import config, utils

# Set up logging
logger = utils.setup_logging(None, logging.INFO)

def test_server_version(token):
    """Test basic connection and get server version."""
    logger.info("Testing server version endpoint...")
    
    url = f"{config.NEUPRINT_SERVER}/api/version"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        version_info = response.json()
        logger.info(f"Server version: {json.dumps(version_info, indent=2)}")
        return True
    except Exception as e:
        logger.error(f"Error getting server version: {str(e)}")
        if 'response' in locals():
            try:
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response status code: {response.status_code}")
                logger.error(f"Response text: {response.text}")
        return False

def test_available_datasets(token):
    """Test listing available datasets."""
    logger.info("Testing available datasets endpoint...")
    
    url = f"{config.NEUPRINT_SERVER}/api/datasets"
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        datasets = response.json()
        logger.info(f"Available datasets: {json.dumps(datasets, indent=2)}")
        return datasets
    except Exception as e:
        logger.error(f"Error getting available datasets: {str(e)}")
        if 'response' in locals():
            try:
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response status code: {response.status_code}")
                logger.error(f"Response text: {response.text}")
        return []

def test_direct_cypher_query(dataset, query, token):
    """Test running a Cypher query directly."""
    logger.info(f"Testing direct Cypher query on dataset '{dataset}'...")
    logger.info(f"Query: {query}")
    
    url = f"{config.NEUPRINT_SERVER}/api/custom/custom"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"cypher": query, "dataset": dataset}
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if 'data' in result:
            count = len(result['data'])
            logger.info(f"Query succeeded! Returned {count} results.")
            if count > 0:
                logger.info(f"First few results: {json.dumps(result['data'][:3], indent=2)}")
            return result['data']
        else:
            logger.info(f"Query succeeded but returned no data: {json.dumps(result, indent=2)}")
            return []
    except Exception as e:
        logger.error(f"Error executing Cypher query: {str(e)}")
        if 'response' in locals():
            try:
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response status code: {response.status_code}")
                logger.error(f"Response text: {response.text}")
        return None

def main():
    """Main function to test API functionality."""
    logger.info("Starting detailed neuPrint API diagnostics...")
    logger.info(f"Using neuPrint server: {config.NEUPRINT_SERVER}")
    logger.info(f"Using token: {config.NEUPRINT_TOKEN[:15]}...")
    
    # Test server connection and version
    if not test_server_version(config.NEUPRINT_TOKEN):
        logger.error("Failed to connect to server with authentication. Aborting further tests.")
        return False
    
    # Test available datasets
    datasets = test_available_datasets(config.NEUPRINT_TOKEN)
    if not datasets:
        logger.error("Failed to retrieve datasets. Aborting further tests.")
        return False
    
    # Check if our configured dataset is available
    dataset_names = [d.get('name') for d in datasets if isinstance(d, dict) and 'name' in d]
    if not dataset_names:
        # Try extracting dataset names differently if the API returns a different format
        logger.info("Could not extract dataset names from response, trying alternative formats...")
        if isinstance(datasets, list):
            dataset_names = datasets
        
    if config.NEUPRINT_DATASET not in dataset_names:
        logger.warning(f"Configured dataset '{config.NEUPRINT_DATASET}' not found in available datasets!")
        logger.info(f"Available datasets: {', '.join(dataset_names)}")
        
        # Use the first available dataset for testing
        if dataset_names:
            test_dataset = dataset_names[0]
            logger.info(f"Will use '{test_dataset}' for testing instead")
        else:
            logger.error("No usable datasets found. Aborting further tests.")
            return False
    else:
        test_dataset = config.NEUPRINT_DATASET
        logger.info(f"Configured dataset '{test_dataset}' is available")
    
    # Test simple query to list some neurons
    query1 = """
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL
    RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance
    ORDER BY bodyId
    LIMIT 5
    """
    
    logger.info("\n--- Testing basic neuron query ---")
    results1 = test_direct_cypher_query(test_dataset, query1, config.NEUPRINT_TOKEN)
    
    if results1 is None:
        logger.error("Failed to execute basic neuron query")
        
        # Try a simpler query without using custom endpoint
        logger.info("\n--- Testing direct neuron query to neuron endpoint ---")
        try:
            url = f"{config.NEUPRINT_SERVER}/api/npexplorer/neurons"
            headers = {"Authorization": f"Bearer {config.NEUPRINT_TOKEN}", "Content-Type": "application/json"}
            payload = {"dataset": test_dataset, "input_ROIs": [], "output_ROIs": [], "all_segments": True, "statuses": ["Traced", "Untraced"], "limit": 5}
            
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Direct neuron endpoint response: {json.dumps(result, indent=2)}")
        except Exception as e:
            logger.error(f"Error with direct neuron endpoint: {str(e)}")
            if 'response' in locals():
                try:
                    error_detail = response.json()
                    logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
                except:
                    logger.error(f"Response status code: {response.status_code}")
                    logger.error(f"Response text: {response.text}")
        
        return False
    elif not results1:
        logger.warning("Basic neuron query succeeded but returned no results")
    
    # Test query specifically for Mushroom Body
    query2 = """
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL AND n.type CONTAINS "KC"
    RETURN n.bodyId AS bodyId, n.type AS type, n.instance AS instance
    ORDER BY bodyId
    LIMIT 5
    """
    
    logger.info("\n--- Testing Mushroom Body (KC) neuron query ---")
    results2 = test_direct_cypher_query(test_dataset, query2, config.NEUPRINT_TOKEN)
    
    # If both queries failed, try without filters to see if we can get any neurons
    if (not results1 or not results1) and (not results2 or not results2):
        logger.info("\n--- Testing minimal neuron query ---")
        query3 = """
        MATCH (n:Neuron)
        RETURN n.bodyId AS bodyId
        LIMIT 5
        """
        results3 = test_direct_cypher_query(test_dataset, query3, config.NEUPRINT_TOKEN)
        
        if results3 is None or not results3:
            logger.error("Failed to retrieve any neurons with minimal constraints")
            return False
    
    logger.info("\nDiagnostics completed. Check the logs for details.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 