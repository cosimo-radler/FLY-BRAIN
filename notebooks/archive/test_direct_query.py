#!/usr/bin/env python3
"""
Direct query test for neuPrint API.

This script tests direct Cypher queries to the neuPrint API using the configured dataset.
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

def run_cypher_query(query, dataset=config.NEUPRINT_DATASET, token=config.NEUPRINT_TOKEN):
    """
    Run a Cypher query against the neuPrint API.
    
    Parameters:
        query (str): Cypher query to run
        dataset (str): Dataset identifier
        token (str): Authentication token
    
    Returns:
        list or dict or None: Query results or None if error
    """
    url = f"{config.NEUPRINT_SERVER}/api/custom/custom"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    payload = {"cypher": query, "dataset": dataset}
    
    logger.info(f"Running query on dataset '{dataset}':")
    logger.info(query)
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Query succeeded, raw response received")
        
        # Print complete raw response for debugging
        logger.debug(f"Raw response: {json.dumps(result, indent=2)}")
        
        return result
            
    except Exception as e:
        logger.error(f"Error executing query: {str(e)}")
        if 'response' in locals():
            try:
                logger.error(f"Response status code: {response.status_code}")
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response text: {response.text}")
        return None

def run_explorer_query(endpoint, payload, token=config.NEUPRINT_TOKEN):
    """
    Run a query against the neuPrint Explorer API endpoints.
    
    Parameters:
        endpoint (str): API endpoint (e.g., 'npexplorer/neurons')
        payload (dict): Query parameters
        token (str): Authentication token
    
    Returns:
        dict or None: Query results or None if error
    """
    url = f"{config.NEUPRINT_SERVER}/api/{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    
    logger.info(f"Running explorer query to endpoint '{endpoint}'")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info(f"Explorer query succeeded")
        return result
            
    except Exception as e:
        logger.error(f"Error executing explorer query: {str(e)}")
        if 'response' in locals():
            try:
                logger.error(f"Response status code: {response.status_code}")
                error_detail = response.json()
                logger.error(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                logger.error(f"Response text: {response.text}")
        return None

def main():
    """Run a series of test queries."""
    logger.info(f"Testing neuPrint API at {config.NEUPRINT_SERVER}")
    logger.info(f"Using dataset: {config.NEUPRINT_DATASET}")
    
    # Test 1: Very simple query to get server version
    logger.info("\n--- Test 1: Server Version ---")
    try:
        url = f"{config.NEUPRINT_SERVER}/api/version"
        headers = {"Authorization": f"Bearer {config.NEUPRINT_TOKEN}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        version_info = response.json()
        logger.info(f"Server version: {json.dumps(version_info, indent=2)}")
    except Exception as e:
        logger.error(f"Error getting server version: {str(e)}")
        logger.error("Server connection test failed. Aborting further tests.")
        return False
    
    # Test 2: Simple Cypher query to get neuron count
    logger.info("\n--- Test 2: Neuron Count Query ---")
    count_query = """
    MATCH (n:Neuron)
    RETURN count(n) as neuron_count
    """
    
    count_result = run_cypher_query(count_query)
    if count_result:
        logger.info(f"Raw count result: {json.dumps(count_result, indent=2)}")
        
        # Try to extract count from different result formats
        try:
            if 'data' in count_result and isinstance(count_result['data'], list):
                # Format: {"data": [{"neuron_count": 123}]}
                neuron_count = count_result['data'][0].get('neuron_count', 'unknown')
            elif isinstance(count_result, list):
                # Format: [{"neuron_count": 123}]
                neuron_count = count_result[0].get('neuron_count', 'unknown')
            else:
                neuron_count = "unknown format"
                
            logger.info(f"Neuron count: {neuron_count}")
        except Exception as e:
            logger.error(f"Error extracting neuron count: {str(e)}")
    else:
        logger.warning("Could not get neuron count. Continuing with other tests...")
    
    # Test 3: Alternate approach using neuron explorer endpoint
    logger.info("\n--- Test 3: Using Neurons Explorer Endpoint ---")
    neuron_payload = {
        "dataset": config.NEUPRINT_DATASET,
        "all_segments": True,
        "statuses": ["Traced", "Untraced"],
        "limit": 5
    }
    
    neuron_result = run_explorer_query("npexplorer/neurons", neuron_payload)
    if neuron_result:
        logger.info(f"Neurons explorer result type: {type(neuron_result).__name__}")
        
        # Handle different result formats
        if isinstance(neuron_result, list) and len(neuron_result) > 0:
            # Show only beginning of the result for readability
            sample = neuron_result[:min(2, len(neuron_result))]
            logger.info(f"Sample results: {json.dumps(sample, indent=2)}")
            logger.info(f"Explorer query successful! Found {len(neuron_result)} neurons")
            return True
        elif isinstance(neuron_result, dict):
            logger.info(f"Sample results: {json.dumps(list(neuron_result.items())[:5], indent=2)}")
            if len(neuron_result) > 0:
                logger.info(f"Explorer query successful! Found data in dictionary format")
                return True
        else:
            logger.warning("Explorer query returned unexpected format or empty result")
    
    # Test 4: Very general Cypher query for any neurons
    logger.info("\n--- Test 4: General Neurons Query ---")
    neuron_query = """
    MATCH (n)
    WHERE labels(n) CONTAINS 'Neuron'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 5
    """
    
    neuron_cypher_result = run_cypher_query(neuron_query)
    if neuron_cypher_result:
        # Handle different result formats
        if 'data' in neuron_cypher_result and isinstance(neuron_cypher_result['data'], list):
            neurons_data = neuron_cypher_result['data']
            if len(neurons_data) > 0:
                logger.info(f"General neuron query results: {json.dumps(neurons_data, indent=2)}")
                logger.info(f"Test successful! Found {len(neurons_data)} neurons")
                return True
        elif isinstance(neuron_cypher_result, list) and len(neuron_cypher_result) > 0:
            logger.info(f"General neuron query results: {json.dumps(neuron_cypher_result, indent=2)}")
            logger.info(f"Test successful! Found {len(neuron_cypher_result)} neurons")
            return True
        else:
            logger.info(f"Result has unexpected format: {json.dumps(neuron_cypher_result, indent=2)}")
    
    logger.warning("Could not get any neurons with general query")
    logger.error("All query tests failed or returned no results")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 