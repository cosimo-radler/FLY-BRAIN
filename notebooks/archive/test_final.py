#!/usr/bin/env python3
"""
Final test script for neuPrint API.

This script uses the findings from previous tests to correctly access the neuPrint API.
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
        dict: Full response from the API
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
        logger.info(f"Query succeeded")
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

def extract_data_from_result(result):
    """
    Extract data from the neuPrint API response format.
    
    Parameters:
        result (dict): API response
        
    Returns:
        list: Extracted data rows
    """
    if not result:
        return []
        
    if 'data' in result and isinstance(result['data'], list):
        data = result['data']
        columns = result.get('columns', [])
        
        # If data is a list of lists, convert to list of dicts using column names
        if data and isinstance(data[0], list) and columns:
            formatted_data = []
            for row in data:
                row_dict = {}
                for i, value in enumerate(row):
                    if i < len(columns):
                        row_dict[columns[i]] = value
                formatted_data.append(row_dict)
            return formatted_data
        return data
    return []

def get_neuron_count():
    """Get total count of neurons in the dataset."""
    query = """
    MATCH (n:Neuron)
    RETURN count(n) as neuron_count
    """
    
    result = run_cypher_query(query)
    if not result:
        return None
        
    try:
        # Extract count from the standard format - a list inside 'data'
        count = result['data'][0][0]
        return count
    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Could not extract neuron count: {str(e)}")
        logger.info(f"Raw result: {json.dumps(result, indent=2)}")
        return None

def get_neurons_by_region(region_prefix):
    """
    Get neurons from a specific brain region.
    
    Parameters:
        region_prefix (str): Prefix for the brain region (e.g., 'EB', 'FB')
        
    Returns:
        list: List of neurons
    """
    # Using bodyId explicitly since we found this works
    query = f"""
    MATCH (n:Neuron)
    WHERE n.bodyId IS NOT NULL AND n.instance CONTAINS '{region_prefix}'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    
    result = run_cypher_query(query)
    if not result:
        return []
        
    # Extract and format the data
    data = extract_data_from_result(result)
    return data

def get_connectivity(neuron_ids, min_weight=3):
    """
    Get connectivity between neurons.
    
    Parameters:
        neuron_ids (list): List of bodyIds
        min_weight (int): Minimum connection weight
        
    Returns:
        list: Connectivity data
    """
    # Convert list to string for Cypher query
    id_str = str(neuron_ids)
    
    query = f"""
    MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
    WHERE a.bodyId IN {id_str} 
        AND b.bodyId IN {id_str}
        AND c.weight >= {min_weight}
    RETURN a.bodyId AS source, b.bodyId AS target, 
           c.weight AS weight, c.roiInfo AS roiInfo
    ORDER BY source, target
    LIMIT 100
    """
    
    result = run_cypher_query(query)
    if not result:
        return []
        
    # Extract and format the data
    data = extract_data_from_result(result)
    return data

def main():
    """Run a series of tests to verify API functionality."""
    logger.info(f"Starting final neuPrint API tests with server: {config.NEUPRINT_SERVER}")
    logger.info(f"Dataset: {config.NEUPRINT_DATASET}")
    
    # Step 1: Check neuron count
    logger.info("\n--- Step 1: Getting neuron count ---")
    count = get_neuron_count()
    if count:
        logger.info(f"Total neuron count: {count}")
    else:
        logger.error("Failed to get neuron count")
        return False
    
    # Step 2: Get neurons from a specific region (trying all region prefixes)
    regions_to_test = list(config.BRAIN_REGIONS.keys())
    success = False
    
    for region in regions_to_test:
        logger.info(f"\n--- Step 2: Getting neurons from {region} region ---")
        neurons = get_neurons_by_region(region)
        
        if neurons:
            logger.info(f"Found {len(neurons)} neurons in {region} region")
            logger.info(f"Sample neurons: {json.dumps(neurons[:3], indent=2)}")
            success = True
            
            # Step 3: Test connectivity if we found some neurons
            if len(neurons) >= 2:
                logger.info(f"\n--- Step 3: Testing connectivity for {region} neurons ---")
                
                # Extract bodyIds from neurons
                neuron_ids = []
                for neuron in neurons:
                    # Different formats based on our test results
                    if isinstance(neuron, dict) and 'n.bodyId' in neuron:
                        neuron_ids.append(neuron['n.bodyId'])
                    elif isinstance(neuron, dict) and 'bodyId' in neuron:
                        neuron_ids.append(neuron['bodyId']) 
                    elif isinstance(neuron, list) and len(neuron) > 0:
                        neuron_ids.append(neuron[0])
                
                if neuron_ids:
                    logger.info(f"Testing connectivity for {len(neuron_ids)} neuron IDs: {neuron_ids[:5]}")
                    connectivity = get_connectivity(neuron_ids)
                    
                    if connectivity:
                        logger.info(f"Found {len(connectivity)} connections between neurons")
                        logger.info(f"Sample connections: {json.dumps(connectivity[:3], indent=2)}")
                        return True
                    else:
                        logger.warning(f"No connectivity found between {region} neurons")
        else:
            logger.warning(f"No neurons found in {region} region")
    
    if success:
        logger.info("API tests completed with some success")
        return True
    else:
        logger.error("All region queries failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 