#!/usr/bin/env python3
"""
Script to test different query patterns for finding neurons in neuPrint.

This script tries various Cypher query patterns to find neurons in different regions.
"""

import os
import sys
import json
import requests
import logging

# Add the project root to Python path
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
        
        # Extract and count the data
        if 'data' in result and isinstance(result['data'], list):
            data_count = len(result['data'])
            logger.info(f"Query returned {data_count} results")
            
            # Show sample results
            if data_count > 0:
                if isinstance(result['data'][0], list):
                    sample = result['data'][:min(3, data_count)]
                    columns = result.get('columns', [])
                    if columns:
                        logger.info(f"Columns: {columns}")
                    logger.info(f"Sample results: {json.dumps(sample, indent=2)}")
                else:
                    logger.info(f"Sample results: {json.dumps(result['data'][:min(3, data_count)], indent=2)}")
        
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

def test_pattern(pattern_name, query):
    """Test a query pattern and log the results."""
    logger.info(f"\n--- Testing pattern: {pattern_name} ---")
    result = run_cypher_query(query)
    if result and 'data' in result and len(result['data']) > 0:
        logger.info(f"Pattern {pattern_name} SUCCESSFUL!")
        return True
    else:
        logger.warning(f"Pattern {pattern_name} did not return any results")
        return False

def main():
    """Test different query patterns."""
    logger.info(f"Testing different query patterns against neuPrint API")
    logger.info(f"Server: {config.NEUPRINT_SERVER}")
    logger.info(f"Dataset: {config.NEUPRINT_DATASET}")
    
    # First, get the neuron count to verify the API works
    count_query = """
    MATCH (n:Neuron)
    RETURN count(n) as neuron_count
    """
    
    logger.info("\n--- Initial test: Neuron count ---")
    result = run_cypher_query(count_query)
    if not result or 'data' not in result or len(result['data']) == 0:
        logger.error("Count query failed. API may not be working correctly.")
        return False
        
    # Pattern 1: Get all neuron types
    pattern1 = """
    MATCH (n:Neuron)
    RETURN DISTINCT n.type
    LIMIT 20
    """
    test_pattern("Get neuron types", pattern1)
    
    # Pattern 2: Neurons with EB in "type" property
    pattern2 = """
    MATCH (n:Neuron)
    WHERE n.type CONTAINS 'EB'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Type contains 'EB'", pattern2)
    
    # Pattern 3: Neurons with EB in "instance" property
    pattern3 = """
    MATCH (n:Neuron)
    WHERE n.instance CONTAINS 'EB'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Instance contains 'EB'", pattern3)
    
    # Pattern 4: MB Kenyon Cells - using starts with instead of contains
    pattern4 = """
    MATCH (n:Neuron)
    WHERE n.type STARTS WITH 'KC'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Type starts with 'KC'", pattern4)
    
    # Pattern 5: ROI-based query for EB
    pattern5 = """
    MATCH (n:Neuron)-[:In]->(r:Roi)
    WHERE r.name = 'EB'
    RETURN DISTINCT n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Neurons in 'EB' ROI", pattern5)
    
    # Pattern 6: Using regex for matching
    pattern6 = """
    MATCH (n:Neuron)
    WHERE n.type =~ '.*EB.*'
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Type matches regex '.*EB.*'", pattern6)
    
    # Pattern 7: List all ROIs
    pattern7 = """
    MATCH (r:Roi)
    RETURN r.name
    LIMIT 50
    """
    test_pattern("List all ROIs", pattern7)
    
    # Pattern 8: Simple generic query
    pattern8 = """
    MATCH (n:Neuron)
    RETURN n.bodyId, n.type, n.instance
    LIMIT 20
    """
    test_pattern("Generic neuron query", pattern8)
    
    logger.info("\nTesting complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 