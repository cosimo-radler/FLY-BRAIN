#!/usr/bin/env python3
"""
Script to fetch all brain regions data using the neuprint_client.

This is the canonical data fetching script to standardize data acquisition
from the neuPrint API using the official client library.
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import networkx as nx
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import our neuprint client and config
from src.neuprint_client import NeuPrintInterface
import src.config as config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fetch_all_regions")

def fetch_all_regions(force=False):
    """
    Fetch data for all brain regions defined in config.
    
    Parameters:
        force (bool): If True, fetch and overwrite existing data
        
    Returns:
        dict: Dictionary with region codes as keys and results as values
    """
    # Initialize the neuPrint client
    npi = NeuPrintInterface()
    
    # Get regions from config
    regions = config.BRAIN_REGIONS.keys()
    
    logger.info(f"Starting data fetch for {len(regions)} regions: {', '.join(regions)}")
    
    # Store results
    results = {}
    
    # Track performance metrics
    performance = {
        'start_time': datetime.now(),
        'region_times': {},
        'total_neurons': 0,
        'total_connections': 0
    }
    
    # Process each region
    for region in regions:
        region_start = time.time()
        
        logger.info(f"Processing region: {region} ({config.BRAIN_REGIONS[region]})")
        
        try:
            # Fetch and process the region
            neurons, connectivity, network = npi.process_region(region, force=force)
            
            # Create a matrix from the connectivity data
            matrix = npi.connectivity_to_matrix(connectivity)
            matrix_file = npi.save_matrix_to_csv(matrix, region)
            
            # Update performance metrics
            region_time = time.time() - region_start
            performance['region_times'][region] = region_time
            performance['total_neurons'] += len(neurons)
            performance['total_connections'] += len(connectivity)
            
            logger.info(f"Region {region} processed in {region_time:.2f} seconds")
            logger.info(f"Found {len(neurons)} neurons and {len(connectivity)} connections")
            logger.info(f"Created matrix with shape {matrix.shape}")
            
            # Store results
            results[region] = {
                'neurons_count': len(neurons),
                'connections_count': len(connectivity),
                'network_nodes': network.number_of_nodes(),
                'network_edges': network.number_of_edges(),
                'matrix_shape': matrix.shape,
                'processing_time': region_time
            }
        
        except Exception as e:
            logger.error(f"Error processing region {region}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            results[region] = {
                'error': str(e)
            }
    
    # Calculate total time
    performance['end_time'] = datetime.now()
    performance['total_time'] = (performance['end_time'] - performance['start_time']).total_seconds()
    
    # Save performance summary
    save_performance_summary(performance, results)
    
    logger.info(f"All regions processed in {performance['total_time']:.2f} seconds")
    logger.info(f"Total neurons: {performance['total_neurons']}")
    logger.info(f"Total connections: {performance['total_connections']}")
    
    return results

def save_performance_summary(performance, results):
    """
    Save a summary of the data fetching performance.
    
    Parameters:
        performance (dict): Performance metrics
        results (dict): Results from fetching
    """
    # Ensure the tables directory exists
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # Create a timestamp
    timestamp = performance['start_time'].strftime('%Y%m%d_%H%M%S')
    
    # Create a summary DataFrame
    summary = []
    
    for region, data in results.items():
        if 'error' in data:
            summary.append({
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Neurons_Count': 0,
                'Connections_Count': 0,
                'Processing_Time': 0,
                'Matrix_Shape': None,
                'Status': 'ERROR',
                'Error': data['error']
            })
        else:
            matrix_shape_str = f"{data['matrix_shape'][0]}x{data['matrix_shape'][1]}" if 'matrix_shape' in data else "N/A"
            
            summary.append({
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Neurons_Count': data['neurons_count'],
                'Connections_Count': data['connections_count'],
                'Matrix_Shape': matrix_shape_str,
                'Processing_Time': data['processing_time'],
                'Status': 'SUCCESS',
                'Error': None
            })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(config.TABLES_DIR, f"fetch_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Performance summary saved to {summary_file}")
    
    # Create a more detailed report as JSON
    performance_report = {
        'timestamp': timestamp,
        'total_time': performance['total_time'],
        'total_neurons': performance['total_neurons'],
        'total_connections': performance['total_connections'],
        'region_times': performance['region_times'],
        'results': results
    }
    
    # Save detailed report
    report_file = os.path.join(config.TABLES_DIR, f"fetch_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    logger.info(f"Detailed report saved to {report_file}")

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fetch data for all brain regions')
    parser.add_argument('--force', action='store_true', help='Force fetching even if data exists')
    args = parser.parse_args()
    
    logger.info("Starting data fetch process")
    logger.info(f"Force mode: {args.force}")
    
    try:
        # Fetch all regions
        results = fetch_all_regions(force=args.force)
        
        # Print a simple summary
        print("\nFetch Summary:")
        for region, data in results.items():
            if 'error' in data:
                print(f"  {region}: ERROR - {data['error']}")
            else:
                matrix_info = f", Matrix: {data['matrix_shape'][0]}x{data['matrix_shape'][1]}" if 'matrix_shape' in data else ""
                print(f"  {region}: {data['neurons_count']} neurons, {data['connections_count']} connections{matrix_info}")
        
        logger.info("Data fetch completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nERROR: Could not complete data fetch due to: {str(e)}")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 