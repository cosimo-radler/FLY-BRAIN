#!/usr/bin/env python3
"""
Script to clean brain region networks using the preprocessing module.

This script applies cleaning operations from preprocessing.py to the raw networks
created by 01_fetch_all_regions.py, producing cleaned networks ready for analysis.
"""

import os
import sys
import json
import time
import logging
import networkx as nx
from pathlib import Path
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
from src import config
from src import preprocessing
import src.utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("clean_data")

def clean_all_regions(force=False):
    """
    Clean networks for all brain regions defined in config.
    
    Parameters:
        force (bool): If True, clean and overwrite existing cleaned data
        
    Returns:
        dict: Dictionary with region codes as keys and cleaning results as values
    """
    # Get regions from config
    regions = config.BRAIN_REGIONS.keys()
    
    logger.info(f"Starting data cleaning for {len(regions)} regions: {', '.join(regions)}")
    
    # Store results
    results = {}
    
    # Track performance metrics
    performance = {
        'start_time': datetime.now(),
        'region_times': {},
        'total_nodes_before': 0,
        'total_nodes_after': 0,
        'total_edges_before': 0,
        'total_edges_after': 0
    }
    
    # Process each region
    for region in regions:
        region_start = time.time()
        
        logger.info(f"Cleaning region: {region} ({config.BRAIN_REGIONS[region]})")
        
        try:
            # Load the network
            raw_network_file = os.path.join(config.DATA_RAW_DIR, f"{region.lower()}_network.gexf")
            processed_network_file = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_cleaned_network.gexf")
            
            # Skip if already cleaned and force is False
            if os.path.exists(processed_network_file) and not force:
                logger.info(f"Cleaned network for {region} already exists. Use --force to overwrite.")
                try:
                    # Load the existing cleaned network to get stats
                    cleaned_network = nx.read_gexf(processed_network_file)
                    original_network = nx.read_gexf(raw_network_file)
                    
                    results[region] = {
                        'nodes_before': original_network.number_of_nodes(),
                        'edges_before': original_network.number_of_edges(),
                        'nodes_after': cleaned_network.number_of_nodes(),
                        'edges_after': cleaned_network.number_of_edges(),
                        'processing_time': 0,  # Not reprocessed
                        'status': 'SKIPPED'
                    }
                    continue
                except Exception as e:
                    logger.warning(f"Error reading existing networks: {e}. Will reprocess.")
            
            # Check if raw network exists
            if not os.path.exists(raw_network_file):
                logger.error(f"Raw network file not found: {raw_network_file}")
                results[region] = {'error': 'Raw network file not found', 'status': 'ERROR'}
                continue
                
            # Load the raw network
            network = nx.read_gexf(raw_network_file)
            logger.info(f"Loaded raw network with {network.number_of_nodes()} nodes and {network.number_of_edges()} edges")
            
            # Apply cleaning operations
            # 1. Remove self-loops
            network_no_loops = preprocessing.remove_self_loops(network)
            
            # 2. Skip threshold weak connections - as requested
            network_thresholded = network_no_loops
            
            # 3. Extract largest connected component
            cleaned_network = preprocessing.extract_largest_connected_component(network_thresholded)
            
            # Save the cleaned network
            preprocessing.save_cleaned_network(cleaned_network, region)
            
            # Update performance metrics
            region_time = time.time() - region_start
            performance['region_times'][region] = region_time
            performance['total_nodes_before'] += network.number_of_nodes()
            performance['total_nodes_after'] += cleaned_network.number_of_nodes()
            performance['total_edges_before'] += network.number_of_edges()
            performance['total_edges_after'] += cleaned_network.number_of_edges()
            
            logger.info(f"Region {region} cleaned in {region_time:.2f} seconds")
            logger.info(f"Before: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            logger.info(f"After: {cleaned_network.number_of_nodes()} nodes, {cleaned_network.number_of_edges()} edges")
            
            # Store results
            results[region] = {
                'nodes_before': network.number_of_nodes(),
                'edges_before': network.number_of_edges(),
                'nodes_after': cleaned_network.number_of_nodes(),
                'edges_after': cleaned_network.number_of_edges(),
                'processing_time': region_time,
                'status': 'SUCCESS'
            }
        
        except Exception as e:
            logger.error(f"Error cleaning region {region}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            results[region] = {
                'error': str(e),
                'status': 'ERROR'
            }
    
    # Calculate total time
    performance['end_time'] = datetime.now()
    performance['total_time'] = (performance['end_time'] - performance['start_time']).total_seconds()
    
    # Save performance summary
    save_cleaning_summary(performance, results)
    
    logger.info(f"All regions cleaned in {performance['total_time']:.2f} seconds")
    logger.info(f"Total nodes before: {performance['total_nodes_before']} -> after: {performance['total_nodes_after']}")
    logger.info(f"Total edges before: {performance['total_edges_before']} -> after: {performance['total_edges_after']}")
    
    return results

def save_cleaning_summary(performance, results):
    """
    Save a summary of the data cleaning performance.
    
    Parameters:
        performance (dict): Performance metrics
        results (dict): Results from cleaning
    """
    # Ensure the tables directory exists
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    
    # Create a timestamp
    timestamp = performance['start_time'].strftime('%Y%m%d_%H%M%S')
    
    # Create a summary DataFrame
    import pandas as pd
    summary = []
    
    for region, data in results.items():
        if data.get('status') == 'ERROR':
            summary.append({
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Nodes_Before': 0,
                'Nodes_After': 0,
                'Edges_Before': 0,
                'Edges_After': 0,
                'Processing_Time': 0,
                'Status': 'ERROR',
                'Error': data.get('error', 'Unknown error')
            })
        else:
            summary.append({
                'Region': region,
                'Region_Name': config.BRAIN_REGIONS[region],
                'Nodes_Before': data.get('nodes_before', 0),
                'Nodes_After': data.get('nodes_after', 0),
                'Edges_Before': data.get('edges_before', 0),
                'Edges_After': data.get('edges_after', 0),
                'Nodes_Reduction_Pct': round((1 - data.get('nodes_after', 0) / max(data.get('nodes_before', 1), 1)) * 100, 2),
                'Edges_Reduction_Pct': round((1 - data.get('edges_after', 0) / max(data.get('edges_before', 1), 1)) * 100, 2),
                'Processing_Time': data.get('processing_time', 0),
                'Status': data.get('status', 'UNKNOWN'),
                'Error': data.get('error', None)
            })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(config.TABLES_DIR, f"cleaning_summary_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    logger.info(f"Cleaning summary saved to {summary_file}")
    
    # Create a more detailed report as JSON
    cleaning_report = {
        'timestamp': timestamp,
        'total_time': performance['total_time'],
        'total_nodes_before': performance['total_nodes_before'],
        'total_nodes_after': performance['total_nodes_after'],
        'total_edges_before': performance['total_edges_before'],
        'total_edges_after': performance['total_edges_after'],
        'region_times': performance['region_times'],
        'results': results
    }
    
    # Save detailed report
    report_file = os.path.join(config.TABLES_DIR, f"cleaning_report_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(cleaning_report, f, indent=2)
    
    logger.info(f"Detailed report saved to {report_file}")

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Clean network data for all brain regions')
    parser.add_argument('--force', action='store_true', help='Force cleaning even if cleaned data exists')
    args = parser.parse_args()
    
    logger.info("Starting data cleaning process")
    logger.info(f"Force mode: {args.force}")
    
    try:
        # Clean all regions
        results = clean_all_regions(force=args.force)
        
        # Print a simple summary
        print("\nCleaning Summary:")
        for region, data in results.items():
            if data.get('status') == 'ERROR':
                print(f"  {region}: ERROR - {data.get('error', 'Unknown error')}")
            elif data.get('status') == 'SKIPPED':
                print(f"  {region}: SKIPPED (already cleaned)")
            else:
                nodes_reduction = round((1 - data.get('nodes_after', 0) / max(data.get('nodes_before', 1), 1)) * 100, 2)
                edges_reduction = round((1 - data.get('edges_after', 0) / max(data.get('edges_before', 1), 1)) * 100, 2)
                print(f"  {region}: {data.get('nodes_before', 0)} -> {data.get('nodes_after', 0)} nodes ({nodes_reduction}% reduction), "
                      f"{data.get('edges_before', 0)} -> {data.get('edges_after', 0)} edges ({edges_reduction}% reduction)")
        
        logger.info("Data cleaning completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nERROR: Could not complete data cleaning due to: {str(e)}")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 