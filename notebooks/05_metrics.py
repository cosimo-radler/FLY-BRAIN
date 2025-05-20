"""
Network Metrics Analysis Script

This script uses the metrics module to compute various network metrics for all available
brain regions and models. It separates the computation logic (in src/metrics.py) from
the execution of metrics computation on actual data.

The script:
1. Loads all available brain regions and their models
2. Computes metrics for each graph in parallel using all available CPU cores
3. Organizes results into DataFrames
4. Saves results to CSV files in the results/tables directory
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from functools import partial
import time
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    print("Install tqdm for progress bars: pip install tqdm")
    TQDM_AVAILABLE = False

# Add the project root to Python path to import from src
sys.path.insert(0, str(Path(__file__).parents[1].resolve()))

# Import from src modules
from src.metrics import compute_all_metrics
from src.data_io import (
    get_brain_regions, 
    get_available_models, 
    load_graph, 
    load_null_models,
    save_metrics_results
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("../logs/metrics_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("metrics_analysis")

# Timestamp for file naming
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def process_model(region, model_type, model_index=None, compute_spectral=True, compute_centrality=True):
    """
    Process a specific model for a given brain region.
    
    Args:
        region (str): Brain region name
        model_type (str): Model type (original, configuration_model, etc.)
        model_index (int, optional): Index for models with multiple instances
        compute_spectral (bool): Whether to compute spectral metrics
        compute_centrality (bool): Whether to compute centrality metrics
        
    Returns:
        dict: Metrics for the model or None if loading failed
    """
    model_name = f"{model_type}{'' if model_index is None else f'_{model_index+1}'}"
    logger.info(f"Processing {model_name} model for {region}")
    
    # Load graph based on model type
    if model_type == "original":
        G = load_graph(region, model_type)
    else:
        # For null models, check if we have an index
        if model_index is not None:
            # Here we're trying to load a specific instance from multiple models
            null_models = load_null_models(region, model_type, max_models=model_index+1)
            G = null_models[model_index] if len(null_models) > model_index else None
        else:
            # Just load a single model
            G = load_graph(region, model_type)
    
    if G is None:
        logger.warning(f"Failed to load {model_name} model for {region}")
        return None
    
    # Compute metrics
    try:
        start_time = time.time()
        metrics = compute_all_metrics(G, compute_spectral, compute_centrality)
        elapsed = time.time() - start_time
        
        metrics["model_type"] = model_name
        metrics["region"] = region
        metrics["processing_time_sec"] = elapsed
        
        logger.info(f"Completed metrics for {region} {model_name} in {elapsed:.2f} seconds")
        return metrics
    except Exception as e:
        logger.error(f"Error computing metrics for {region} {model_name}: {str(e)}")
        return None


def analyze_brain_region(region, compute_spectral=True, compute_centrality=True):
    """
    Compute metrics for all models of a specific brain region.
    
    Args:
        region (str): Name of the brain region
        compute_spectral (bool): Whether to compute spectral metrics
        compute_centrality (bool): Whether to compute centrality metrics
        
    Returns:
        pd.DataFrame: DataFrame with metrics for all models of the region
    """
    region_start_time = time.time()
    logger.info(f"Analyzing brain region: {region}")
    
    # Get available models for this region
    available_models = get_available_models(region)
    logger.info(f"Available models for {region}: {available_models}")
    
    if not available_models:
        logger.warning(f"No models found for region {region}")
        return None
    
    # Prepare a list of tasks (region, model_type, model_index)
    tasks = []
    
    # Process original graph if available
    if "original" in available_models:
        tasks.append((region, "original", None))
    
    # Process configuration models if available
    if "configuration_model" in available_models:
        # Try loading multiple CM instances
        cms = load_null_models(region, "configuration_model", max_models=10)
        
        if cms:
            # Add each CM as a separate task
            for i in range(len(cms)):
                tasks.append((region, "configuration_model", i))
        else:
            # Just one CM
            tasks.append((region, "configuration_model", None))
    
    # Process spectral sparsifiers if available
    if "spectral_sparsifier" in available_models:
        # Try loading multiple SSM instances
        ssms = load_null_models(region, "spectral_sparsifier", max_models=10)
        
        if ssms:
            # Add each SSM as a separate task
            for i in range(len(ssms)):
                tasks.append((region, "spectral_sparsifier", i))
        else:
            # Just one SSM
            tasks.append((region, "spectral_sparsifier", None))
    
    # Process tasks and collect metrics
    all_metrics = []
    
    # Use tqdm if available for a progress bar
    task_iter = tqdm(tasks, desc=f"Processing {region}", unit="model") if TQDM_AVAILABLE else tasks
    
    for task in task_iter:
        region, model_type, model_index = task
        metrics = process_model(region, model_type, model_index, compute_spectral, compute_centrality)
        if metrics:
            all_metrics.append(metrics)
    
    # Convert list of metrics dictionaries to DataFrame
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        
        # Ensure region and model_type are the first columns
        cols = ["region", "model_type"] + [col for col in metrics_df.columns if col not in ["region", "model_type"]]
        metrics_df = metrics_df[cols]
        
        # Save region-specific metrics to CSV
        save_metrics_results(metrics_df, "metrics", region, TIMESTAMP)
        
        region_elapsed = time.time() - region_start_time
        logger.info(f"Completed analysis of {region} in {region_elapsed:.2f} seconds")
        
        return metrics_df
    else:
        logger.warning(f"No metrics computed for region {region}")
        return None


def process_region_parallel(region, compute_spectral=True, compute_centrality=True):
    """Wrapper function for parallel processing of a region"""
    try:
        return analyze_brain_region(region, compute_spectral, compute_centrality)
    except Exception as e:
        logger.error(f"Error processing region {region}: {str(e)}")
        return None


def main():
    """
    Main function to analyze all brain regions and their models in parallel.
    """
    total_start_time = time.time()
    logger.info("Starting network metrics analysis")
    
    # Get list of available brain regions
    regions = get_brain_regions()
    logger.info(f"Found {len(regions)} brain regions: {regions}")
    
    if not regions:
        logger.error("No brain regions found. Please check data_io.py and data paths.")
        return
    
    # Determine CPU count for parallel processing
    num_cpus = mp.cpu_count()
    logger.info(f"Using {num_cpus} CPU cores for parallel processing")
    
    # For progress display when using multiprocessing
    if TQDM_AVAILABLE:
        print(f"Processing {len(regions)} brain regions: {', '.join(regions)}")
        print(f"Using {num_cpus} CPU cores in parallel")
        print("Each region will show its own progress bar when processing starts")
    
    # Process regions in parallel
    with mp.Pool(processes=num_cpus) as pool:
        if TQDM_AVAILABLE:
            # Using tqdm with imap for progress updates
            results = list(tqdm(
                pool.imap(process_region_parallel, regions),
                total=len(regions),
                desc="Brain regions",
                unit="region"
            ))
        else:
            results = pool.map(process_region_parallel, regions)
    
    # Filter out None results
    valid_results = [df for df in results if df is not None]
    
    # Combine all metrics into a single DataFrame
    if valid_results:
        all_metrics_df = pd.concat(valid_results, ignore_index=True)
        
        # Save combined metrics to CSV
        output_path = save_metrics_results(all_metrics_df, "all_regions_metrics", timestamp=TIMESTAMP)
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"Completed metrics analysis for {len(valid_results)} brain regions in {total_elapsed:.2f} seconds")
        print(f"\nResults saved to: {output_path}")
        print(f"Total processing time: {total_elapsed:.2f} seconds")
    else:
        logger.warning("No metrics computed for any region")


def process_single_region(region_name):
    """Process a single region - useful for testing or targeted analysis"""
    if region_name not in get_brain_regions():
        logger.error(f"Region {region_name} not found")
        return
    
    start_time = time.time()
    region_metrics = analyze_brain_region(region_name)
    elapsed = time.time() - start_time
    
    if region_metrics is not None:
        logger.info(f"Successfully computed metrics for {region_name} in {elapsed:.2f} seconds")
        print(f"Completed processing {region_name} in {elapsed:.2f} seconds")
        return region_metrics
    else:
        logger.error(f"Failed to compute metrics for {region_name}")
        return None


# Create a simple test graph if we're running this script directly
def test_metrics_calculation():
    """
    Test the metrics calculation on a simple graph.
    """
    G = nx.erdos_renyi_graph(100, 0.1, seed=42)
    
    print("Testing metrics calculation on random graph...")
    start_time = time.time()
    metrics = compute_all_metrics(G)
    elapsed = time.time() - start_time
    
    print(f"Metrics calculation completed in {elapsed:.2f} seconds")
    print("Graph Metrics:")
    for key, value in metrics.items():
        if not isinstance(value, (dict, list, np.ndarray)):
            print(f"  {key}: {value}")
    
    return metrics


if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn' for macOS compatibility
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        # Method already set
        pass
        
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_metrics_calculation()
        else:
            # Process a specific region if provided
            process_single_region(sys.argv[1])
    else:
        # By default, process all regions in parallel
        main()
