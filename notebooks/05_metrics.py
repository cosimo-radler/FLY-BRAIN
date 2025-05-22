"""
Network Metrics Analysis Script (Simplified)

This script uses the metrics module to compute basic network metrics for all available
brain regions and models, and provides a single comprehensive bar graph visualization 
to compare metrics across models.

The script:
1. Loads all available brain regions and their models
2. Computes basic metrics for each graph
3. Organizes results into DataFrames
4. Creates a single comprehensive bar graph visualization
5. Saves results to CSV files and one visualization plot
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

# Create directory for plots if it doesn't exist
FIGURES_DIR = Path("../results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def process_model(region, model_type, model_index=None):
    """
    Process a specific model for a given brain region.
    
    Args:
        region (str): Brain region name
        model_type (str): Model type (original, configuration_model, etc.)
        model_index (int, optional): Index for models with multiple instances
        
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
        metrics = compute_all_metrics(G)
        elapsed = time.time() - start_time
        
        metrics["model_type"] = model_name
        metrics["region"] = region
        metrics["processing_time_sec"] = elapsed
        
        logger.info(f"Completed metrics for {region} {model_name} in {elapsed:.2f} seconds")
        return metrics
    except Exception as e:
        logger.error(f"Error computing metrics for {region} {model_name}: {str(e)}")
        return None


def plot_comprehensive_comparison(all_metrics_df, timestamp=None):
    """
    Create a single comprehensive bar plot comparing metrics across all models and regions.
    
    Args:
        all_metrics_df (DataFrame): DataFrame containing metrics for all regions and models
        timestamp (str, optional): Timestamp for file naming
    """
    if all_metrics_df is None or all_metrics_df.empty:
        logger.warning("No data available to plot comprehensive comparison")
        return
    
    # Select metrics to plot (basic metrics that are meaningful to compare)
    metrics_to_plot = [
        'avg_degree', 'max_degree', 'transitivity', 'avg_clustering', 
        'global_efficiency'
    ]
    
    # Path metrics will have varying names depending on connectivity
    path_metric_names = [col for col in all_metrics_df.columns if 'avg_shortest_path' in col]
    if path_metric_names:
        metrics_to_plot.extend(path_metric_names[:1])  # Add only one path metric
    
    # Add diameter if available
    diameter_names = [col for col in all_metrics_df.columns if 'diameter' in col]
    if diameter_names:
        metrics_to_plot.extend(diameter_names[:1])
    
    # Filter to only metrics that exist in the DataFrame
    metrics_to_plot = [m for m in metrics_to_plot if m in all_metrics_df.columns]
    
    if not metrics_to_plot:
        logger.warning("None of the selected metrics are available")
        return
    
    # Count the number of regions
    regions = all_metrics_df['region'].unique()
    n_regions = len(regions)
    
    # Create a large figure for the comprehensive comparison
    plt.figure(figsize=(20, 12))
    
    # Calculate the layout
    n_metrics = len(metrics_to_plot)
    n_cols = min(3, n_metrics)  # Maximum 3 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Prepare a color map for different model types
    model_types = all_metrics_df['model_type'].unique()
    try:
        color_map = plt.cm.get_cmap('tab10', len(model_types))
    except AttributeError:
        # Matplotlib 3.7+ compatibility
        color_map = plt.colormaps['tab10']
    model_colors = {model: color_map(i) for i, model in enumerate(model_types)}
    
    # Create subplots for each metric
    for i, metric in enumerate(metrics_to_plot):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Set up positions for bar groups
        x = np.arange(n_regions)
        width = 0.8 / len(model_types)  # Width of the bars adjusted for number of model types
        
        # Plot bars for each model type
        for j, model_type in enumerate(model_types):
            # Extract data for this model type
            model_data = all_metrics_df[all_metrics_df['model_type'] == model_type].copy()
            
            # Prepare the data for each region
            region_values = []
            for region in regions:
                value = model_data[model_data['region'] == region][metric].values
                if len(value) > 0:
                    region_values.append(value[0])
                else:
                    region_values.append(np.nan)
            
            # Plot bars
            bars = ax.bar(x + (j - len(model_types)/2 + 0.5) * width, region_values, 
                    width, label=model_type, color=model_colors[model_type])
            
            # Add value labels on top of bars if they're not too crowded
            if len(regions) <= 5:
                for bar_idx, bar in enumerate(bars):
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}', ha='center', va='bottom',
                               fontsize=8, rotation=45)
        
        # Set title and labels
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_xticks(x)
        ax.set_xticklabels(regions, rotation=45, ha='right')
        
        # Add grid lines for readability
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot
        if i == 0:
            ax.legend(title="Model Types", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.suptitle("Comprehensive Metrics Comparison Across Brain Regions", fontsize=16, y=1.02)
    
    # Save the plot to the main figures directory
    output_path = FIGURES_DIR / f"comprehensive_metrics_comparison{'_'+timestamp if timestamp else ''}.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    logger.info(f"Created comprehensive comparison plot saved to {output_path}")
    return output_path


def analyze_brain_region(region):
    """
    Compute metrics for all models of a specific brain region.
    
    Args:
        region (str): Name of the brain region
        
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
    
    # Process coarsened graph if available
    if "coarsened" in available_models:
        tasks.append((region, "coarsened", None))
    
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
        metrics = process_model(region, model_type, model_index)
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


def process_region_parallel(region):
    """Wrapper function for parallel processing of a region"""
    try:
        return analyze_brain_region(region)
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
        
        # Create the comprehensive comparison plot
        plot_path = plot_comprehensive_comparison(all_metrics_df, TIMESTAMP)
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"Completed metrics analysis for {len(valid_results)} brain regions in {total_elapsed:.2f} seconds")
        print(f"\nResults saved to: {output_path}")
        print(f"Comprehensive plot saved to: {plot_path}")
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
