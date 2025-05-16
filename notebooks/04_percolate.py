#!/usr/bin/env python3
"""
Script to perform bond percolation analysis on brain networks.

This script applies both random and degree-based percolation to the cleaned
networks from the data/processed directory, their configuration models,
and generates visualization to compare how different brain regions and models
respond to these attacks.
"""

import os
import sys
import time
import logging
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
import glob

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import project modules
from src import config
from src import percolation
from src import data_io
import src.utils as utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("percolate")

def load_cleaned_networks(regions=None):
    """
    Load all cleaned networks from the processed directory.
    
    Parameters:
        regions (list): List of region codes to load. If None, load all regions.
        
    Returns:
        dict: Dictionary with region codes as keys and networks as values
    """
    # Get regions from config if not specified
    if regions is None:
        regions = config.BRAIN_REGIONS.keys()
    
    networks = {}
    
    for region in regions:
        try:
            # Construct path to cleaned network file
            network_path = os.path.join(config.DATA_PROCESSED_DIR, f"{region.lower()}_cleaned.gexf")
            
            if os.path.exists(network_path):
                network = nx.read_gexf(network_path)
                networks[region] = network
                logger.info(f"Loaded {region} network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
            else:
                logger.warning(f"Network file not found for region {region}: {network_path}")
                
        except Exception as e:
            logger.error(f"Error loading network for region {region}: {str(e)}")
    
    return networks

def load_configuration_models(regions, seeds=None, scaled=False, n_target=1500):
    """
    Load configuration models for specified regions and seeds.
    
    Parameters:
        regions (list): List of region codes to load models for
        seeds (list): List of seed values to load models for
        scaled (bool): Whether to load scaled models (otherwise, unscaled models are loaded)
        n_target (int): Target node count for scaled models
        
    Returns:
        dict: Dictionary with region_seed keys and configuration model networks as values
    """
    if seeds is None:
        # Default seeds used in configuration model generation
        seeds = [42, 123, 456] 
    
    model_type = "scaled_configuration" if scaled else "configuration"
    models = {}
    
    for region in regions:
        for seed in seeds:
            try:
                # Load the model
                if scaled:
                    model = data_io.load_null_model(region, model_type, seed=seed, n_target=n_target)
                else:
                    model = data_io.load_null_model(region, model_type, seed=seed)
                
                if model is not None:
                    # Create a key that combines region and seed
                    model_key = f"{region}_cm"
                    if scaled:
                        model_key = f"{region}_scaled_cm"
                    
                    # Add unique identifier with seed
                    model_key = f"{model_key}_{seed}"
                    
                    models[model_key] = model
                    logger.info(f"Loaded {model_type} model for {region} (seed {seed}): "
                               f"{model.number_of_nodes()} nodes, {model.number_of_edges()} edges")
                else:
                    logger.warning(f"No {model_type} model found for region {region} with seed {seed}")
                    
            except Exception as e:
                logger.error(f"Error loading {model_type} model for region {region}, seed {seed}: {str(e)}")
    
    return models

def main():
    """Main function to run the script."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run percolation analysis on brain networks')
    parser.add_argument('--regions', nargs='+', help='Brain regions to process (e.g., MB FB EB)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials for random percolation')
    parser.add_argument('--steps', type=int, default=20, help='Number of percolation steps')
    parser.add_argument('--model-seeds', nargs='+', type=int, default=[42, 123], 
                      help='Seeds of configuration models to analyze')
    parser.add_argument('--original-only', action='store_true', 
                      help='Only run percolation on original networks (not config models)')
    parser.add_argument('--scaled-only', action='store_true',
                      help='Only run percolation on scaled config models')
    parser.add_argument('--unscaled-only', action='store_true',
                      help='Only run percolation on unscaled config models')
    args = parser.parse_args()
    
    logger.info("Starting percolation analysis")
    logger.info(f"Regions: {args.regions}")
    logger.info(f"Seed: {args.seed}, Trials: {args.trials}, Steps: {args.steps}")
    
    try:
        # Define fractions to remove, include 1.0 for complete edge removal
        fractions = np.linspace(0.0, 1.0, args.steps + 1)
        
        # Load original networks
        original_networks = load_cleaned_networks(regions=args.regions)
        
        if not original_networks:
            logger.error("No networks found. Please run the data cleaning script first.")
            return
        
        # Combined networks for percolation (start with originals)
        all_networks = {k: v for k, v in original_networks.items()}
        
        # Load configuration models if requested
        if not args.original_only:
            regions = list(original_networks.keys())
            
            # Load unscaled configuration models
            if not args.scaled_only:
                unscaled_models = load_configuration_models(regions, seeds=args.model_seeds, scaled=False)
                all_networks.update(unscaled_models)
                
            # Load scaled configuration models
            if not args.unscaled_only:
                scaled_models = load_configuration_models(regions, seeds=args.model_seeds, scaled=True, n_target=1500)
                all_networks.update(scaled_models)
        
        # Run percolation experiments
        start_time = time.time()
        results, thresholds = percolation.run_percolation_experiments(
            all_networks, 
            fractions=fractions, 
            seed=args.seed, 
            num_trials=args.trials
        )
        
        # Calculate total time
        total_time = time.time() - start_time
        logger.info(f"Percolation experiments completed in {total_time:.2f} seconds")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = percolation.save_percolation_results(results, thresholds, timestamp)
        
        # Print summary
        print("\nPercolation Analysis Summary:")
        
        # First, original networks
        print("\nOriginal Networks:")
        for region in original_networks.keys():
            print(f"  {region} ({config.BRAIN_REGIONS[region]}):")
            print(f"    Analytic percolation threshold p_c = {thresholds[region]:.4f}")
            for strategy in results[region].keys():
                max_fraction = max(results[region][strategy].keys())
                final_lcc = results[region][strategy][max_fraction]
                print(f"    {strategy}: {final_lcc:.3f} LCC size at {max_fraction:.1f} fraction removed")
        
        # Then, configuration models
        if not args.original_only:
            # Unscaled
            if not args.scaled_only:
                print("\nUnscaled Configuration Models:")
                for key in results.keys():
                    if "_cm_" in key and "scaled" not in key:
                        region, _, seed = key.split("_")
                        print(f"  {region} (seed {seed}):")
                        if key in thresholds:
                            print(f"    Analytic percolation threshold p_c = {thresholds[key]:.4f}")
                        for strategy in results[key].keys():
                            max_fraction = max(results[key][strategy].keys())
                            final_lcc = results[key][strategy][max_fraction]
                            print(f"    {strategy}: {final_lcc:.3f} LCC size at {max_fraction:.1f} fraction removed")
            
            # Scaled
            if not args.unscaled_only:
                print("\nScaled Configuration Models:")
                for key in results.keys():
                    if "scaled_cm" in key:
                        region, _, _, seed = key.split("_")
                        print(f"  {region} (seed {seed}):")
                        if key in thresholds:
                            print(f"    Analytic percolation threshold p_c = {thresholds[key]:.4f}")
                        for strategy in results[key].keys():
                            max_fraction = max(results[key][strategy].keys())
                            final_lcc = results[key][strategy][max_fraction]
                            print(f"    {strategy}: {final_lcc:.3f} LCC size at {max_fraction:.1f} fraction removed")
        
        print(f"\nGenerated plots:")
        for key, file in saved_files.items():
            if key in ['comprehensive_comparison', 'percolation_by_strategy']:
                print(f"  {key}: {file}")
                
        print(f"\nData files:")
        for key, file in saved_files.items():
            if key not in ['comprehensive_comparison', 'percolation_by_strategy']:
                print(f"  {key}: {file}")
        
        print("\nA comprehensive visualization has been created showing all models by region and attack strategy,")
        print("along with a percolation by attack strategy plot comparing regions.")
        
        logger.info("Percolation analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\nERROR: Could not complete percolation analysis due to: {str(e)}")
        print("Check logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 