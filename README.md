# Fly Brain Connectome Analysis

This repository contains code for fetching, processing, and analyzing Drosophila brain connectome data from the neuPrint API.

## Current Project State

The current implementation includes:
- Fetching connectivity data for specific brain regions from the neuPrint API
- Cleaning networks by removing self-loops and extracting largest connected components
- Saving both raw and cleaned networks as GEXF files
- Generating configuration models (both unscaled and scaled to target node count)
- Computing various network metrics and statistics
- Performing bond percolation analysis with different edge removal strategies
- Comprehensive visualization comparing original networks with their null models
- Degree distribution comparisons across different model types
- Laplacian matrix analysis and eigenvalue computations
- Coarsened network comparisons

## Directory Structure

```
FLY BRAIN/
├── data/                         # All graph data files
│   ├── raw/                      # └── Original downloads from API
│   ├── processed/                # └── Cleaned networks (self-loops removed, LCC extracted)
│   │   └── null_models/          #     └── Configuration models (unscaled and scaled)
│   └── Coarsened Networks (0.5)/ # └── Coarsened/sparsified networks at 50% density
│
├── notebooks/                    # Scripts and notebooks for data processing
│   ├── 01_fetch_all_regions.py   # └── Download data for all regions from API ⚠️ SLOW
│   ├── 02_clean_data.py          # └── Clean networks and extract largest components
│   ├── 03_configuration_models.py # └── Generate configuration models ⚠️ SLOW
│   ├── 04_percolate.py           # └── Perform bond percolation analysis ⚠️ SLOW
│   ├── 05_metrics.py             # └── Compute comprehensive network metrics ⚠️ SLOW
│   ├── 06_coarsened_percolation.py # └── Compare percolation on original vs coarsened networks ⚠️ SLOW
│   ├── 07_degree_distribution_comparison.py # └── Compare degree distributions across models
│   ├── 07_degree_distribution_comparison_multi.py # └── Multi-region degree comparison ⚠️ SLOW
│   ├── 08_visualization.py       # └── 3D neuron morphology visualization
│   ├── 09_Laplacian.py           # └── Compute Laplacian matrices and eigenvalues ⚠️ SLOW
│   ├── 09_Laplacian_utils.py     # └── Utility functions for Laplacian analysis
│   ├── 09_Laplacian_eigenval_viz.py # └── Eigenvalue distribution visualizations
│   ├── 09_Laplacian_clean_viz.py # └── Clean Laplacian visualization plots
│   └── archive/                  # └── Archive of older scripts
│
├── src/                          # Python modules—core logic
│   ├── config.py                 # └── Configuration settings and parameters
│   ├── data_io.py                # └── Functions to fetch, cache, and read/write graph files
│   ├── metrics.py                # └── Compute network statistics (degree, clustering, path-length)
│   ├── neuprint_client.py        # └── Interface to the neuPrint API
│   ├── null_models.py            # └── Build plain & clustered CMs, enforce connectivity
│   ├── percolation.py            # └── Simulate bond removal (random/targeted) and record Sₘₐₓ(p)
│   ├── preprocessing.py          # └── LCC extraction, cleaning, normalization
│   └── utils.py                  # └── Helper utilities, logging, randomization
│
├── results/                      # Outputs from analyses
│   ├── figures/                  # └── Plots and visualizations
│   ├── tables/                   # └── Summary statistics and reports
│   └── laplacian_matrices/       # └── Stored Laplacian matrices as NPZ files
│
├── logs/                         # Log files from script executions
├── requirements.txt              # pip-installable dependencies
└── README.md                     # This file
```

## Performance Notes

Scripts marked with ⚠️ **SLOW** may take significant time to run:

- **01_fetch_all_regions.py**: API requests can be slow, depends on network and server response
- **03_configuration_models.py**: Generates multiple configuration models with connectivity enforcement
- **04_percolate.py**: Runs multiple percolation trials across different strategies
- **05_metrics.py**: Uses multiprocessing but computes extensive metrics for all networks
- **06_coarsened_percolation.py**: Compares percolation across original and coarsened networks
- **07_degree_distribution_comparison_multi.py**: Processes multiple regions with optional parallel processing
- **09_Laplacian.py**: Eigenvalue computations are computationally intensive for large networks

For faster testing, consider:
- Using `--regions` parameter to limit to specific brain regions
- Using `--seeds` parameter to limit configuration model instances
- Running scripts on smaller subsets first

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your neuPrint API credentials (required for data fetching)

## Usage

### Data Fetching ⚠️ SLOW

To fetch data for all brain regions:

```
python notebooks/01_fetch_all_regions.py
```

This script will:
1. Connect to the neuPrint API
2. Fetch neurons for each brain region
3. Fetch connectivity data
4. Create NetworkX graphs
5. Save raw graphs as GEXF files in the `data/raw` directory
6. Create adjacency matrices in CSV format
7. Save performance reports in the `results/tables` directory

**Note:** This can take several minutes to hours depending on network connectivity and data size.

### Data Cleaning

To clean the fetched networks:

```
python notebooks/02_clean_data.py
```

This script will:
1. Load the raw networks from `data/raw`
2. Remove self-loops
3. Extract the largest connected component
4. Save the cleaned networks to the `data/processed` directory
5. Generate cleaning reports in the `results/tables` directory

Add the `--force` flag to reprocess networks even if they already exist:

```
python notebooks/02_clean_data.py --force
```

### Configuration Models ⚠️ SLOW

To generate configuration models for the cleaned networks:

```
python notebooks/03_configuration_models.py
```

This script will:
1. Load the cleaned networks from `data/processed`
2. Generate unscaled configuration models preserving the exact degree sequence
3. Generate scaled configuration models with a target node count (default: 1500 nodes)
4. Ensure connectedness of all models
5. Save models to `data/processed/null_models/`
6. Generate summary reports in `results/tables`

Optional parameters:
- `--regions`: List of brain regions to process (e.g., `EB FB MB`)
- `--seeds`: List of random seeds for model generation (default: `42 123 456 789 101112`)
- `--n-target`: Target node count for scaled models (default: 1500)
- `--force`: Regenerate models even if they already exist
- `--no-connect`: Skip ensuring models are connected

Example:
```
python notebooks/03_configuration_models.py --regions EB MB --seeds 42 123 --n-target 1000
```

### Percolation Analysis ⚠️ SLOW

To perform bond percolation analysis on the cleaned networks and configuration models:

```
python notebooks/04_percolate.py
```

This script will:
1. Load the cleaned networks from `data/processed`
2. Load configuration models from `data/processed/null_models/`
3. Perform percolation with three different strategies:
   - Random edge removal
   - Targeted removal of high-degree nodes first
   - Targeted removal of low-degree nodes first
4. Generate comprehensive visualizations comparing all networks and models
5. Save results to CSV in `results/tables`
6. Save plots in `results/figures`

Optional parameters:
- `--regions`: List of brain regions to process (e.g., `EB FB MB`)
- `--model-seeds`: List of configuration model seeds to analyze (default: `42 123`)
- `--trials`: Number of trials for random percolation (default: 5)
- `--steps`: Number of percolation steps (default: 20)
- `--seed`: Random seed for reproducibility (default: 42)
- `--original-only`: Only analyze original networks (not config models)
- `--scaled-only`: Only analyze scaled configuration models
- `--unscaled-only`: Only analyze unscaled configuration models

Example:
```
python notebooks/04_percolate.py --regions EB FB MB LH AL --steps 20
```

### Network Metrics Analysis ⚠️ SLOW

To compute comprehensive network metrics for all networks and models:

```
python notebooks/05_metrics.py
```

This script will:
1. Load all available brain regions and their models
2. Compute basic metrics for each graph using multiprocessing
3. Organize results into DataFrames
4. Create comprehensive bar graph visualizations
5. Save results to CSV files and visualization plots

The script uses parallel processing to speed up computation but can still take considerable time for large networks.

### Coarsened Network Percolation ⚠️ SLOW

To compare bond percolation between original and coarsened networks:

```
python notebooks/06_coarsened_percolation.py
```

This script will:
1. Load original cleaned networks from `data/processed`
2. Load coarsened networks from `data/Coarsened Networks (0.5)`
3. Run percolation experiments on both network types
4. Generate multipanel comparison figures
5. Save results and visualizations

### Degree Distribution Comparison

To compare degree distributions across different network models:

```
python notebooks/07_degree_distribution_comparison.py --region FB
```

This script will:
1. Load original, configuration model, scaled configuration model, and coarsened networks
2. Compute normalized degree distributions
3. Create comparison plots on log-log scales
4. Save figures with detailed annotations

For multi-region analysis ⚠️ SLOW:

```
python notebooks/07_degree_distribution_comparison_multi.py --parallel
```

Optional parameters:
- `--regions`: List of brain regions to process
- `--parallel`: Use multiprocessing for faster execution
- `--output-dir`: Custom output directory for results

### 3D Visualization

To visualize the 3D structure of neurons in specific brain regions:

```
python notebooks/08_visualization.py
```

This script will:
1. Connect to the neuPrint API
2. Display a list of available brain regions
3. Allow you to select a specific region for visualization

To visualize a specific region directly:

```
python notebooks/08_visualization.py --region FB --limit 25
```

Optional parameters:
- `--region`: Brain region to visualize (e.g., FB, EB, MB)
- `--limit`: Maximum number of neurons to fetch (default: 50)
- `--seed`: Random seed for reproducibility (default: 42)

### Laplacian Matrix Analysis ⚠️ SLOW

To compute Laplacian matrices and eigenvalues for all networks:

```
python notebooks/09_Laplacian.py
```

This script will:
1. Load all original brain region networks
2. Compute both standard and normalized Laplacian matrices
3. Compute Laplacian eigenvalues and related metrics (algebraic connectivity, spectral gap)
4. Save results to `results/tables/` and matrices to `results/laplacian_matrices/`

**Note:** Eigenvalue computations are computationally intensive and can take significant time for large networks.

#### Laplacian Visualization Tools

After running the main Laplacian analysis, use these visualization scripts:

```
python notebooks/09_Laplacian_utils.py
```

This provides utility functions for analyzing and plotting Laplacian data.

```
python notebooks/09_Laplacian_eigenval_viz.py
```

Creates clean eigenvalue distribution plots for all regions and models.

```
python notebooks/09_Laplacian_clean_viz.py
```

Generates clean summary visualizations of Laplacian metrics across regions.

## Brain Regions

The following brain regions are included:
- EB: Ellipsoid Body
- FB: Fan-shaped Body
- MB: Mushroom Body (Kenyon Cells)
- LH: Lateral Horn
- AL: Antennal Lobe

## Dependencies

- neuprint-python
- networkx
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- tqdm
- jupyter
- navis (for 3D visualization) 
