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

## Directory Structure

```
FLY BRAIN/
├── data/                         # All graph data files
│   ├── raw/                      # └── Original downloads from API
│   └── processed/                # └── Cleaned networks (self-loops removed, LCC extracted)
│       └── null_models/          #     └── Configuration models (unscaled and scaled)
│
├── notebooks/                    # Scripts and notebooks for data processing
│   ├── 01_fetch_all_regions.py   # └── Download data for all regions from API
│   ├── 02_clean_data.py          # └── Clean networks and extract largest components
│   ├── 03_configuration_models.py # └── Generate configuration models
│   ├── 04_percolate.py           # └── Perform bond percolation analysis
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
│   └── tables/                   # └── Summary statistics and reports
│
├── requirements.txt              # pip-installable dependencies
└── README.md                     # This file
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your neuPrint API credentials (required for data fetching)

## Usage

### Data Fetching

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

### Configuration Models

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

### Percolation Analysis

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

This will generate two key visualizations:
1. A comprehensive comparison plot showing all regions, attack strategies, and models side-by-side
2. A percolation by strategy plot comparing the original networks across attack strategies

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

This script uses the navis library to create 3D visualizations of neuron morphology, displaying the skeletal structure of neurons in the specified brain region.

## Brain Regions

The following brain regions are included:
- EB: Ellipsoid Body
- FB: Fan-shaped Body
- MB: Mushroom Body (Kenyon Cells)
- LH: Lateral Horn
- AL: Antennal Lobe

## Next Steps

1. Implement additional network metrics and statistics
2. Apply spectral sparsification techniques
3. Compare structural properties across brain regions
4. Implement additional percolation strategies
5. Import the GEXF files into visualization tools like Gephi

## Dependencies

- neuprint-python
- networkx
- pandas
- numpy
- matplotlib
- seaborn
- tqdm
- jupyter 