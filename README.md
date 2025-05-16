# Fly Brain Connectome Analysis

This repository contains code for fetching, processing, and analyzing Drosophila brain connectome data from the neuPrint API.

## Current Project State

The current implementation includes:
- Fetching connectivity data for specific brain regions from the neuPrint API
- Cleaning networks by removing self-loops and extracting largest connected components
- Saving both raw and cleaned networks as GEXF files
- Computing various network metrics and statistics

## Directory Structure

```
FLY BRAIN/
├── data/                         # All graph data files
│   ├── raw/                      # └── Original downloads from API
│   └── processed/                # └── Cleaned networks (self-loops removed, LCC extracted)
│
├── notebooks/                    # Scripts and notebooks for data processing
│   ├── 01_fetch_all_regions.py   # └── Download data for all regions from API
│   ├── 02_clean_data.py          # └── Clean networks and extract largest components
│   └── archive/                  # └── Archive of older scripts
│
├── src/                          # Python modules—core logic
│   ├── config.py                 # └── Configuration settings and parameters
│   ├── data_io.py                # └── Functions to fetch, cache, and read/write graph files
│   ├── metrics.py                # └── Compute network statistics (degree, clustering, path-length)
│   ├── neuprint_client.py        # └── Interface to the neuPrint API
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

## Brain Regions

The following brain regions are included:
- EB: Ellipsoid Body
- FB: Fan-shaped Body
- MB: Mushroom Body (Kenyon Cells)
- LH: Lateral Horn
- AL: Antennal Lobe

## Next Steps

1. Calculate network metrics and statistics
2. Apply sparsification and null model techniques
3. Compare structural properties across brain regions
4. Perform percolation simulations
5. Import the GEXF files into visualization tools like Gephi

## Dependencies

- neuprint-python
- networkx
- pandas
- numpy
- matplotlib
- jupyter 