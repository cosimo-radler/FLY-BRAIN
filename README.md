# Fly Brain Connectome Analysis

This repository contains code for fetching, processing, and analyzing Drosophila brain connectome data from the neuPrint API.

## Current Project State

The current implementation focuses on fetching connectivity data for specific brain regions from the neuPrint API and saving them as GEXF files for visualization and further analysis.

## Directory Structure

```
FLY BRAIN/
├── data/                         # All graph data files
│   ├── raw/                      # └── Original downloads from API
│   └── processed/                # └── Processed GEXF files
│
├── notebooks/                    # Jupyter notebooks
│   └── 01_ingest.ipynb           # └── Download and save data as GEXF
│
├── src/                          # Python modules
│   ├── config.py                 # └── Configuration settings
│   ├── data_io.py                # └── Functions to fetch and save data
│   └── utils.py                  # └── Helper utilities
│
├── results/                      # Outputs from analyses
│   ├── figures/                  # └── Future plots
│   └── tables/                   # └── Future tables
│
├── convert_to_gexf.py            # Script to fetch and save all regions as GEXF
├── test_api_connection.py        # Script to test neuPrint API connection
├── requirements.txt              # pip-installable dependencies
└── README.md                     # This file
```

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Test API connection:
   ```
   python test_api_connection.py
   ```

## Usage

### Using the Script

To fetch data for all brain regions and save as GEXF files:

```
python convert_to_gexf.py
```

### Using the Notebook

Open and run `notebooks/01_ingest.ipynb` to:
1. Connect to the neuPrint API
2. Fetch neurons for each brain region
3. Fetch connectivity data
4. Create NetworkX graphs
5. Save graphs as GEXF files

## Brain Regions

The following brain regions are included:
- EB: Ellipsoid Body
- FB: Fan-shaped Body
- MB: Mushroom Body (Kenyon Cells)
- LH: Lateral Horn
- AL: Antennal Lobe

## Next Steps

1. Import the GEXF files into visualization tools like Gephi
2. Apply network analysis and sparsification techniques
3. Develop additional notebooks for deeper analysis

## Dependencies

- neuprint-python
- networkx
- pandas
- jupyter 