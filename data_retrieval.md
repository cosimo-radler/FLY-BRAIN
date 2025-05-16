# Drosophila Connectome Data Retrieval

This document explains how to retrieve and work with connectome data from the neuPrint database for the Drosophila fly brain.

## Data Source

All data is retrieved from the Janelia FlyEM Hemibrain database v1.2.1 using the official `neuprint-python` client.

- Database URL: https://neuprint.janelia.org
- Dataset: hemibrain:v1.2.1
- Documentation: https://connectome-neuprint.github.io/neuprint-python/docs/

## Brain Regions

The project currently works with the following brain regions:

| Code | Name | Description |
|------|------|-------------|
| EB | Ellipsoid Body | Part of the central complex, involved in orientation and navigation |
| FB | Fan-shaped Body | Part of the central complex, involved in higher-order behavior control |
| MB | Mushroom Body | Involved in learning and memory |
| LH | Lateral Horn | Processes innate olfactory information |
| AL | Antennal Lobe | Primary olfactory processing center |

## Unified Data Retrieval Approach

The `src/neuprint_client.py` module provides a unified interface to fetch data from neuPrint using the official client library. This is the **only** method that should be used for data retrieval to ensure consistency.

### Key Components

1. **NeuPrintInterface class**: Handles all interactions with the neuPrint API
2. **Region mapping**: Correctly maps our simple region codes to actual neuPrint ROI names
3. **Data saving**: Standardized approach to save neuron metadata, connectivity, and network graphs

### How to Use

To fetch data for all regions, simply run:

```bash
python notebooks/fetch_all_regions.py
```

To force re-fetching of all data (even if it exists already):

```bash
python notebooks/fetch_all_regions.py --force
```

### Output Files

For each region, the following files are generated:

1. **Neuron metadata**: `data/raw/{region}_neurons.json`
2. **Connectivity data**: `data/raw/{region}_connectivity.csv`
3. **Network graph**: `data/processed/{region}_network.gexf`

Additionally, summary reports are saved in:

- `results/tables/fetch_summary_{timestamp}.csv`
- `results/tables/fetch_report_{timestamp}.json`

## Data Access in Code

To access the data in your code:

```python
from src.neuprint_client import NeuPrintInterface

# Create the interface
npi = NeuPrintInterface()

# Fetch data for a specific region
neurons, connectivity, network = npi.process_region("EB")

# Now you have:
# - neurons: list of dictionaries with neuron metadata
# - connectivity: DataFrame with source, target, weight columns
# - network: NetworkX DiGraph object
```

## Network Statistics

As of the latest fetch, here are the statistics for each region:

| Region | Neurons | Connections | Avg. Degree | Density |
|--------|---------|-------------|------------|---------|
| EB | 1,062 | 43,855 | 82.59 | 0.039 |
| FB | 6,019 | 172,672 | 57.38 | 0.005 |
| MB | 603 | 13,191 | 43.75 | 0.036 |
| LH | 4,499 | 100,567 | 44.71 | 0.005 |
| AL | 1,099 | 10,945 | 19.92 | 0.009 |

## Troubleshooting

If you encounter authentication issues:

1. Obtain a new token from https://neuprint.janelia.org
2. Update the token in `src/config.py`

For other API issues, check the neuPrint API status at https://neuprint.janelia.org/dashboard 