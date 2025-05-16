"""
Configuration settings for the Drosophila connectome analysis pipeline.

This module centralizes all configuration parameters used across the project,
including API credentials, brain regions of interest, and analysis parameters.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
RESULTS_FIGURES_DIR = ROOT_DIR / "results" / "figures"
RESULTS_TABLES_DIR = ROOT_DIR / "results" / "tables"

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, RESULTS_FIGURES_DIR, RESULTS_TABLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# neuPrint API settings
NEUPRINT_SERVER = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "hemibrain:v1.2.1"
NEUPRINT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImNvc2ltb3JhZGxlckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0x4RlE0SFREbTNDNE1BUXJGbjU4RFFxWG1QaURSNFhGMzR3U0ZaVldKejJ2UVRwdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTkyNjc3NjU2NX0.7-Y0avYqISHioWvWHckuEZXoB3iYgBDtYg5aV9rnMRc"

# Brain regions of interest
BRAIN_REGIONS = {
    "EB": "Ellipsoid Body",
    "FB": "Fan-shaped Body",
    "MB": "Mushroom Body",
    "LH": "Lateral Horn",
    "AL": "Antennal Lobe"
}

# Random seed for reproducibility
DEFAULT_SEED = 42

# Analysis parameters
CONNECTIVITY_THRESHOLD = 3  # Minimum number of synapses to consider a connection 