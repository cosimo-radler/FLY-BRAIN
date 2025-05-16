"""
Configuration settings for the Drosophila connectome analysis pipeline.

This module centralizes all configuration parameters used across the project,
including API credentials, brain regions of interest, and analysis parameters.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DATA_RAW_DIR = os.path.join(DATA_DIR, "raw")
DATA_PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# neuPrint connection settings
# Replace with your own token from https://neuprint.janelia.org/
NEUPRINT_SERVER = "https://neuprint.janelia.org"  # Adding 'https://' prefix
NEUPRINT_DATASET = "hemibrain:v1.2.1"
# Use a valid token or set to None for anonymous access (with limitations)
NEUPRINT_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImNvc2ltb3JhZGxlckBnbWFpbC5jb20iLCJsZXZlbCI6Im5vYXV0aCIsImltYWdlLXVybCI6Imh0dHBzOi8vbGgzLmdvb2dsZXVzZXJjb250ZW50LmNvbS9hL0FDZzhvY0x4RlE0SFREbTNDNE1BUXJGbjU4RFFxWG1QaURSNFhGMzR3U0ZaVldKejJ2UVRwdz1zOTYtYz9zej01MD9zej01MCIsImV4cCI6MTkyNjc3NjU2NX0.7-Y0avYqISHioWvWHckuEZXoB3iYgBDtYg5aV9rnMRc"

# Brain regions of interest
BRAIN_REGIONS = {
    "EB": "Ellipsoid Body",
    "FB": "Fan-shaped Body",
    "MB": "Mushroom Body",
    "LH": "Lateral Horn",
    "AL": "Antennal Lobe"
}

# Analysis parameters
CONNECTIVITY_THRESHOLD = 3  # Minimum synaptic weight to include
RANDOM_SEED = 42  # For reproducibility 