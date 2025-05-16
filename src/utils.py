"""
Utility functions for the Drosophila connectome analysis pipeline.

This module provides helper functions used throughout the project,
including seed management, logging setup, and common data operations.
"""

import random
import logging
import numpy as np
import os
from datetime import datetime
from pathlib import Path

def set_seed(seed):
    """
    Set random seeds for reproducibility across Python, NumPy, and other libraries.
    
    Parameters:
        seed (int): The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    # Add other libraries that need seed setting (e.g., TensorFlow, PyTorch) if used
    
    return seed

def setup_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for the project.
    
    Parameters:
        log_file (str, optional): Path to log file. If None, logs to console only.
        level (int, optional): Logging level. Default is INFO.
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger("fly_brain")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def generate_timestamp():
    """
    Generate a timestamp for file naming.
    
    Returns:
        str: Current datetime formatted as YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(directory):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Parameters:
        directory (str or Path): Directory path to ensure exists
    """
    Path(directory).mkdir(parents=True, exist_ok=True) 