"""
Network Metrics Module (Simplified)

This module provides functions to compute basic network metrics for graph analysis.
Focuses on the most common and computationally efficient metrics:
- Basic graph properties
- Degree-based metrics
- Clustering metrics
- Path-length metrics
"""

import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# 1. DEGREE-BASED METRICS
# -------------------------------------------------------------------------

def get_degree_sequence(G):
    """Get the degree sequence of the graph as a numpy array."""
    return np.array([d for _, d in G.degree()])

def compute_degree_metrics(G):
    """Compute basic degree-related metrics."""
    degs = get_degree_sequence(G)
    return {
        "avg_degree": degs.mean(),
        "median_degree": np.median(degs),
        "max_degree": np.max(degs)
    }

# -------------------------------------------------------------------------
# 2. CLUSTERING METRICS
# -------------------------------------------------------------------------

def compute_clustering_metrics(G):
    """Compute global and average local clustering coefficients."""
    return {
        "transitivity": nx.transitivity(G),
        "avg_clustering": nx.average_clustering(G)
    }

# -------------------------------------------------------------------------
# 3. PATH-LENGTH METRICS
# -------------------------------------------------------------------------

def compute_path_length_metrics(G):
    """Compute average shortest path length and diameter.
    Note: Only computes on the largest connected component if graph is not connected."""
    
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
        prefix = "lcc_"  # Prefix to indicate metrics are for LCC only
    else:
        G_lcc = G
        prefix = ""
    
    return {
        f"{prefix}avg_shortest_path": nx.average_shortest_path_length(G_lcc),
        f"{prefix}diameter": nx.diameter(G_lcc)
    }

def compute_efficiency(G):
    """Compute global efficiency metric."""
    return {"global_efficiency": nx.global_efficiency(G)}

# -------------------------------------------------------------------------
# MAIN METRIC COMPUTATION FUNCTION
# -------------------------------------------------------------------------

def compute_all_metrics(G):
    """
    Compute basic graph metrics for the given network.
    
    Parameters:
    - G: NetworkX graph object
    
    Returns a dictionary with all computed metrics.
    """
    metrics = {}
    
    # Store original graph type
    is_directed = G.is_directed()
    metrics["is_directed"] = is_directed
    
    # Convert to undirected for metrics that require it
    if is_directed:
        logger.info(f"Converting directed graph with {len(G)} nodes to undirected for metrics calculation")
        G_undirected = G.to_undirected()
    else:
        G_undirected = G
    
    # Basic graph info
    metrics["num_nodes"] = len(G)
    metrics["num_edges"] = G.number_of_edges()
    metrics["density"] = nx.density(G)
    
    # Check connectivity on undirected graph
    metrics["is_connected"] = nx.is_connected(G_undirected)
    
    # 1. Degree-based metrics
    metrics.update(compute_degree_metrics(G_undirected))
    
    # 2. Clustering metrics
    metrics.update(compute_clustering_metrics(G_undirected))
    
    # 3. Path-length metrics
    try:
        metrics.update(compute_path_length_metrics(G_undirected))
    except Exception as e:
        metrics["path_metrics_error"] = f"Path computation failed (graph may be disconnected): {str(e)}"
    
    metrics.update(compute_efficiency(G_undirected))
    
    return metrics 