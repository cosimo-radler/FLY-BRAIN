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
# 4. LAPLACIAN MATRIX METRICS
# -------------------------------------------------------------------------

def compute_laplacian_matrix(G, normalized=False):
    """
    Compute the Laplacian matrix of the graph.
    
    Parameters:
    - G: NetworkX graph object
    - normalized: bool, if True compute normalized Laplacian, else standard Laplacian
    
    Returns:
    - L: numpy array, the Laplacian matrix
    """
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=sorted(G.nodes()))
    else:
        L = nx.laplacian_matrix(G, nodelist=sorted(G.nodes()))
    
    return L.toarray()  # Convert sparse matrix to dense array

def compute_laplacian_eigenvalues(G, normalized=False):
    """
    Compute eigenvalues of the Laplacian matrix.
    
    Parameters:
    - G: NetworkX graph object
    - normalized: bool, if True compute normalized Laplacian eigenvalues
    
    Returns:
    - eigenvals: numpy array, sorted eigenvalues (ascending order)
    """
    L = compute_laplacian_matrix(G, normalized=normalized)
    eigenvals = np.linalg.eigvals(L)
    return np.sort(np.real(eigenvals))  # Sort and take real part

def compute_laplacian_metrics(G):
    """
    Compute various metrics related to the Laplacian matrix.
    
    Parameters:
    - G: NetworkX graph object
    
    Returns:
    - dict: Dictionary containing Laplacian-related metrics
    """
    metrics = {}
    
    # Only compute for connected graphs or largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(largest_cc).copy()
        prefix = "lcc_"
        logger.info(f"Graph not connected, computing Laplacian metrics on LCC with {len(G_lcc)} nodes")
    else:
        G_lcc = G
        prefix = ""
    
    # Standard Laplacian eigenvalues
    eigenvals = compute_laplacian_eigenvalues(G_lcc, normalized=False)
    metrics[f"{prefix}laplacian_eigenvals"] = eigenvals
    metrics[f"{prefix}laplacian_second_smallest"] = eigenvals[1] if len(eigenvals) > 1 else 0.0
    metrics[f"{prefix}laplacian_largest"] = eigenvals[-1]
    metrics[f"{prefix}algebraic_connectivity"] = eigenvals[1] if len(eigenvals) > 1 else 0.0
    
    # Normalized Laplacian eigenvalues
    eigenvals_norm = compute_laplacian_eigenvalues(G_lcc, normalized=True)
    metrics[f"{prefix}normalized_laplacian_eigenvals"] = eigenvals_norm
    metrics[f"{prefix}normalized_laplacian_largest"] = eigenvals_norm[-1]
    
    # Spectral gap (difference between two smallest eigenvalues)
    if len(eigenvals) > 1:
        metrics[f"{prefix}spectral_gap"] = eigenvals[1] - eigenvals[0]
    else:
        metrics[f"{prefix}spectral_gap"] = 0.0
    
    return metrics

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
    
    # 4. Laplacian matrix metrics
    metrics.update(compute_laplacian_metrics(G_undirected))
    
    return metrics 