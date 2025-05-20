"""
Network Metrics Module

This module provides functions to compute various network metrics for graph analysis.
Metrics are organized into categories:
- Degree-based metrics
- Clustering & motifs
- Path-length & connectivity
- Degree-degree & community structure
- Centrality & core-periphery
- Spectral metrics
- Percolation-specific metrics

Each function takes a NetworkX graph object and returns the computed metric(s).
"""

import numpy as np
import networkx as nx
import scipy.sparse.linalg
import collections
from collections import Counter
import logging

try:
    import community as louvain
except ImportError:
    print("Warning: python-louvain package not installed. Community detection functions will not work.")
    print("Install with: pip install python-louvain")

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# 1. DEGREE-BASED METRICS
# -------------------------------------------------------------------------

def get_degree_sequence(G):
    """Get the degree sequence of the graph as a numpy array."""
    return np.array([d for _, d in G.degree()])

def compute_degree_distribution(G):
    """Compute the empirical degree distribution (p_k)."""
    degs = get_degree_sequence(G)
    values, counts = np.unique(degs, return_counts=True)
    return dict(zip(values, counts / len(degs)))

def compute_degree_moments(G):
    """Compute the first three moments of degree distribution."""
    degs = get_degree_sequence(G)
    k1 = degs.mean()
    k2 = (degs**2).mean()
    k3 = (degs**3).mean()
    return {"avg_degree": k1, "second_moment": k2, "third_moment": k3}

def compute_degree_statistics(G):
    """Compute min, max, median, and quartiles of degree distribution."""
    degs = get_degree_sequence(G)
    return {
        "min_degree": np.min(degs),
        "q1_degree": np.percentile(degs, 25),
        "median_degree": np.median(degs),
        "q3_degree": np.percentile(degs, 75),
        "max_degree": np.max(degs)
    }


# -------------------------------------------------------------------------
# 2. CLUSTERING & MOTIFS
# -------------------------------------------------------------------------

def compute_clustering_metrics(G):
    """Compute global and average local clustering coefficients."""
    return {
        "transitivity": nx.transitivity(G),
        "avg_clustering": nx.average_clustering(G)
    }

def compute_triangle_metrics(G):
    """Compute triangle-related metrics."""
    tri_dict = nx.triangles(G)
    per_node = np.array(list(tri_dict.values()))
    
    return {
        "total_triangles": sum(tri_dict.values()) / 3,
        "avg_triangles_per_node": per_node.mean(),
        "max_triangles": np.max(per_node)
    }

def compute_higher_order_motifs(G):
    """Compute higher-order motif statistics (if supported)."""
    # This is a placeholder as motif computation may require specialized libraries
    # Implementation would depend on available libraries and needs
    return {"motifs_implemented": False}


# -------------------------------------------------------------------------
# 3. PATH-LENGTH & CONNECTIVITY
# -------------------------------------------------------------------------

def compute_path_length_metrics(G):
    """Compute average shortest path length, diameter, and radius.
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
        f"{prefix}diameter": nx.diameter(G_lcc),
        f"{prefix}radius": nx.radius(G_lcc)
    }

def compute_efficiency_metrics(G):
    """Compute global and local efficiency metrics."""
    global_eff = nx.global_efficiency(G)
    
    # Compute local efficiency
    local_effs = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 1:  # Need at least 2 neighbors for a subgraph
            Gi = G.subgraph(neighbors).copy()
            if nx.is_connected(Gi) and len(Gi) > 1:
                local_effs.append(nx.global_efficiency(Gi))
            else:
                local_effs.append(0)  # Disconnected subgraph or singleton has 0 efficiency
        else:
            local_effs.append(0)  # Node with 0 or 1 neighbor
    
    return {
        "global_efficiency": global_eff,
        "local_efficiency": np.mean(local_effs) if local_effs else 0
    }


# -------------------------------------------------------------------------
# 4. DEGREE-DEGREE & COMMUNITY STRUCTURE
# -------------------------------------------------------------------------

def compute_assortativity(G):
    """Compute degree assortativity coefficient."""
    return {"assortativity": nx.degree_pearson_correlation_coefficient(G)}

def compute_community_metrics(G):
    """Compute community structure metrics using Louvain algorithm."""
    try:
        part = louvain.best_partition(G)
        Q = louvain.modularity(part, G)
        n_comms = len(set(part.values()))
        sizes = list(Counter(part.values()).values())
        
        return {
            "modularity": Q,
            "num_communities": n_comms,
            "largest_community_size": max(sizes),
            "avg_community_size": np.mean(sizes)
        }
    except:
        return {
            "modularity": None,
            "community_metrics_error": "Louvain algorithm not available or failed"
        }


# -------------------------------------------------------------------------
# 5. CENTRALITY & CORE-PERIPHERY
# -------------------------------------------------------------------------

def compute_betweenness_metrics(G):
    """Compute betweenness centrality and centralization."""
    bc = nx.betweenness_centrality(G)
    max_bc = max(bc.values())
    N = len(G)
    
    if N > 2:
        C_B = sum(max_bc - v for v in bc.values()) / ((N-1) * (N-2) / 2)
    else:
        C_B = 0  # Cannot compute centralization for N <= 2
    
    return {
        "avg_betweenness": np.mean(list(bc.values())),
        "max_betweenness": max_bc,
        "betweenness_centralization": C_B
    }

def compute_degree_centrality_metrics(G):
    """Compute degree centrality metrics."""
    dc = nx.degree_centrality(G)
    return {
        "avg_degree_centrality": np.mean(list(dc.values())),
        "max_degree_centrality": max(dc.values())
    }

def compute_core_decomposition(G):
    """Compute k-core decomposition metrics."""
    cores = nx.core_number(G)
    core_values = list(cores.values())
    
    return {
        "max_core": max(core_values),
        "avg_core": np.mean(core_values)
    }

def compute_rich_club_metrics(G):
    """Compute rich-club coefficient metrics."""
    try:
        phi = nx.rich_club_coefficient(G, normalized=False)
        if phi:  # Can be empty if graph is not suitable
            return {
                "rich_club_coefficient": phi,
                "max_phi": max(phi.values()) if phi else None,
                "phi_values": phi
            }
    except:
        pass
    
    return {"rich_club_coefficient": None}


# -------------------------------------------------------------------------
# 6. SPECTRAL METRICS
# -------------------------------------------------------------------------

def compute_laplacian_spectrum(G, k=10):
    """Compute first k eigenvalues of the Laplacian matrix."""
    try:
        L = nx.laplacian_matrix(G)  # SciPy sparse matrix
        # Convert to CSR format with float64 data type for eigsh
        L = L.astype(np.float64).tocsr()
        eigs = scipy.sparse.linalg.eigsh(L, k=min(k, len(G)-1), which='SM')[0]
        sorted_eigs = sorted(eigs)
        
        # Compute spectral gap (λ₂) and gap between λ₂ and λ₃
        lambda2 = sorted_eigs[1] if len(sorted_eigs) > 1 else None
        gap = sorted_eigs[2] - lambda2 if len(sorted_eigs) > 2 else None
        
        return {
            "laplacian_eigenvalues": sorted_eigs,
            "algebraic_connectivity": lambda2,  # λ₂
            "spectral_gap": gap
        }
    except Exception as e:
        return {
            "laplacian_eigenvalues": None,
            "spectral_metrics_error": f"Eigenvalue computation failed: {str(e)}"
        }

def compute_spectral_density(G, bins=10):
    """Compute histogram of eigenvalues (spectral density)."""
    try:
        L = nx.laplacian_matrix(G)
        # Convert to CSR format with float64 data type for eigsh
        L = L.astype(np.float64).tocsr()
        eigs = scipy.sparse.linalg.eigsh(L, k=min(len(G)-1, 100), which='LM')[0]
        hist, bin_edges = np.histogram(eigs, bins=bins)
        
        return {
            "spectral_density_hist": hist,
            "spectral_density_bins": bin_edges
        }
    except Exception as e:
        return {"spectral_density_error": f"Spectral density computation failed: {str(e)}"}


# -------------------------------------------------------------------------
# 7. PERCOLATION-SPECIFIC METRICS
# -------------------------------------------------------------------------

def compute_analytic_threshold(G):
    """Compute the analytic percolation threshold for a configuration model."""
    degs = get_degree_sequence(G)
    k1 = degs.mean()
    k2 = (degs**2).mean()
    
    if k2 > k1:
        p_c = k1 / (k2 - k1)
        f_c = 1 - p_c
    else:
        p_c = None
        f_c = None
    
    return {
        "analytic_pc": p_c,
        "analytic_fc": f_c
    }

def compute_percolation_metrics(S_values, f_values):
    """
    Compute metrics from percolation simulation results.
    
    Parameters:
    - S_values: List of relative giant component sizes
    - f_values: List of corresponding removal fractions
    
    Returns dictionary of percolation metrics.
    """
    # Find approximate critical threshold (where S drops below 0.5)
    fc_index = next((i for i, s in enumerate(S_values) if s < 0.5), len(S_values) - 1)
    fc_estimate = f_values[fc_index] if fc_index < len(f_values) else None
    
    # Compute area under curve (robustness measure)
    auc = np.trapz(S_values, f_values)
    
    return {
        "estimated_fc": fc_estimate,
        "percolation_robustness": auc
    }


# -------------------------------------------------------------------------
# MAIN METRIC COMPUTATION FUNCTION
# -------------------------------------------------------------------------

def compute_all_metrics(G, compute_spectral=True, compute_centrality=True):
    """
    Compute all graph metrics for the given network.
    
    Parameters:
    - G: NetworkX graph object
    - compute_spectral: Whether to compute spectral metrics (can be slow for large graphs)
    - compute_centrality: Whether to compute centrality metrics (can be slow for large graphs)
    
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
    
    # 1. Degree-based metrics (use original graph for directed in/out degrees)
    if is_directed:
        # For directed graphs, compute in/out degree metrics separately
        in_degs = np.array([d for _, d in G.in_degree()])
        out_degs = np.array([d for _, d in G.out_degree()])
        
        metrics["avg_in_degree"] = in_degs.mean()
        metrics["avg_out_degree"] = out_degs.mean()
        metrics["second_moment_in"] = (in_degs**2).mean()
        metrics["second_moment_out"] = (out_degs**2).mean()
        
        # For undirected metrics, use the undirected version
        metrics.update(compute_degree_moments(G_undirected))
        metrics.update(compute_degree_statistics(G_undirected))
    else:
        # For undirected graphs, compute standard degree metrics
        metrics.update(compute_degree_moments(G))
        metrics.update(compute_degree_statistics(G))
    
    # 2. Clustering metrics (undirected)
    metrics.update(compute_clustering_metrics(G_undirected))
    metrics.update(compute_triangle_metrics(G_undirected))
    
    # 3. Path-length metrics (undirected)
    try:
        metrics.update(compute_path_length_metrics(G_undirected))
    except Exception as e:
        metrics["path_metrics_error"] = f"Path computation failed (graph may be disconnected): {str(e)}"
    
    metrics.update(compute_efficiency_metrics(G_undirected))
    
    # 4. Community structure (undirected)
    metrics.update(compute_assortativity(G_undirected))
    metrics.update(compute_community_metrics(G_undirected))
    
    # 5. Centrality metrics (optional - can be slow for large graphs)
    if compute_centrality:
        # Use the original graph for centrality (directed or undirected)
        metrics.update(compute_degree_centrality_metrics(G))
        
        # Use undirected for other centrality measures
        metrics.update(compute_core_decomposition(G_undirected))
        
        if len(G) < 10000:  # Skip for very large graphs
            metrics.update(compute_betweenness_metrics(G_undirected))
            metrics.update(compute_rich_club_metrics(G_undirected))
    
    # 6. Spectral metrics (optional - can be slow for large graphs)
    if compute_spectral and len(G) < 20000:
        metrics.update(compute_laplacian_spectrum(G_undirected))
        if len(G) < 5000:  # Skip for larger graphs
            metrics.update(compute_spectral_density(G_undirected))
    
    # 7. Percolation threshold estimate (undirected)
    metrics.update(compute_analytic_threshold(G_undirected))
    
    return metrics 