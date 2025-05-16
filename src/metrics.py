"""
Metrics module for the Drosophila connectome analysis pipeline.

This module provides functions for computing network statistics and metrics
on the connectome graphs, including degree distributions, clustering coefficients,
and path length analyses.
"""

import logging
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm

from . import config

logger = logging.getLogger("fly_brain")

def compute_basic_metrics(G):
    """
    Compute basic network metrics.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        dict: Dictionary of basic metrics
    """
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_directed': nx.is_directed(G),
        'is_connected': nx.is_weakly_connected(G) if nx.is_directed(G) else nx.is_connected(G)
    }
    
    # Compute average degree
    if nx.is_directed(G):
        metrics['avg_in_degree'] = np.mean([d for _, d in G.in_degree()])
        metrics['avg_out_degree'] = np.mean([d for _, d in G.out_degree()])
    else:
        metrics['avg_degree'] = np.mean([d for _, d in G.degree()])
    
    logger.info(f"Computed basic metrics for graph with {metrics['nodes']} nodes and {metrics['edges']} edges")
    return metrics

def compute_degree_metrics(G):
    """
    Compute degree-related metrics.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        dict: Dictionary of degree metrics
    """
    metrics = {}
    
    if nx.is_directed(G):
        # In-degree
        in_degrees = [d for _, d in G.in_degree()]
        metrics['in_degree_min'] = min(in_degrees)
        metrics['in_degree_max'] = max(in_degrees)
        metrics['in_degree_mean'] = np.mean(in_degrees)
        metrics['in_degree_median'] = np.median(in_degrees)
        metrics['in_degree_std'] = np.std(in_degrees)
        
        # Out-degree
        out_degrees = [d for _, d in G.out_degree()]
        metrics['out_degree_min'] = min(out_degrees)
        metrics['out_degree_max'] = max(out_degrees)
        metrics['out_degree_mean'] = np.mean(out_degrees)
        metrics['out_degree_median'] = np.median(out_degrees)
        metrics['out_degree_std'] = np.std(out_degrees)
        
        # Assortativity
        metrics['in_degree_assortativity'] = nx.degree_assortativity_coefficient(G, x='in', y='in')
        metrics['out_degree_assortativity'] = nx.degree_assortativity_coefficient(G, x='out', y='out')
        metrics['in_out_degree_assortativity'] = nx.degree_assortativity_coefficient(G, x='in', y='out')
    else:
        # Undirected graph
        degrees = [d for _, d in G.degree()]
        metrics['degree_min'] = min(degrees)
        metrics['degree_max'] = max(degrees)
        metrics['degree_mean'] = np.mean(degrees)
        metrics['degree_median'] = np.median(degrees)
        metrics['degree_std'] = np.std(degrees)
        
        # Assortativity
        metrics['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    
    logger.info("Computed degree metrics")
    return metrics

def compute_clustering_metrics(G, use_weights=False):
    """
    Compute clustering-related metrics.
    
    Parameters:
        G (nx.Graph): Input graph
        use_weights (bool): Whether to use edge weights in the calculation
    
    Returns:
        dict: Dictionary of clustering metrics
    """
    metrics = {}
    
    # Convert directed graph to undirected for clustering calculations if needed
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
        logger.info("Converting directed graph to undirected for clustering calculations")
    else:
        G_undirected = G
    
    # Global clustering coefficient (transitivity)
    metrics['transitivity'] = nx.transitivity(G_undirected)
    
    # Average local clustering coefficient
    if use_weights:
        clustering = nx.clustering(G_undirected, weight='weight')
    else:
        clustering = nx.clustering(G_undirected)
    
    clustering_values = list(clustering.values())
    metrics['avg_clustering'] = np.mean(clustering_values)
    metrics['clustering_std'] = np.std(clustering_values)
    
    logger.info("Computed clustering metrics")
    return metrics

def compute_path_metrics(G, max_paths=1000):
    """
    Compute path-related metrics.
    
    Parameters:
        G (nx.Graph): Input graph
        max_paths (int): Maximum number of node pairs to sample for path calculations
    
    Returns:
        dict: Dictionary of path metrics
    """
    metrics = {}
    
    # Check if the graph is connected or weakly connected
    is_connected = nx.is_weakly_connected(G) if nx.is_directed(G) else nx.is_connected(G)
    if not is_connected:
        logger.warning("Graph is not connected, path metrics will be computed on the largest component")
        if nx.is_directed(G):
            largest_cc = max(nx.weakly_connected_components(G), key=len)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Diameter (longest shortest path)
    try:
        # This can be slow for large graphs, so we use an approximation
        metrics['diameter'] = nx.diameter(G)
    except nx.NetworkXError:
        logger.warning("Graph is not strongly connected, diameter computation skipped")
        metrics['diameter'] = None
    
    # Average shortest path length
    if G.number_of_nodes() > 100:
        # For large graphs, sample node pairs
        nodes = list(G.nodes())
        sampled_nodes = np.random.choice(nodes, min(len(nodes), max_paths), replace=False)
        
        path_lengths = []
        for i, source in enumerate(tqdm(sampled_nodes, desc="Computing path lengths")):
            for target in sampled_nodes[i+1:]:
                try:
                    path_lengths.append(nx.shortest_path_length(G, source, target))
                except nx.NetworkXNoPath:
                    # No path between these nodes
                    pass
        
        if path_lengths:
            metrics['avg_path_length'] = np.mean(path_lengths)
            metrics['path_length_std'] = np.std(path_lengths)
        else:
            metrics['avg_path_length'] = None
            metrics['path_length_std'] = None
    else:
        # For small graphs, compute all-pairs shortest paths
        try:
            avg_path_length = nx.average_shortest_path_length(G)
            metrics['avg_path_length'] = avg_path_length
            
            # Compute standard deviation
            all_paths = dict(nx.all_pairs_shortest_path_length(G))
            path_lengths = [all_paths[u][v] for u in all_paths for v in all_paths[u] if u != v]
            metrics['path_length_std'] = np.std(path_lengths)
        except nx.NetworkXError:
            logger.warning("Graph is not strongly connected, average path length computation skipped")
            metrics['avg_path_length'] = None
            metrics['path_length_std'] = None
    
    logger.info("Computed path metrics")
    return metrics

def compute_centrality_metrics(G, max_nodes=1000):
    """
    Compute centrality metrics.
    
    Parameters:
        G (nx.Graph): Input graph
        max_nodes (int): Maximum number of nodes to consider for betweenness centrality
    
    Returns:
        dict: Dictionary of centrality metrics
    """
    metrics = {}
    
    # Degree centrality
    if nx.is_directed(G):
        in_degree_centrality = nx.in_degree_centrality(G)
        out_degree_centrality = nx.out_degree_centrality(G)
        
        metrics['in_degree_centrality_mean'] = np.mean(list(in_degree_centrality.values()))
        metrics['in_degree_centrality_std'] = np.std(list(in_degree_centrality.values()))
        metrics['out_degree_centrality_mean'] = np.mean(list(out_degree_centrality.values()))
        metrics['out_degree_centrality_std'] = np.std(list(out_degree_centrality.values()))
    else:
        degree_centrality = nx.degree_centrality(G)
        metrics['degree_centrality_mean'] = np.mean(list(degree_centrality.values()))
        metrics['degree_centrality_std'] = np.std(list(degree_centrality.values()))
    
    # Betweenness centrality (can be slow for large graphs, so we use approximation)
    if G.number_of_nodes() > max_nodes:
        logger.info(f"Graph has {G.number_of_nodes()} nodes, computing approximate betweenness centrality")
        k = min(max_nodes, G.number_of_nodes())
        betweenness_centrality = nx.betweenness_centrality(G, k=k, normalized=True)
    else:
        betweenness_centrality = nx.betweenness_centrality(G, normalized=True)
    
    metrics['betweenness_centrality_mean'] = np.mean(list(betweenness_centrality.values()))
    metrics['betweenness_centrality_std'] = np.std(list(betweenness_centrality.values()))
    
    # Closeness centrality
    if G.number_of_nodes() <= max_nodes:
        try:
            closeness_centrality = nx.closeness_centrality(G)
            metrics['closeness_centrality_mean'] = np.mean(list(closeness_centrality.values()))
            metrics['closeness_centrality_std'] = np.std(list(closeness_centrality.values()))
        except:
            logger.warning("Failed to compute closeness centrality")
            metrics['closeness_centrality_mean'] = None
            metrics['closeness_centrality_std'] = None
    else:
        logger.info(f"Graph has {G.number_of_nodes()} nodes, skipping closeness centrality")
        metrics['closeness_centrality_mean'] = None
        metrics['closeness_centrality_std'] = None
    
    logger.info("Computed centrality metrics")
    return metrics

def compute_rich_club_coefficient(G, normalized=True):
    """
    Compute rich club coefficient for a range of degrees.
    
    Parameters:
        G (nx.Graph): Input graph
        normalized (bool): Whether to normalize the coefficient
    
    Returns:
        dict: Dictionary mapping degrees to rich club coefficients
    """
    if nx.is_directed(G):
        logger.warning("Rich club coefficient is defined for undirected graphs, converting to undirected")
        G = G.to_undirected()
    
    try:
        # Get the range of degrees to compute the coefficient for
        degrees = sorted(set([d for _, d in G.degree()]))
        rich_club = {}
        
        # Compute the coefficient for each degree
        for k in tqdm(degrees, desc="Computing rich club coefficients"):
            try:
                if normalized:
                    # This computes the normalized coefficient by generating random graphs
                    # It can be very slow for large graphs
                    rich_club[k] = nx.rich_club_coefficient(G, k, normalized=True)
                else:
                    rich_club[k] = nx.rich_club_coefficient(G, k, normalized=False)
            except Exception as e:
                logger.warning(f"Failed to compute rich club coefficient for degree {k}: {str(e)}")
                rich_club[k] = None
        
        logger.info(f"Computed rich club coefficients for {len(degrees)} degrees")
        return rich_club
    except Exception as e:
        logger.error(f"Failed to compute rich club coefficients: {str(e)}")
        return {}

def compute_small_world_metrics(G, num_random=10):
    """
    Compute small-world metrics by comparing to random graphs.
    
    Parameters:
        G (nx.Graph): Input graph
        num_random (int): Number of random graphs to generate for comparison
    
    Returns:
        dict: Dictionary of small-world metrics
    """
    metrics = {}
    
    # Convert to undirected if needed
    if nx.is_directed(G):
        G_undirected = G.to_undirected()
        logger.info("Converting directed graph to undirected for small-world metrics")
    else:
        G_undirected = G
    
    # Extract largest connected component
    if not nx.is_connected(G_undirected):
        largest_cc = max(nx.connected_components(G_undirected), key=len)
        G_undirected = G_undirected.subgraph(largest_cc).copy()
        logger.info(f"Using largest connected component with {G_undirected.number_of_nodes()} nodes")
    
    # Compute clustering coefficient of the actual graph
    C = nx.average_clustering(G_undirected)
    
    # Compute average shortest path length of the actual graph
    try:
        L = nx.average_shortest_path_length(G_undirected)
    except nx.NetworkXError:
        logger.warning("Graph is not connected, average path length computation skipped")
        return {'small_world_coefficient': None}
    
    # Generate random graphs with same number of nodes and edges
    random_C = 0
    random_L = 0
    
    n = G_undirected.number_of_nodes()
    m = G_undirected.number_of_edges()
    
    for i in tqdm(range(num_random), desc="Generating random graphs"):
        # Create Erdos-Renyi random graph
        random_G = nx.gnm_random_graph(n, m)
        
        # Make sure it's connected
        if not nx.is_connected(random_G):
            largest_cc = max(nx.connected_components(random_G), key=len)
            random_G = random_G.subgraph(largest_cc).copy()
        
        # Compute metrics
        random_C += nx.average_clustering(random_G)
        try:
            random_L += nx.average_shortest_path_length(random_G)
        except:
            # Skip this random graph if it's not connected
            logger.warning(f"Random graph {i+1}/{num_random} is not connected, skipping")
            num_random -= 1
            continue
    
    # Average over all random graphs
    if num_random > 0:
        random_C /= num_random
        random_L /= num_random
        
        # Calculate small-world coefficient
        metrics['clustering_ratio'] = C / random_C
        metrics['path_length_ratio'] = L / random_L
        metrics['small_world_coefficient'] = metrics['clustering_ratio'] / metrics['path_length_ratio']
        
        logger.info(f"Computed small-world coefficient: {metrics['small_world_coefficient']:.4f}")
    else:
        metrics['small_world_coefficient'] = None
        logger.warning("Failed to compute small-world metrics due to connectivity issues in random graphs")
    
    return metrics

def compute_all_metrics(G):
    """
    Compute all network metrics.
    
    Parameters:
        G (nx.Graph): Input graph
    
    Returns:
        dict: Dictionary of all metrics
    """
    all_metrics = {}
    
    # Basic metrics
    basic_metrics = compute_basic_metrics(G)
    all_metrics.update(basic_metrics)
    
    # Degree metrics
    degree_metrics = compute_degree_metrics(G)
    all_metrics.update(degree_metrics)
    
    # Clustering metrics
    clustering_metrics = compute_clustering_metrics(G)
    all_metrics.update(clustering_metrics)
    
    # Path metrics (may be slow for large graphs)
    if G.number_of_nodes() <= 1000:
        path_metrics = compute_path_metrics(G)
        all_metrics.update(path_metrics)
    else:
        logger.warning(f"Graph has {G.number_of_nodes()} nodes, skipping path metrics calculation")
    
    # Centrality metrics
    centrality_metrics = compute_centrality_metrics(G)
    all_metrics.update(centrality_metrics)
    
    # Small-world metrics (may be very slow for large graphs)
    if G.number_of_nodes() <= 500:
        small_world_metrics = compute_small_world_metrics(G)
        all_metrics.update(small_world_metrics)
    else:
        logger.warning(f"Graph has {G.number_of_nodes()} nodes, skipping small-world metrics calculation")
    
    logger.info("Computed all network metrics")
    return all_metrics

def save_metrics_to_csv(metrics, filename, output_dir=config.RESULTS_TABLES_DIR):
    """
    Save network metrics to a CSV file.
    
    Parameters:
        metrics (dict): Dictionary of metrics
        filename (str): Name of the output file
        output_dir (Path): Directory to save the file
    
    Returns:
        str: Path to the saved file
    """
    import os
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output file path
    output_file = os.path.join(output_dir, filename)
    
    # Convert metrics to DataFrame
    if isinstance(metrics, dict):
        # If metrics is a single dictionary, convert to DataFrame with one row
        df = pd.DataFrame([metrics])
    elif isinstance(metrics, list):
        # If metrics is a list of dictionaries, convert to DataFrame with multiple rows
        df = pd.DataFrame(metrics)
    else:
        raise ValueError("Metrics must be a dictionary or a list of dictionaries")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved metrics to {output_file}")
    
    return output_file 