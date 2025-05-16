"""
Configuration model generators for brain networks.

This module provides functions to create null models that preserve specific
structural aspects of the original networks while randomizing others.
"""

import networkx as nx
import random
import logging

def build_plain_cm(G: nx.Graph, seed: int = None) -> nx.Graph:
    """
    Builds an unscaled configuration model preserving the exact degree sequence.
    
    Args:
        G: Original network graph
        seed: Random seed for reproducibility
        
    Returns:
        A new graph with the same degree sequence but randomized connections
    """
    # Extract original degree sequence
    deg_seq = [d for _, d in G.degree()]
    
    # Build the multigraph configuration model
    G_cm = nx.configuration_model(deg_seq, seed=seed)
    
    # Convert to simple graph: collapse parallel edges & remove self-loops
    G_simple = nx.Graph(G_cm)  
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    
    # Verify key properties
    logging.info(f"Original edges: {G.number_of_edges()}, CM edges: {G_simple.number_of_edges()}")
    
    return G_simple


def build_scaled_cm(G: nx.Graph, n_target: int, seed: int = None) -> nx.Graph:
    """
    Builds a scaled configuration model with n_target nodes.
    
    Samples n_target degrees from the original network's degree distribution,
    ensures even sum for valid graph creation, then builds the configuration model.
    
    Args:
        G: Original network graph
        n_target: Target number of nodes for the scaled model
        seed: Random seed for reproducibility
        
    Returns:
        A new graph with n_target nodes following the original degree distribution
        
    Raises:
        RuntimeError: If unable to ensure even sum of degrees after multiple attempts
    """
    # Extract original degree sequence
    orig_deg = [d for _, d in G.degree()]
    
    # Sample new degree sequence of length n_target
    rnd = random.Random(seed)
    sampled = [rnd.choice(orig_deg) for _ in range(n_target)]
    
    # Enforce even total stubs (required for valid graph construction)
    if sum(sampled) % 2 != 0:
        # Tweak the last draw until sum is even
        for attempt in range(10_000):
            sampled[-1] = rnd.choice(orig_deg)
            if sum(sampled) % 2 == 0:
                logging.debug(f"Found even degree sum after {attempt+1} attempts")
                break
        else:
            raise RuntimeError("Could not enforce even sum on sampled degrees")
    
    # Build the CM on the sampled sequence
    G_cm = nx.configuration_model(sampled, seed=seed)
    
    # Convert to simple graph
    G_simple = nx.Graph(G_cm)
    G_simple.remove_edges_from(nx.selfloop_edges(G_simple))
    
    # Verify key properties
    logging.info(f"Scaled CM: {G_simple.number_of_nodes()} nodes, {G_simple.number_of_edges()} edges")
    
    return G_simple


def ensure_connected(G: nx.Graph, original: nx.Graph = None, seed: int = None) -> nx.Graph:
    """
    Ensures the graph is connected by extracting the largest connected component
    and adding minimal edges to connect any remaining nodes.
    
    Args:
        G: Graph to ensure connectivity for
        original: Original graph to use as reference for edge addition (optional)
        seed: Random seed for reproducibility
        
    Returns:
        A connected version of the input graph
    """
    if nx.is_connected(G):
        return G
    
    # Extract largest connected component
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    
    # Start with the largest component
    G_connected = G.subgraph(largest_cc).copy()
    disconnected = [node for node in G.nodes() if node not in largest_cc]
    
    logging.info(f"Making graph connected: {len(disconnected)} disconnected nodes")
    
    # Add minimal edges to connect remaining components
    rnd = random.Random(seed)
    for node in disconnected:
        # Pick random node from connected component to link to
        target = rnd.choice(list(G_connected.nodes()))
        G_connected.add_node(node)
        G_connected.add_edge(node, target)
    
    return G_connected 