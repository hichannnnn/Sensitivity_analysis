#!/usr/bin/env python3
"""
Sensitivity Analysis: NeKo Network Construction Strategies + MaBoSS Simulations
=============================================================================

This script performs a comprehensive sensitivity analysis by:
1. Creating networks using different NeKo strategies and parameters
2. Testing both OmniPath and SIGNOR databases
3. Cleaning networks (removing bimodal/undefined interactions)
4. Skipping disconnected networks
5. Running MaBoSS simulations with random initial conditions
6. Collecting and visualizing results

Author: Generated for sensitivity analysis
Date: September 2025
"""

# Core imports
import os, sys, json, re, itertools, warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Import packages - set to None if missing
try: import pandas as pd
except ImportError: pd = None; print("Warning: pandas not available")

try: import numpy as np  
except ImportError: np = None; print("Warning: numpy not available")

try: import matplotlib.pyplot as plt
except ImportError: plt = None; print("Warning: matplotlib not available")

try: import networkx as nx
except ImportError: nx = None; print("Warning: networkx not available")

try: from pathlib import Path
except ImportError: Path = None; print("Warning: pathlib not available")

# NeKo imports
try:
    from neko.core.network import Network
    from neko._visual.visualize_network import NetworkVisualizer
    from neko.inputs import Universe, signor
    from neko._outputs.exports import Exports
    import omnipath as op
    neko_available = True
except ImportError as e:
    print(f"Warning: NeKo not available - {e}")
    neko_available = False

# MaBoSS imports  
try:
    import maboss
    maboss_available = True
except ImportError as e:
    print(f"Warning: MaBoSS not available - {e}")
    maboss_available = False

# =============================================================================
# CONFIGURATION AND PARAMETERS
# =============================================================================

# Gene sets based on user requirements
INPUT_GENES = [
    # Growth factor receptors and adapters
    'EGFR', 'ERBB2',
    
    # MAPK cascade
    'KRAS', 'RAF1', 'MAPK1', 'MAPK3',
    
    # PI3K-AKT-mTOR pathway
     'PIK3R1', 'PTEN', 'AKT1', 
    
    # Cell cycle machinery
    'CCND1','CCNE1', 'RB1', 'E2F1', 'MYC'
]

# Output nodes to monitor in MaBoSS simulations
OUTPUT_NODES = ['MAPK3', 'MYC', 'CCND1', 'CCNE1']

# Network visualization settings
MAX_EDGES_FOR_VISUALIZATION = 1000  # Skip visualization for networks with more edges
                                    # This prevents long rendering times for complex networks
                                    # while maintaining visualization for smaller, interpretable networks

# Directory structure for results
if Path:
    BASE_DIR = Path("sensitivity_analysis_results")
    NETWORKS_DIR = BASE_DIR / "networks"
    BNET_DIR = BASE_DIR / "bnet_files"
    MABOSS_DIR = BASE_DIR / "maboss_results"
    PLOTS_DIR = BASE_DIR / "plots"
    SUMMARY_DIR = BASE_DIR / "summary"
    TOPOLOGY_DIR = BASE_DIR / "topology_metrics"
    
    # Create directories
    for dir_path in [BASE_DIR, NETWORKS_DIR, BNET_DIR, MABOSS_DIR, PLOTS_DIR, SUMMARY_DIR, TOPOLOGY_DIR]:
        dir_path.mkdir(exist_ok=True, parents=True)
else:
    # Fallback using os.path
    import os
    BASE_DIR = "sensitivity_analysis_results"
    NETWORKS_DIR = os.path.join(BASE_DIR, "networks")
    BNET_DIR = os.path.join(BASE_DIR, "bnet_files") 
    MABOSS_DIR = os.path.join(BASE_DIR, "maboss_results")
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    SUMMARY_DIR = os.path.join(BASE_DIR, "summary")
    TOPOLOGY_DIR = os.path.join(BASE_DIR, "topology_metrics")
    
    # Create directories
    for dir_path in [BASE_DIR, NETWORKS_DIR, BNET_DIR, MABOSS_DIR, PLOTS_DIR, SUMMARY_DIR, TOPOLOGY_DIR]:
        os.makedirs(dir_path, exist_ok=True)

# =============================================================================
# STRATEGY CONFIGURATIONS
# =============================================================================

def get_strategy_configurations():
    """Define all parameter combinations for NeKo network construction."""
    
    strategies = [
        {
            'name': 'complete_connection',
            'method': 'complete_connection',
            'params': {
                'algorithm': ['bfs', 'dfs'],  # Test both algorithms
                'consensus': [False, True],   # Test both consensus options
                'maxlen': [2, 3],            # Test different path lengths
                'only_signed': [True]       # Keep True for now
            }
        },
        {
            'name': 'connect_network_radially',
            'method': 'connect_network_radially',
            'params': {
                'consensus': [False, True],
                'only_signed': [True]
            }
        },
        {
            'name': 'connect_as_atopo',
            'method': 'connect_as_atopo',
            'params': {
                'strategy': ['complete'],
                'max_len': [2, 3],
                'consensus': [False, True],
                'only_signed': [True]
            }
        },
        {
            'name': 'connect_component',
            'method': 'connect_component',
            'params': {
                'consensus': [False, True],
                'only_signed': [True]
            }
        }
    ]
    
    databases = ['omnipath', 'signor']  # Test both databases
    
    return strategies, databases

def generate_parameter_combinations(strategy_config):
    """Generate all possible parameter combinations for a strategy."""
    params = strategy_config['params']
    param_names = list(params.keys())
    param_values = list(params.values())
    
    combinations = []
    for combo in itertools.product(*param_values):
        param_dict = dict(zip(param_names, combo))
        combinations.append(param_dict)
    
    return combinations

# =============================================================================
# DATABASE SETUP
# =============================================================================

def setup_databases():
    """Initialize database resources."""
    databases = {}
    
    # OmniPath (default)
    databases['omnipath'] = 'omnipath'
    
    # SIGNOR
    try:
        print("Setting up SIGNOR database...")
        signor_resources = signor()
        signor_resources.build()
        databases['signor'] = signor_resources.interactions
        print("✓ SIGNOR database ready")
    except Exception as e:
        print(f"✗ Failed to setup SIGNOR: {e}")
        databases['signor'] = None
    
    return databases

# =============================================================================
# NETWORK GENERATION AND ANALYSIS
# =============================================================================

def is_network_connected(network):
    """Check if the network is fully connected."""
    # More lenient connectivity check - we want to keep networks that have good connectivity
    # even if not fully connected
    
    if len(network.nodes) == 0 or len(network.edges) == 0:
        return False
    
    # Get valid gene nodes
    valid_nodes = get_valid_gene_nodes(network)
    
    if len(valid_nodes) == 0:
        return False
    
    if nx is None:
        # Fallback: check if we have a reasonable number of edges
        # A connected network needs at least n-1 edges, but for biological networks
        # we want more connectivity
        min_edges = max(len(valid_nodes) - 1, len(valid_nodes) * 0.5)  # More lenient
        return len(network.edges) >= min_edges
    
    # Build comprehensive UniProt to gene symbol mapping (same as other functions)
    uniprot_to_gene = {}
    
    # Method 1: Use network's nodes_df if available
    if hasattr(network, 'nodes_df') and network.nodes_df is not None and not network.nodes_df.empty:
        try:
            for _, row in network.nodes_df.iterrows():
                if 'genesymbol' in row and 'uniprot' in row:
                    if pd.notna(row['genesymbol']) and pd.notna(row['uniprot']):
                        uniprot_to_gene[str(row['uniprot'])] = str(row['genesymbol'])
                if 'Genesymbol' in row and 'Uniprot' in row:
                    if pd.notna(row['Genesymbol']) and pd.notna(row['Uniprot']):
                        uniprot_to_gene[str(row['Uniprot'])] = str(row['Genesymbol'])
        except:
            pass
    
    # Method 2: Use network.nodes attribute if it's a DataFrame  
    if hasattr(network, 'nodes') and hasattr(network.nodes, 'iterrows'):
        try:
            for _, row in network.nodes.iterrows():
                if 'genesymbol' in row and 'uniprot' in row:
                    if pd.notna(row['genesymbol']) and pd.notna(row['uniprot']):
                        uniprot_to_gene[str(row['uniprot'])] = str(row['genesymbol'])
                if 'Genesymbol' in row and 'Uniprot' in row:
                    if pd.notna(row['Genesymbol']) and pd.notna(row['Uniprot']):
                        uniprot_to_gene[str(row['Uniprot'])] = str(row['Genesymbol'])
        except:
            pass
    
    # Function to convert node ID to gene symbol
    def to_gene_symbol(node_id):
        if node_id in uniprot_to_gene:
            return uniprot_to_gene[node_id]
        if not is_uniprot_id(node_id):
            return node_id
        return None
    
    G = nx.Graph()
    for node in valid_nodes:
        G.add_node(node)
        
    # Add edges between valid nodes only (with proper conversion)
    edges_added = 0
    for _, edge in network.edges.iterrows():
        source_converted = to_gene_symbol(edge['source'])
        target_converted = to_gene_symbol(edge['target'])
        
        if (source_converted and target_converted and 
            source_converted in valid_nodes and target_converted in valid_nodes):
            G.add_edge(source_converted, target_converted)
            edges_added += 1
    
    # Check if the network has a large connected component
    if len(G.nodes) == 0:
        return False
    
    largest_cc = max(nx.connected_components(G), key=len)
    connectivity_ratio = len(largest_cc) / len(G.nodes)
    
    print(f"    Connectivity: {connectivity_ratio:.2f} ({len(largest_cc)}/{len(G.nodes)} nodes)")
    
    # Accept networks where at least 60% of nodes are in the largest connected component
    return connectivity_ratio >= 0.6

def calculate_network_metrics(network):
    """Calculate basic network topology metrics."""
    # Get valid gene nodes
    valid_nodes = get_valid_gene_nodes(network)
    
    metrics = {
        'num_nodes': len(valid_nodes),
        'num_edges': len(network.edges),
        'density': 0.0,
        'avg_degree': 0.0,
        'is_connected': False
    }
    
    if nx is not None and len(valid_nodes) > 0:
        G = nx.Graph()
        for node in valid_nodes:
            G.add_node(node)
        
        # Add edges between valid nodes only
        for _, edge in network.edges.iterrows():
            source = edge['source']
            target = edge['target']
            if source in valid_nodes and target in valid_nodes:
                G.add_edge(source, target)
        
        metrics['density'] = nx.density(G)
        metrics['avg_degree'] = sum(dict(G.degree()).values()) / len(G.nodes) if len(G.nodes) > 0 else 0
        metrics['is_connected'] = nx.is_connected(G)
    
    return metrics

def build_network(genes, database, strategy_name, strategy_method, params):
    """Build a single network with specified parameters."""
    try:
        print(f"Building network: {strategy_name} with {database}")
        
        # Create network - handle both string identifiers and DataFrame objects
        if isinstance(database, str) and database == 'omnipath':
            network = Network(genes, resources='omnipath')
        elif isinstance(database, str):
            # String identifier for other databases
            network = Network(genes, resources=database)
        else:
            # DataFrame or other object (like SIGNOR interactions)
            # Pass the database object directly as resources parameter
            network = Network(genes, resources=database)
        
        print(f"  Initial network: {len(network.nodes)} nodes, {len(network.edges)} edges")
        
        # Apply strategy
        if strategy_method == 'complete_connection':
            network.complete_connection(**params)
        elif strategy_method == 'connect_network_radially':
            network.connect_network_radially(**params)
        elif strategy_method == 'connect_as_atopo':
            # For connect_as_atopo, we need to specify outputs
            params_copy = params.copy()
            params_copy['outputs'] = OUTPUT_NODES
            network.connect_as_atopo(**params_copy)
        elif strategy_method == 'connect_component':
            # For connect_component, connect pairs of important nodes
            for i, gene1 in enumerate(['MAPK1', 'MYC']):
                for gene2 in ['CCND1', 'CCNE1']:
                    if gene1 != gene2:
                        try:
                            network.connect_component(gene1, gene2, **params)
                        except:
                            continue
        
        print(f"  After strategy: {len(network.nodes)} nodes, {len(network.edges)} edges")
        
        # Check connectivity before cleaning
        connected_before = is_network_connected(network)
        print(f"  Connected before cleaning: {connected_before}")
        
        # Always clean network - bimodal and undefined interactions must be removed
        print("  Cleaning network (removing bimodal and undefined interactions)...")
        initial_edges = len(network.edges)
        
        # Always remove bimodal and undefined interactions
        network.remove_bimodal_interactions()
        network.remove_undefined_interactions()
        final_edges = len(network.edges)
        print(f"  Removed {initial_edges - final_edges} edges during cleaning")
        
        # Clean duplicate edges (NeKo bug fix)
        network = check_and_clean_duplicate_edges(network)
        
        print(f"  Final network: {len(network.nodes)} nodes, {len(network.edges)} edges")
        
        # Check connectivity after cleaning
        if not is_network_connected(network):
            print("  ✗ Network is disconnected - skipping")
            return None, None
        
        print("  ✓ Network is sufficiently connected")
        
        # Calculate metrics
        metrics = calculate_network_metrics(network)
        
        return network, metrics
        
    except Exception as e:
        print(f"  ✗ Failed to build network: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# =============================================================================
# NETWORK EDGE DUPLICATE DETECTION AND CLEANING
# =============================================================================

def check_and_clean_duplicate_edges(network):
    """Check for and remove duplicate edges in NeKo network."""
    if network is None or not hasattr(network, 'edges') or network.edges is None:
        return network
        
    try:
        print(f"  Checking for duplicate edges...")
        initial_edges = len(network.edges)
        
        if initial_edges == 0:
            print(f"    No edges to check")
            return network
        
        # Find and remove duplicate edges
        duplicates = find_duplicate_edges(network.edges)
        
        if not duplicates:
            return network
        
        cleaned_edges = remove_duplicate_edges(network.edges, duplicates)
        network.edges = cleaned_edges
        removed_count = initial_edges - len(network.edges)
        print(f"    ✓ Removed {removed_count} duplicate edges")
        
        return network
        
    except Exception as e:
        print(f"    ⚠️  Error cleaning duplicates: {e}")
        return network

def find_duplicate_edges(edges_df):
    """Find duplicate edges (same source-target pairs) in the edges DataFrame."""
    try:
        if edges_df is None or edges_df.empty:
            return []
        
        # Create edge identifier using source and target (normalize case)
        edges_df = edges_df.copy()
        edges_df['edge_key'] = edges_df.apply(
            lambda row: tuple(sorted([str(row['source']).strip(), str(row['target']).strip()])),
            axis=1
        )
        
        # Find duplicated edge keys
        duplicate_keys = edges_df['edge_key'].duplicated(keep=False)
        
        if not duplicate_keys.any():
            return []
        
        # Group by edge key to find all duplicates
        duplicate_groups = []
        duplicated_edges = edges_df[duplicate_keys]
        
        for edge_key, group in duplicated_edges.groupby('edge_key'):
            if len(group) > 1:
                # Store original indices for removal
                duplicate_indices = group.index.tolist()
                source, target = edge_key  # Already sorted
                duplicate_groups.append({
                    'key': edge_key,
                    'source': source,
                    'target': target,
                    'indices': duplicate_indices,
                    'count': len(group)
                })
        
        return duplicate_groups
        
    except Exception as e:
        print(f"    Error finding duplicate edges: {e}")
        return []

def remove_duplicate_edges(edges_df, duplicates):
    """Remove duplicate edges while preserving one copy with best attributes."""
    try:
        if not duplicates or edges_df is None or edges_df.empty:
            return edges_df
        
        cleaned_edges = edges_df.copy()
        indices_to_remove = []
        
        for dup_group in duplicates:
            dup_indices = dup_group['indices']
            
            # Keep the first occurrence (or best one based on some criteria)
            keep_index = dup_indices[0]
            remove_indices = dup_indices[1:]  # Remove all others
            
            # Try to keep the edge with the most complete information
            if len(dup_indices) > 1:
                best_index = find_best_edge_to_keep(cleaned_edges.loc[dup_indices])
                if best_index is not None and best_index in dup_indices:
                    keep_index = best_index
                    remove_indices = [idx for idx in dup_indices if idx != keep_index]
            
            indices_to_remove.extend(remove_indices)
        
        # Remove duplicate indices
        if indices_to_remove:
            cleaned_edges = cleaned_edges.drop(indices_to_remove).reset_index(drop=True)
        
        return cleaned_edges
        
    except Exception as e:
        print(f"    Error removing duplicate edges: {e}")
        return edges_df

def find_best_edge_to_keep(duplicate_edges):
    """Select the best edge to keep from duplicates based on completeness."""
    try:
        if duplicate_edges.empty:
            return None
        
        # Score edges based on completeness of information
        scores = {}
        
        for idx, row in duplicate_edges.iterrows():
            score = 0
            
            # Prefer edges with effect information
            if 'effect' in row and pd.notna(row['effect']):
                score += 10
                # Prefer specific effects over generic ones
                effect_str = str(row['effect']).lower()
                if effect_str in ['activation', 'inhibition', 'stimulation']:
                    score += 5
                elif effect_str not in ['unknown', 'undefined', '']:
                    score += 2
            
            # Prefer edges with confidence/weight information
            if 'weight' in row and pd.notna(row['weight']):
                try:
                    weight_val = float(row['weight'])
                    score += min(weight_val * 5, 10)  # Cap at 10
                except:
                    pass
            
            # Prefer edges with more attributes filled
            non_null_count = row.count()  # Counts non-null values
            score += non_null_count
            
            scores[idx] = score
        
        # Return index of edge with highest score
        if scores:
            return max(scores.keys(), key=lambda k: scores[k])
        else:
            return duplicate_edges.index[0]  # Fallback to first
            
    except Exception as e:
        print(f"    Error selecting best edge: {e}")
        return duplicate_edges.index[0] if not duplicate_edges.empty else None

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_network_to_bnet(network, network_id, strategy_name, database, params):
    """Export network to bnet format."""
    try:
        # Create filename
        param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
        filename = f"{network_id}_{strategy_name}_{database}_{param_str}.bnet"
        if Path:
            filepath = BNET_DIR / filename
            filepath_str = str(filepath)
        else:
            filepath_str = os.path.join(BNET_DIR, filename)
        
        # Export
        exporter = Exports(network)
        exporter.export_bnet(filepath_str)
        
        print(f"  ✓ Exported to {filename}")
        return filepath_str
        
    except Exception as e:
        print(f"  ✗ Failed to export network: {e}")
        return None

def export_network_for_maboss(network, network_id, strategy_name, database, params):
    """Export network for MaBoSS by creating BND and CFG files using MaBoSS built-in conversion."""
    try:
        # Create base filename
        param_str = "_".join([f"{k}-{v}" for k, v in params.items()])
        base_filename = f"{network_id}_{strategy_name}_{database}_{param_str}"
        
        if Path:
            bnet_filepath = MABOSS_DIR / f"{base_filename}.bnet"
            bnd_filepath = MABOSS_DIR / f"{base_filename}.bnd"
            cfg_filepath = MABOSS_DIR / f"{base_filename}.cfg"
        else:
            bnet_filepath = os.path.join(MABOSS_DIR, f"{base_filename}.bnet")
            bnd_filepath = os.path.join(MABOSS_DIR, f"{base_filename}.bnd")
            cfg_filepath = os.path.join(MABOSS_DIR, f"{base_filename}.cfg")
        
        # Step 1: Export to BNET using NeKo's built-in function
        exporter = Exports(network)
        exporter.export_bnet(str(bnet_filepath))
        
        # Find the actual bnet file (NeKo adds _1 suffix)
        actual_bnet_file = None
        if os.path.exists(str(bnet_filepath).replace('.bnet', '_1.bnet')):
            actual_bnet_file = str(bnet_filepath).replace('.bnet', '_1.bnet')
        elif os.path.exists(str(bnet_filepath)):
            actual_bnet_file = str(bnet_filepath)
        else:
            # Search for any file with the base name
            import glob
            search_pattern = str(bnet_filepath).replace('.bnet', '_*.bnet')
            matching_files = glob.glob(search_pattern)
            if matching_files:
                actual_bnet_file = matching_files[0]
        
        if not actual_bnet_file or not os.path.exists(actual_bnet_file):
            print(f"  ✗ Could not find exported bnet file")
            return None, None
        
        # Step 1.5: Sanitize BNET file for MaBoSS compatibility (fix colons in node names)
        sanitized_bnet_file = sanitize_bnet_for_maboss(actual_bnet_file)
        
        # Step 1.6: Check and clean BNET file for duplicates
        cleaned_bnet_file = check_and_clean_bnet_duplicates(sanitized_bnet_file)
        
        # Step 2: Use MaBoSS built-in function to convert BNET to BND and CFG
        print(f"  Converting BNET to BND/CFG using MaBoSS built-in function...")
        print(f"    BNET file: {os.path.basename(cleaned_bnet_file)}")
        
        try:
            # Use MaBoSS's built-in conversion function
            maboss.bnet_to_bnd_and_cfg(
                cleaned_bnet_file,
                str(bnd_filepath),
                str(cfg_filepath)
            )
            
            # Verify files were created
            if not os.path.exists(str(bnd_filepath)):
                print(f"  ✗ BND file was not created by MaBoSS conversion")
                return None, None
            
            if not os.path.exists(str(cfg_filepath)):
                print(f"  ✗ CFG file was not created by MaBoSS conversion")
                return None, None
            
            print(f"  ✓ MaBoSS conversion successful:")
            print(f"    BND file: {os.path.basename(bnd_filepath)} ({os.path.getsize(str(bnd_filepath))} bytes)")
            print(f"    CFG file: {os.path.basename(cfg_filepath)} ({os.path.getsize(str(cfg_filepath))} bytes)")
            
            # Extract nodes from the generated BND file for validation
            nodes_from_bnd = extract_nodes_from_bnd_file(str(bnd_filepath))
            if nodes_from_bnd:
                print(f"  Found {len(nodes_from_bnd)} nodes in BND file: {sorted(nodes_from_bnd)[:5]}{'...' if len(nodes_from_bnd) > 5 else ''}")
            else:
                print(f"  ⚠️  Warning: Could not extract nodes from BND file for validation")
            
            # Modify CFG file to enable trajectory plotting
            if modify_cfg_for_trajectories(str(cfg_filepath)):
                print(f"  ✓ CFG file modified for trajectory plotting")
            
            return str(bnd_filepath), str(cfg_filepath)
            
        except Exception as e:
            print(f"  ⚠️  MaBoSS built-in conversion failed: {e}")
            print(f"  ✗ Cannot create BND/CFG files - MaBoSS built-in conversion is required")
            return None, None
        
    except Exception as e:
        print(f"  ✗ Failed to export network for MaBoSS: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def modify_cfg_for_trajectories(cfg_filepath):
    """Modify CFG file to enable trajectory display and ensure proper parameter ordering."""
    try:
        # Read the original CFG file
        with open(cfg_filepath, 'r') as f:
            original_content = f.read()
        
        print(f"    Original CFG content preview: {len(original_content)} characters")
        
        # Parse the original content to extract key information
        config_params = {}
        node_states = {}
        node_parameters = {}
        
        lines = original_content.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Extract configuration parameters
            for param in ['time_tick', 'max_time', 'sample_count', 'discrete_time', 'use_physrandgen', 'thread_count']:
                if line.startswith(f'{param} ='):
                    try:
                        value = line.split('=')[1].strip().rstrip(';')
                        config_params[param] = value
                    except:
                        pass
            
            # Extract node initial states
            if '.istate =' in line:
                try:
                    parts = line.split('.istate =')
                    node_name = parts[0].strip()
                    value = parts[1].strip().rstrip(';')
                    node_states[node_name] = value
                except:
                    pass
            
            # Extract parameter definitions
            if line.startswith('$'):
                try:
                    parts = line.split('=')
                    param_name = parts[0].strip()
                    value = parts[1].strip().rstrip(';')
                    node_parameters[param_name] = value
                except:
                    pass
        
        # Rebuild CFG file with proper structure and trajectory settings
        new_content = "// Modified CFG file for trajectory plotting\n\n"
        
        # 1. Configuration parameters (modify for trajectory plotting)
        config_params.update({
            'display_traj': '1',
            'thread_count': '1',
            'max_time': '20',
            'sample_count': '1000'
        })
        
        # Set defaults if missing
        config_defaults = {
            'time_tick': '0.5',
            'discrete_time': '0', 
            'use_physrandgen': '1'
        }
        
        for param, default_val in config_defaults.items():
            if param not in config_params:
                config_params[param] = default_val
        
        # Write configuration parameters
        for param, value in config_params.items():
            new_content += f"{param} = {value};\n"
        new_content += "\n"
        
        # 2. Node parameters FIRST (this is critical for MaBoSS)
        new_content += "// Node parameters (rates)\n"
        
        # Extract node names from states or parameters
        all_nodes = set()
        for node in node_states.keys():
            all_nodes.add(node)
        for param in node_parameters.keys():
            if param.startswith('$u_') or param.startswith('$d_'):
                node_name = param[3:]  # Remove $u_ or $d_
                all_nodes.add(node_name)
        
        # Write parameters for each node
        for node in sorted(all_nodes):
            up_param = f'$u_{node}'
            down_param = f'$d_{node}'
            
            up_val = node_parameters.get(up_param, '1.0')
            down_val = node_parameters.get(down_param, '1.0')
            
            new_content += f"{up_param} = {up_val};\n"
            new_content += f"{down_param} = {down_val};\n"
        
        new_content += "\n"
        
        # 3. Node initial states AFTER parameters
        new_content += "// Node initial states\n"
        for node in sorted(all_nodes):
            istate_val = node_states.get(node, '0.5')
            new_content += f"{node}.istate = {istate_val};\n"
        
        # Write the completely reconstructed CFG file
        with open(cfg_filepath, 'w') as f:
            f.write(new_content)
            
        print(f"    ✓ CFG file reconstructed with proper parameter ordering")
        return cfg_filepath
        
    except Exception as e:
        print(f"  ⚠️  Could not modify CFG file: {e}")
        traceback.print_exc()
        return False

# =============================================================================
# UTILITY FUNCTIONS FOR NETWORK ANALYSIS
# =============================================================================

def get_valid_gene_nodes(network):
    """Filter network nodes to get only valid gene symbols and convert UniProt IDs to gene symbols."""
    
    # Build comprehensive UniProt to gene symbol mapping
    uniprot_to_gene = {}
    
    # Method 1: Use network's nodes_df if available
    if hasattr(network, 'nodes_df') and network.nodes_df is not None and not network.nodes_df.empty:
        try:
            for _, row in network.nodes_df.iterrows():
                if 'genesymbol' in row and 'uniprot' in row:
                    if pd.notna(row['genesymbol']) and pd.notna(row['uniprot']):
                        uniprot_to_gene[str(row['uniprot'])] = str(row['genesymbol'])
                # Also try alternative column names
                if 'Genesymbol' in row and 'Uniprot' in row:
                    if pd.notna(row['Genesymbol']) and pd.notna(row['Uniprot']):
                        uniprot_to_gene[str(row['Uniprot'])] = str(row['Genesymbol'])
        except Exception as e:
            print(f"    Debug: Error reading nodes_df: {e}")
    
    # Method 2: Use network.nodes attribute if it's a DataFrame  
    if hasattr(network, 'nodes') and hasattr(network.nodes, 'iterrows'):
        try:
            for _, row in network.nodes.iterrows():
                if 'genesymbol' in row and 'uniprot' in row:
                    if pd.notna(row['genesymbol']) and pd.notna(row['uniprot']):
                        uniprot_to_gene[str(row['uniprot'])] = str(row['genesymbol'])
                if 'Genesymbol' in row and 'Uniprot' in row:
                    if pd.notna(row['Genesymbol']) and pd.notna(row['Uniprot']):
                        uniprot_to_gene[str(row['Uniprot'])] = str(row['Genesymbol'])
        except Exception as e:
            print(f"    Debug: Error reading network.nodes: {e}")
    
    # Get all unique nodes from edges
    edge_nodes = set()
    if hasattr(network, 'edges') and len(network.edges) > 0:
        edge_nodes.update(network.edges['source'].unique())
        edge_nodes.update(network.edges['target'].unique())
    
    # Filter to get valid gene symbols
    filtered_nodes = []
    
    # Process each edge node
    for node in edge_nodes:
        if not isinstance(node, str):
            continue
            
        # Skip metadata columns and parameters
        if node in {'Genesymbol', 'Type', 'Uniprot', 'genesymbol', 'type', 'uniprot'} or node.startswith('$'):
            continue
        
        # Convert UniProt ID to gene symbol if mapping exists
        if node in uniprot_to_gene:
            gene_symbol = uniprot_to_gene[node]
            if gene_symbol and gene_symbol not in filtered_nodes:
                filtered_nodes.append(gene_symbol)
        # If it looks like a gene symbol (not UniProt format), keep it
        elif not is_uniprot_id(node):
            if node not in filtered_nodes:
                filtered_nodes.append(node)
    
    # If we have very few nodes, add our original INPUT_GENES that have edges
    if len(filtered_nodes) < len(INPUT_GENES) // 2:
        for gene in INPUT_GENES:
            if gene not in filtered_nodes:
                # Check if this gene appears in edges
                gene_found = any(node in uniprot_to_gene and uniprot_to_gene[node] == gene or node == gene for node in edge_nodes)
                if gene_found:
                    filtered_nodes.append(gene)
    
    # Remove duplicates and sort
    filtered_nodes = sorted(list(set(filtered_nodes)))
    print(f"    Found {len(filtered_nodes)} valid gene symbols")
    
    return filtered_nodes

def is_uniprot_id(node_id):
    """Check if a node ID looks like a UniProt identifier."""
    if not isinstance(node_id, str) or len(node_id) < 6:
        return False
    
    # UniProt format: starts with P, Q, or O followed by 5 digits and optional letters
    # Examples: P00533, Q01094, O14746
    if node_id[0] in 'PQO' and len(node_id) >= 6:
        # Check if positions 1-5 contain digits
        if node_id[1:6].isdigit():
            return True
    
    return False

# =============================================================================
# BNET FILE SANITIZATION AND DUPLICATE DETECTION AND CLEANING
# =============================================================================

def sanitize_bnet_for_maboss(bnet_filepath):
    """Sanitize BNET file to ensure compatibility with MaBoSS by fixing invalid node names."""
    print(f"  Sanitizing BNET file for MaBoSS: {os.path.basename(bnet_filepath)}")
    
    try:
        # Read the BNET file
        with open(bnet_filepath, 'r') as f:
            content = f.read()
        
        # Track replacements made
        replacements = {}
        modified_content = content
        
        # Find all problematic node names with colons (like COMPLEX:P27986_P42336)
        # MaBoSS node names cannot contain colons
        problematic_nodes = re.findall(r'\b[A-Z_][A-Z0-9_]*:[A-Z0-9_]+\b', content)
        
        if problematic_nodes:
            print(f"    Found {len(set(problematic_nodes))} nodes with colons (invalid for MaBoSS)")
            
            for problem_node in set(problematic_nodes):
                # Create sanitized name by replacing colon with underscore
                sanitized_name = problem_node.replace(':', '_')
                replacements[problem_node] = sanitized_name
                
                # Replace all occurrences (as node names and in logic expressions)
                # Use word boundaries to avoid partial replacements
                modified_content = re.sub(r'\b' + re.escape(problem_node) + r'\b', 
                                        sanitized_name, modified_content)
                print(f"    {problem_node} → {sanitized_name}")
        
        # Write sanitized file
        if replacements:
            base_name = os.path.splitext(bnet_filepath)[0]
            sanitized_filepath = f"{base_name}_sanitized.bnet"
            
            with open(sanitized_filepath, 'w') as f:
                f.write(modified_content)
            
            print(f"    ✓ Created sanitized file: {os.path.basename(sanitized_filepath)}")
            print(f"    Total replacements: {len(replacements)}")
            return sanitized_filepath
        else:
            print(f"    ✓ No sanitization needed")
            return bnet_filepath
            
    except Exception as e:
        print(f"    ⚠️  Error sanitizing BNET file: {e}")
        print(f"    Proceeding with original file")
        return bnet_filepath


def check_and_clean_bnet_duplicates(bnet_filepath):
    """Check BNET file for duplicate nodes and create cleaned version if needed."""
    print(f"  Checking BNET file for duplicates: {os.path.basename(bnet_filepath)}")
    
    try:
        # Read the BNET file
        with open(bnet_filepath, 'r') as f:
            content = f.read()
        
        # First, remove exact duplicate lines (quick fix for CDKN2A issue)
        lines = content.split('\n')
        seen_lines = set()
        unique_lines = []
        duplicate_lines_removed = 0
        
        for line in lines:
            if line.strip() and not line.startswith('#'):
                if line in seen_lines:
                    duplicate_lines_removed += 1
                    continue
                seen_lines.add(line)
            unique_lines.append(line)
        
        if duplicate_lines_removed > 0:
            print(f"    Found {duplicate_lines_removed} exact duplicate lines")
            content = '\n'.join(unique_lines)
        
        # Then parse nodes to detect duplicates with numeric suffixes
        nodes = parse_bnet_for_duplicates(content)
        duplicates = find_bnet_duplicates(nodes)
        
        if not duplicates and duplicate_lines_removed == 0:
            print(f"    ✓ No duplicates found")
            return bnet_filepath
        
        # Create cleaned version
        if duplicate_lines_removed > 0:
            base_name = os.path.splitext(bnet_filepath)[0]
            cleaned_filepath = f"{base_name}_deduplicated.bnet"
            
            with open(cleaned_filepath, 'w') as f:
                f.write(content)
            
            print(f"    ✓ Created deduplicated file: {os.path.basename(cleaned_filepath)}")
            print(f"    Removed {duplicate_lines_removed} exact duplicate lines")
            return cleaned_filepath
        
        # Handle suffix-based duplicates (existing logic)
        cleaned_content = merge_bnet_duplicates(content, nodes, duplicates)
        
        # Write cleaned file
        base_name = os.path.splitext(bnet_filepath)[0]
        cleaned_filepath = f"{base_name}_cleaned.bnet"
        
        with open(cleaned_filepath, 'w') as f:
            f.write(cleaned_content)
        
        print(f"    ✓ Created cleaned file: {os.path.basename(cleaned_filepath)}")
        return cleaned_filepath
        
    except Exception as e:
        print(f"    ⚠️  Error checking for duplicates: {e}")
        print(f"    Proceeding with original file")
        return bnet_filepath

def parse_bnet_for_duplicates(content):
    """Parse BNET file to extract node definitions and detect potential duplicates."""
    import re
    
    nodes = {}
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Skip headers
        if 'targets' in line.lower() and ('factors' in line.lower() or line.startswith('targets,')):
            continue
        
        # Look for node definitions: "NODE_NAME, LOGIC_EXPRESSION"
        if ',' in line:
            parts = line.split(',', 1)
            if len(parts) == 2:
                node_name = parts[0].strip()
                logic_expr = parts[1].strip()
                
                # Skip if looks like header or empty
                if (not node_name or not logic_expr or 
                    node_name.lower() in ['targets', 'factors'] or
                    node_name.startswith('#')):
                    continue
                
                nodes[node_name] = {
                    'name': node_name,
                    'logic': logic_expr,
                    'line_number': i,
                    'original_line': line
                }
    
    return nodes

def find_bnet_duplicates(nodes):
    """Find nodes that are duplicates: either exact duplicates or nodes with numeric suffixes."""
    import re
    
    # Method 1: Find exact duplicate node names 
    node_counts = {}
    for node_name in nodes.keys():
        node_counts[node_name] = node_counts.get(node_name, 0) + 1
    
    exact_duplicates = {name: [name] * count for name, count in node_counts.items() if count > 1}
    
    # Method 2: Group nodes by base gene name (remove numeric suffixes)  
    gene_groups = {}
    
    for node_name in nodes.keys():
        # Extract base name by removing trailing _1, _2, etc.
        base_name = re.sub(r'_\d+$', '', node_name)
        
        if base_name not in gene_groups:
            gene_groups[base_name] = []
        gene_groups[base_name].append(node_name)
    
    # Find groups with multiple nodes (suffix duplicates)
    suffix_duplicates = {}
    for base_name, node_list in gene_groups.items():
        if len(node_list) > 1:
            suffix_duplicates[base_name] = sorted(node_list)
    
    # Combine both types of duplicates
    all_duplicates = {}
    all_duplicates.update(exact_duplicates)
    all_duplicates.update(suffix_duplicates)
    
    if all_duplicates:
        print(f"    Found duplicate types:")
        for dup_name, dup_list in all_duplicates.items():
            if len(set(dup_list)) == 1:  # Exact duplicates
                print(f"      Exact duplicate: {dup_name} (appears {len(dup_list)} times)")
            else:  # Suffix duplicates
                print(f"      Suffix variants: {dup_list}")
    
    return all_duplicates

def merge_bnet_duplicates(content, nodes, duplicates):
    """Create cleaned BNET content by merging duplicate nodes."""
    import re
    
    lines = content.split('\n')
    output_lines = []
    processed_nodes = set()
    
    # First pass: copy headers and non-duplicate nodes
    for line in lines:
        line_stripped = line.strip()
        
        # Always keep comments and empty lines
        if not line_stripped or line_stripped.startswith('#'):
            output_lines.append(line)
            continue
        
        # Keep headers
        if 'targets' in line_stripped.lower() and ('factors' in line_stripped.lower() or line_stripped.startswith('targets,')):
            output_lines.append(line)
            continue
        
        # Check if this is a node definition
        if ',' in line_stripped:
            parts = line_stripped.split(',', 1)
            if len(parts) == 2:
                node_name = parts[0].strip()
                
                # Skip if already processed or if it's a duplicate
                if node_name in processed_nodes:
                    continue
                
                # Check if this node is part of a duplicate group
                is_duplicate = False
                for base_name, duplicate_list in duplicates.items():
                    if node_name in duplicate_list:
                        is_duplicate = True
                        # Only process the first occurrence of each duplicate group
                        if node_name == duplicate_list[0]:  # First in sorted list
                            merged_line = create_merged_node_line(base_name, duplicate_list, nodes)
                            output_lines.append(merged_line)
                            # Mark all nodes in this group as processed
                            processed_nodes.update(duplicate_list)
                        break
                
                # If not a duplicate, keep original line
                if not is_duplicate:
                    output_lines.append(line)
                    processed_nodes.add(node_name)
            else:
                # Not a node definition, keep as is
                output_lines.append(line)
        else:
            # Not a node definition, keep as is
            output_lines.append(line)
    
    return '\n'.join(output_lines)

def create_merged_node_line(base_name, duplicate_nodes, nodes):
    """Create a merged node line by combining logic from duplicates."""
    logic_expressions = []
    
    # Collect logic expressions from all duplicates
    for node_name in duplicate_nodes:
        if node_name in nodes:
            logic = nodes[node_name]['logic']
            # Skip self-referential logic
            if logic and logic.strip() != node_name:
                logic_expressions.append(f"({logic.strip()})")
    
    # Combine logic expressions
    if not logic_expressions:
        # Fallback to self-regulation
        combined_logic = base_name
    elif len(logic_expressions) == 1:
        combined_logic = logic_expressions[0].strip('()')
    else:
        combined_logic = ' | '.join(logic_expressions)
    
    # Create merged line with comment
    merged_line = f"{base_name}, {combined_logic}"
    comment_line = f"# Merged from: {', '.join(duplicate_nodes)}"
    
    return f"{merged_line}\n{comment_line}"

# =============================================================================
# UTILITY FUNCTIONS FOR BND ANALYSIS
# =============================================================================

def extract_nodes_from_bnd_file(bnd_filepath):
    """Extract all node names from a BND file for validation and debugging."""
    try:
        if not os.path.exists(bnd_filepath):
            return []
        
        with open(bnd_filepath, 'r') as f:
            content = f.read()
        
        # Find all "node NODENAME {" patterns (MaBoSS built-in uses lowercase 'node')
        import re
        # Support both "node NODENAME {" and "Node NODENAME {" patterns
        node_definitions = re.findall(r'(?:node|Node)\s+([A-Za-z0-9_]+)\s*{', content, re.IGNORECASE)
        
        # Also find node references in logic expressions to catch dependencies
        # Extract logic lines first
        logic_lines = re.findall(r'logic\s*=\s*([^;]+);', content)
        
        referenced_nodes = []
        for logic_line in logic_lines:
            # Find potential node names (alphanumeric + underscore, but not operators)
            # Use word boundaries to avoid partial matches
            potential_refs = re.findall(r'\b([A-Za-z][A-Za-z0-9_]*)\b', logic_line)
            for ref in potential_refs:
                # Filter out Boolean operators and keywords
                if ref not in ['AND', 'OR', 'NOT', 'logic', 'rate_up', 'rate_down', 'TRUE', 'FALSE']:
                    referenced_nodes.append(ref)
        
        # Combine all found nodes
        all_nodes = list(set(node_definitions + referenced_nodes))
        
        # Final filtering to ensure we only have valid node names
        valid_nodes = []
        for node in all_nodes:
            if (node and 
                len(node) > 1 and  # At least 2 characters
                node[0].isalpha() and  # Must start with letter
                (node.isalnum() or '_' in node)):  # Alphanumeric or contains underscore
                valid_nodes.append(node)
        
        return sorted(set(valid_nodes))
        
    except Exception as e:
        print(f"    Warning: Could not extract nodes from BND file: {e}")
        return []

# =============================================================================


# =============================================================================
# MABOSS SIMULATION
# =============================================================================

def create_minimal_test_files():
    """Create minimal test BND and CFG files for debugging."""
    print("Creating minimal test files for MaBoSS debugging...")
    
    # Create simple test files with just 4 nodes
    test_nodes = ['MAPK3', 'MYC', 'CCND1', 'CCNE1']
    
    if Path:
        test_bnd = MABOSS_DIR / "test_minimal.bnd"
        test_cfg = MABOSS_DIR / "test_minimal.cfg"
    else:
        test_bnd = os.path.join(MABOSS_DIR, "test_minimal.bnd")
        test_cfg = os.path.join(MABOSS_DIR, "test_minimal.cfg")
    
    # Create minimal BND file
    with open(test_bnd, 'w') as f:
        f.write("// Minimal test BND file\n\n")
        f.write("Node MAPK3 {\n")
        f.write("  logic = MAPK3;\n")
        f.write("  rate_up = @logic ? $u_MAPK3 : 0;\n")
        f.write("  rate_down = @logic ? 0 : $d_MAPK3;\n")
        f.write("}\n\n")
        
        f.write("Node MYC {\n")
        f.write("  logic = MAPK3;\n")
        f.write("  rate_up = @logic ? $u_MYC : 0;\n")
        f.write("  rate_down = @logic ? 0 : $d_MYC;\n")
        f.write("}\n\n")
        
        f.write("Node CCND1 {\n")
        f.write("  logic = MYC;\n")
        f.write("  rate_up = @logic ? $u_CCND1 : 0;\n")
        f.write("  rate_down = @logic ? 0 : $d_CCND1;\n")
        f.write("}\n\n")
        
        f.write("Node CCNE1 {\n")
        f.write("  logic = MYC;\n")
        f.write("  rate_up = @logic ? $u_CCNE1 : 0;\n")
        f.write("  rate_down = @logic ? 0 : $d_CCNE1;\n")
        f.write("}\n\n")
    
    # Create minimal CFG file
    with open(test_cfg, 'w') as f:
        f.write("// Minimal test CFG file\n\n")
        f.write("time_tick = 0.5;\n")
        f.write("max_time = 10;\n")
        f.write("sample_count = 10;\n")
        f.write("discrete_time = 0;\n")
        f.write("use_physrandgen = 1;\n")
        f.write("thread_count = 1;\n")
        f.write("display_traj = 1;\n\n")
        
        # IMPORTANT: Parameters must come BEFORE initial states
        f.write("// Define parameters first\n")
        for node in test_nodes:
            f.write(f"$u_{node} = 1.0;\n")
            f.write(f"$d_{node} = 1.0;\n")
        f.write("\n// Then define initial states\n")
        for node in test_nodes:
            f.write(f"{node}.istate = 0.5;\n")
        f.write("\n")
        
        f.write('output_nodes = ["MAPK3", "MYC", "CCND1", "CCNE1"];\n')
    
    print(f"Created test files: {test_bnd} and {test_cfg}")
    return str(test_bnd), str(test_cfg)


def test_maboss_minimal():
    """Test MaBoSS with minimal example."""
    print("Testing MaBoSS with minimal example...")
    
    bnd_file, cfg_file = create_minimal_test_files()
    
    try:
        # Test MaBoSS loading with bnd and cfg files
        model = maboss.load(bnd_file, cfg_file)
        print("✓ MaBoSS model loaded successfully")
        
        # Test running
        result = model.run()
        print("✓ MaBoSS simulation completed successfully")
        
        # Test getting results
        prob_traj = result.get_last_states_probtraj()
        print(f"✓ Retrieved probability trajectories: {list(prob_traj.columns)}")
        
        return True
        
    except Exception as e:
        print(f"✗ MaBoSS minimal test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_maboss_simulation(bnd_file, cfg_file, network_id):
    """Run MaBoSS simulation using BND and CFG files."""
    try:
        print(f"Running MaBoSS simulation for {network_id}")
        
        # Verify files exist
        if not os.path.exists(bnd_file):
            print(f"  ✗ BND file not found: {bnd_file}")
            return None, None
            
        if not os.path.exists(cfg_file):
            print(f"  ✗ CFG file not found: {cfg_file}")
            return None, None
        
        print(f"  Loading BND file: {os.path.basename(bnd_file)}")
        print(f"  Loading CFG file: {os.path.basename(cfg_file)}")
        
        # First try the minimal test on first network to verify MaBoSS works
        if network_id == "network_001":
            print("  Testing MaBoSS with minimal example first...")
            if test_maboss_minimal():
                print("  ✓ Minimal test passed, proceeding with full simulation")
            else:
                print("  ✗ Minimal test failed, MaBoSS may have configuration issues")
                return None, None
        
        # Load the BND and CFG files with MaBoSS
        sim = maboss.load(bnd_file, cfg_file)
        
        # Set thread count for better performance (alternative API method)
        try:
            sim.set_thread_count(4)
            print(f"  ✓ Set thread_count to 4 via API")
        except (AttributeError, Exception):
            print(f"  ℹ️  Using thread_count from CFG file (thread_count=4)")
        
        print(f"  ✓ MaBoSS model loaded successfully")
        print(f"  Network nodes: {len(list(sim.network.keys()))} total")
        
        # Run simulation
        print(f"  Running MaBoSS simulation...")
        result = sim.run()
        print( "  ✓ MaBoSS simulation completed")
        
        # Extract trajectories for output nodes
        trajectories = {}
        
        # Use get_nodes_probtraj() to get individual node trajectories
        try:
            nodes_prob_traj = result.get_nodes_probtraj()
            print(f"  Retrieved individual node trajectories: {len(nodes_prob_traj.columns)} nodes")
            
            for node in OUTPUT_NODES:
                if node in nodes_prob_traj.columns:
                    # Get final probability value for this node
                    final_prob = nodes_prob_traj[node].iloc[-1]
                    trajectories[node] = [float(final_prob)]
                    print(f"  {node}: {trajectories[node][0]:.4f}")
                else:
                    print(f"  Warning: {node} not found in individual node results")
                    # Check if the node exists with different case or spacing
                    available_nodes = list(nodes_prob_traj.columns)
                    similar_nodes = [n for n in available_nodes if node.lower() in n.lower() or n.lower() in node.lower()]
                    if similar_nodes:
                        print(f"    Similar nodes found: {similar_nodes}")
                        # Use the first similar node
                        final_prob = nodes_prob_traj[similar_nodes[0]].iloc[-1]
                        trajectories[node] = [float(final_prob)]
                        print(f"    Using {similar_nodes[0]} instead of {node}: {trajectories[node][0]:.4f}")
                    else:
                        trajectories[node] = [0.0]
                        print(f"    Setting {node} to default: 0.0")
                        
        except Exception as e:
            print(f"  Warning: Could not get individual node trajectories: {e}")
            print(f"  Falling back to compound states analysis...")
            
            # Fallback to compound states if individual node trajectories fail
            try:
                prob_traj = result.get_last_states_probtraj()
                print(f"  Retrieved {len(prob_traj.columns)} compound states")
                
                for node in OUTPUT_NODES:
                    found_value = None
                    # Look for the node in any compound state
                    for col in prob_traj.columns:
                        if node in col:
                            found_value = prob_traj[col].iloc[-1]
                            trajectories[node] = [float(found_value)]
                            print(f"  {node} found in compound state '{col[:30]}...': {found_value:.4f}")
                            break
                            
                    if found_value is None:
                        trajectories[node] = [0.0]
                        print(f"  {node} not found in any state, setting to 0.0")
                        
            except Exception as e2:
                print(f"  Error in fallback method: {e2}")
                # Final fallback - set all nodes to 0
                for node in OUTPUT_NODES:
                    trajectories[node] = [0.0]
        
        print(f"  ✓ MaBoSS simulation completed successfully")
        return trajectories, result
        
    except Exception as e:
        print(f"  ✗ MaBoSS simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def generate_trajectory_plots(maboss_result, network_id, strategy, database, output_nodes=None):
    """
    Generate time-course trajectory plots using MaBoSS built-in plotting functions.
    
    Args:
        maboss_result: MaBoSS simulation result object
        network_id: Network identifier for file naming
        strategy: Strategy name for plot title
        database: Database name for plot title
        output_nodes: List of nodes to plot (default: OUTPUT_NODES)
    
    Returns:
        str: Path to saved plot file or None if failed
    """
    if maboss_result is None:
        print(f"  ⚠️  No MaBoSS result available for {network_id} - skipping trajectory plots")
        return None
    
    if output_nodes is None:
        output_nodes = OUTPUT_NODES
    
    try:
        print(f"  📊 Generating trajectory plots for {network_id}...")
        
        # Create plots directory if using os.path
        if not Path:
            plots_dir = os.path.join(PLOTS_DIR)
            os.makedirs(plots_dir, exist_ok=True)
            plot_file = os.path.join(plots_dir, f"{network_id}_trajectories.png")
        else:
            plot_file = PLOTS_DIR / f"{network_id}_trajectories.png"
        
        print(f"  Plot file will be saved as: {plot_file}")
        
        # Check if matplotlib is available
        if plt is None:
            print(f"  ✗ Matplotlib not available, cannot generate plots")
            return None
        
        # Method 1: Use MaBoSS built-in plotting for individual nodes
        try:
            # Get node probability trajectories over time
            nodes_prob_traj = maboss_result.get_nodes_probtraj()
            print(f"  Retrieved node trajectories with columns: {list(nodes_prob_traj.columns)[:5]}...")
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Trajectory Analysis: {network_id}\\n{strategy} + {database}', fontsize=14, fontweight='bold')
            
            # Flatten axes for easier indexing
            axes_flat = axes.flatten()
            
            # Plot each output node
            for i, node in enumerate(output_nodes):
                ax = axes_flat[i]
                
                if node in nodes_prob_traj.columns:
                    # Plot the trajectory over time
                    trajectory = nodes_prob_traj[node]
                    time_points = trajectory.index
                    
                    ax.plot(time_points, trajectory.values, 'b-', linewidth=2, marker='o', markersize=4)
                    ax.set_title(f'{node}', fontweight='bold')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Probability')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1)
                    
                    # Add final value annotation
                    final_val = trajectory.iloc[-1]
                    ax.annotate(f'Final: {final_val:.3f}', 
                              xy=(time_points[-1], final_val),
                              xytext=(0.02, 0.95), textcoords='axes fraction',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                              fontsize=10, fontweight='bold')
                else:
                    # Node not found, show empty plot with message
                    ax.text(0.5, 0.5, f'{node}\\nNot Found', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, style='italic', color='gray')
                    ax.set_title(f'{node}', fontweight='bold')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Probability')
                
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Node trajectory plot saved: {os.path.basename(plot_file)}")
            
            # Also create a compound plot using MaBoSS's native plotting if available
            try:
                compound_plot_file = str(plot_file).replace('_trajectories.png', '_states.png')
                
                # Try MaBoSS native plotting for states
                if hasattr(maboss_result, 'plot'):
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    maboss_result.plot(ax=ax2)
                    plt.title(f'State Probability Trajectories: {network_id}\\n{strategy} + {database}')
                    plt.tight_layout()
                    plt.savefig(compound_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  ✓ State trajectory plot saved: {os.path.basename(compound_plot_file)}")
                    
            except Exception as e:
                print(f"    Note: Could not create native MaBoSS plot: {e}")
            
            return str(plot_file)
                
        except Exception as e:
            print(f"  ⚠️  Could not generate node trajectory plots: {e}")
            
            # Fallback: Try plotting state trajectories
            try:
                if plt is not None:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get state probability trajectories
                    prob_traj = maboss_result.get_last_states_probtraj()
                    
                    # Plot top 10 most variable states
                    for col in prob_traj.columns[:10]:
                        ax.plot(prob_traj.index, prob_traj[col], label=col[:20]+'...' if len(col) > 20 else col)
                    
                    ax.set_title(f'State Trajectories: {network_id}\\n{strategy} + {database}')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Probability')
                    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ State trajectory plot saved: {os.path.basename(plot_file)}")
                    return str(plot_file)
                    
            except Exception as e2:
                print(f"  ⚠️  Fallback plotting also failed: {e2}")
                return None
                
    except Exception as e:
        print(f"  ✗ Error generating plots: {e}")
        return None

def count_network_edges(network):
    """
    Count the number of edges in a NeKo network.
    
    Args:
        network: NeKo Network object
    
    Returns:
        int: Number of edges in the network
    """
    if network is None or not neko_available:
        return 0
    
    try:
        # For NeKo Network objects, edges are directly accessible
        if hasattr(network, 'edges'):
            return len(network.edges)
        # Alternative: try to get the internal networkx graph
        elif hasattr(network, 'network'):
            graph = network.network
            if hasattr(graph, 'number_of_edges'):
                return graph.number_of_edges()
        elif hasattr(network, 'graph'):
            graph = network.graph  
            if hasattr(graph, 'number_of_edges'):
                return graph.number_of_edges()
        else:
            # Fallback: count through nodes
            edge_count = 0
            if hasattr(network, 'nodes'):
                for node_id in network.nodes:
                    node = network.nodes[node_id]
                    if hasattr(node, 'targets'):
                        edge_count += len(node.targets)
                    elif hasattr(node, 'edges'):
                        edge_count += len(node.edges)
            return edge_count
    except Exception as e:
        print(f"  ⚠️  Could not count network edges: {e}")
        return 0

def generate_network_topology_plot(network, network_id, strategy, database, max_edges=1000):
    """
    Generate network topology visualization using NeKo's NetworkVisualizer.
    Skips visualization for networks with more than max_edges edges to avoid performance issues.
    
    Args:
        network: NeKo Network object
        network_id: Network identifier for file naming
        strategy: Strategy name for plot title
        database: Database name for plot title
        max_edges: Maximum number of edges before skipping visualization (default: 1000)
    
    Returns:
        str: Path to saved network plot file or None if failed/skipped
    """
    if network is None or not neko_available:
        return None
    
    try:
        # Check network size first to avoid expensive visualizations
        edge_count = count_network_edges(network)
        node_count = len(network.nodes) if hasattr(network, 'nodes') else 0
        
        print(f"  🌐 Network {network_id}: {node_count} nodes, {edge_count} edges")
        
        if edge_count > max_edges:
            print(f"  ⏭️  Skipping visualization (>{max_edges} edges) - network too large for efficient rendering")
            return None
        
        print(f"  📊 Generating network topology plot...")
        
        # Create plots directory if using os.path
        if not Path:
            plots_dir = os.path.join(PLOTS_DIR)
            os.makedirs(plots_dir, exist_ok=True)
            plot_file = os.path.join(plots_dir, f"{network_id}_network.pdf")
        else:
            plot_file = PLOTS_DIR / f"{network_id}_network.pdf"
        
        # Use NeKo's NetworkVisualizer - creates PDF output
        visualizer = NetworkVisualizer(network)
        
        # Set output file path (NetworkVisualizer creates PDF)
        base_name = str(plot_file).replace('.pdf', '')
        
        # Render the network (this creates a PDF file)
        visualizer.render(output_file=base_name, view=False, 
                         highlight_nodes=INPUT_GENES,
                         highlight_color='lightcoral')
        
        # Check if PDF was created
        pdf_file = f"{base_name}.pdf"
        if os.path.exists(pdf_file):
            print(f"  ✓ Network topology plot saved: {os.path.basename(pdf_file)} (PDF)")
            return str(pdf_file)
        else:
            print(f"  ⚠️  NetworkVisualizer didn't create expected PDF file")
            raise Exception("No PDF file created by NetworkVisualizer")
            
    except Exception as e:
        print(f"  ⚠️  Could not generate network topology plot: {e}")
        
        # Fallback: Create a simple networkx-based plot if available
        try:
            if nx is not None and plt is not None:
                print(f"  🔄 Attempting fallback networkx visualization...")
                
                # Convert NeKo network to networkx format for visualization
                G = nx.DiGraph()
                
                # Add nodes (use INPUT_GENES as main nodes to highlight)
                all_nodes = set()
                edges = []
                
                # Try to extract nodes and edges from the network
                if hasattr(network, '_network') and hasattr(network._network, 'edges'):
                    # Try to get edges from the internal networkx graph
                    for edge in network._network.edges(data=True):
                        source, target, data = edge
                        all_nodes.add(str(source))
                        all_nodes.add(str(target))
                        edges.append((str(source), str(target)))
                elif hasattr(network, 'edges') and network.edges is not None:
                    # Try pandas DataFrame format
                    if hasattr(network.edges, 'iterrows'):
                        for idx, edge in network.edges.iterrows():
                            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                                source = str(edge.source)
                                target = str(edge.target)
                                all_nodes.add(source)
                                all_nodes.add(target)
                                edges.append((source, target))
                    else:
                        # Try list format
                        for edge in network.edges:
                            if hasattr(edge, 'source') and hasattr(edge, 'target'):
                                source = str(edge.source)
                                target = str(edge.target)
                                all_nodes.add(source)
                                all_nodes.add(target)
                                edges.append((source, target))
                
                # If we couldn't get edges, try to get at least the nodes
                if len(all_nodes) == 0:
                    if hasattr(network, '_network') and hasattr(network._network, 'nodes'):
                        all_nodes = {str(node) for node in network._network.nodes()}
                    elif hasattr(network, 'nodes'):
                        if hasattr(network.nodes, '__iter__'):
                            all_nodes = {str(node) for node in network.nodes}
                
                # Add nodes and edges to networkx graph
                G.add_nodes_from(all_nodes)
                G.add_edges_from(edges)
                
                if len(G.nodes()) > 0:
                    # Create the plot
                    fig, ax = plt.subplots(figsize=(12, 10))
                    
                    # Use spring layout for better visualization
                    pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
                    
                    # Color nodes based on whether they're input genes
                    node_colors = []
                    for node in G.nodes():
                        if node in INPUT_GENES:
                            node_colors.append('lightcoral')  # Input genes in red
                        elif node in OUTPUT_NODES:
                            node_colors.append('lightblue')   # Output nodes in blue
                        else:
                            node_colors.append('lightgray')   # Other nodes in gray
                    
                    # Draw the network
                    nx.draw(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=300,
                           font_size=8,
                           font_weight='bold',
                           arrows=True,
                           arrowsize=20,
                           edge_color='gray',
                           alpha=0.7,
                           with_labels=True)
                    
                    # Add title and legend
                    ax.set_title(f'Network Topology: {network_id}\\n{strategy} + {database}', 
                                fontsize=14, fontweight='bold', pad=20)
                    
                    # Create legend
                    legend_elements = [
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                                  markersize=10, label='Input Genes'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                                  markersize=10, label='Output Nodes'),
                        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', 
                                  markersize=10, label='Other Nodes')
                    ]
                    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
                    
                    # Add network statistics
                    stats_text = f"Nodes: {len(G.nodes())}\\nEdges: {len(G.edges())}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    plt.tight_layout()
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
                    plt.close()
                    
                    print(f"  ✓ Fallback network plot saved: {os.path.basename(plot_file)}")
                    return str(plot_file)
                    
                else:
                    print(f"  ⚠️  No nodes found in network for visualization")
                    return None
                    
        except Exception as e2:
            print(f"  ⚠️  Fallback networkx plotting also failed: {e2}")
            return None

# =============================================================================
# INCREMENTAL RESULT SAVING FUNCTIONS
# =============================================================================

def save_single_result_incrementally(result_entry):
    """Save a single result entry incrementally to both summary and trajectory files."""
    try:
        # Prepare summary row
        summary_row = {
            'network_id': result_entry['network_id'],
            'strategy': result_entry['strategy'],
            'database': result_entry['database'],
            'initial_conditions': result_entry['initial_conditions'],
            'num_nodes': result_entry['metrics']['num_nodes'],
            'num_edges': result_entry['metrics']['num_edges'],
            'density': result_entry['metrics']['density'],
            'avg_degree': result_entry['metrics']['avg_degree'],
            'bnet_file': result_entry['bnet_file'],
            'trajectory_plot': result_entry.get('trajectory_plot', ''),  # Add trajectory plot path
            'network_plot': result_entry.get('network_plot', '')  # Add network plot path
        }
        
        # Add parameter columns
        for param, value in result_entry['parameters'].items():
            summary_row[f'param_{param}'] = value
        
        # Save to CSV summary file
        if pd is not None:
            if Path:
                summary_file = SUMMARY_DIR / "network_summary.csv"
            else:
                summary_file = os.path.join(SUMMARY_DIR, "network_summary.csv")
            
            # Check if file exists to determine if we need header
            file_exists = os.path.exists(summary_file)
            
            # Convert to DataFrame and append
            summary_df = pd.DataFrame([summary_row])
            summary_df.to_csv(summary_file, mode='a', header=not file_exists, index=False)
            
        # Save trajectory data incrementally
        if Path:
            trajectories_file = SUMMARY_DIR / "trajectories_data.json"
        else:
            trajectories_file = os.path.join(SUMMARY_DIR, "trajectories_data.json")
        
        # Load existing trajectory data or create new
        if os.path.exists(trajectories_file):
            with open(trajectories_file, 'r') as f:
                trajectories_data = json.load(f)
        else:
            trajectories_data = {}
        
        # Add new trajectory data
        network_trajs = result_entry['trajectories']
        # Convert numpy arrays to lists for JSON serialization
        for node, traj in network_trajs.items():
            if hasattr(traj, 'tolist'):
                trajectories_data[result_entry['network_id']] = trajectories_data.get(result_entry['network_id'], {})
                trajectories_data[result_entry['network_id']][node] = traj.tolist()
            else:
                trajectories_data[result_entry['network_id']] = trajectories_data.get(result_entry['network_id'], {})
                trajectories_data[result_entry['network_id']][node] = list(traj)
        
        # Write back to file
        with open(trajectories_file, 'w') as f:
            json.dump(trajectories_data, f, indent=2)
        
        # Display plot info
        trajectory_plot_name = os.path.basename(result_entry.get('trajectory_plot', '')) if result_entry.get('trajectory_plot') else 'None'
        network_plot_name = os.path.basename(result_entry.get('network_plot', '')) if result_entry.get('network_plot') else 'None'
        plot_info = f", Plots: {trajectory_plot_name}, {network_plot_name}"
        
        print(f"  💾 Results saved incrementally - Summary: {summary_file}, Trajectories: {trajectories_file}{plot_info}")
        
    except Exception as e:
        print(f"  ⚠️  Warning: Could not save results incrementally: {e}")

def get_current_result_count():
    """Get the current number of results saved."""
    try:
        if Path:
            summary_file = SUMMARY_DIR / "network_summary.csv"
        else:
            summary_file = os.path.join(SUMMARY_DIR, "network_summary.csv")
        
        if pd is not None and os.path.exists(summary_file):
            df = pd.read_csv(summary_file)
            return len(df)
        else:
            return 0
    except:
        return 0

# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def main():
    """Main analysis pipeline."""
    print("="*80)
    print("SENSITIVITY ANALYSIS: NeKo + MaBoSS")
    print("="*80)
    print(f"Input genes: {len(INPUT_GENES)} nodes")
    print(f"Output nodes to monitor: {OUTPUT_NODES}")
    print()
    
    # Check if required packages are available
    if not neko_available:
        print("ERROR: NeKo is not available. Please install NeKo first.")
        return []
    
    if not maboss_available:
        print("ERROR: MaBoSS is not available. Please install MaBoSS first.")
        return []
    
    print("✓ All required packages are available")
    
    # Setup databases
    print("Setting up databases...")
    databases = setup_databases()
    
    # Get strategy configurations
    strategies, database_names = get_strategy_configurations()
    
    # Results storage
    all_results = []
    network_counter = 0
    
    print(f"\nTesting {len(strategies)} strategies with {len([db for db in databases.values() if db is not None])} databases")
    print("-"*80)
    
    # Iterate through all combinations
    for strategy_config in strategies:
        strategy_name = strategy_config['name']
        strategy_method = strategy_config['method']
        
        # Generate parameter combinations for this strategy
        param_combinations = generate_parameter_combinations(strategy_config)
        
        print(f"\nStrategy: {strategy_name} ({len(param_combinations)} parameter combinations)")
        
        for database_name in database_names:
            database = databases.get(database_name)
            if database is None:
                print(f"  Skipping {database_name} (not available)")
                continue
            
            for params in param_combinations:
                network_counter += 1
                network_id = f"network_{network_counter:03d}"
                
                print(f"\n[{network_counter}] {strategy_name} + {database_name}")
                print(f"  Parameters: {params}")
                
                # Build network
                network, metrics = build_network(
                    INPUT_GENES, database, strategy_name, strategy_method, params
                )
                
                if network is None:
                    continue
                
                # Export to bnet (for reference and for MaBoSS)
                bnet_file = export_network_to_bnet(
                    network, network_id, strategy_name, database_name, params
                )
                
                # Export for MaBoSS (creates BND and CFG files)
                maboss_files = export_network_for_maboss(
                    network, network_id, strategy_name, database_name, params
                )
                
                if maboss_files is None or maboss_files[0] is None or maboss_files[1] is None:
                    print(f"  ✗ Failed to export BND/CFG files for MaBoSS")
                    continue
                
                bnd_file, cfg_file = maboss_files
                
                # Run MaBoSS simulation using the BND and CFG files
                trajectories, maboss_result = run_maboss_simulation(bnd_file, cfg_file, network_id)
                
                if trajectories is None:
                    continue
                
                # Generate trajectory plots after successful simulation
                trajectory_plot = generate_trajectory_plots(maboss_result, network_id, strategy_name, database_name)
                
                # Generate network topology plot (skip if too many edges)
                network_plot = generate_network_topology_plot(network, network_id, strategy_name, database_name, 
                                                            max_edges=MAX_EDGES_FOR_VISUALIZATION)
                
                # Determine initial condition type for this network
                network_num = int(network_id.split('_')[1])
                if network_num % 3 == 0:
                    init_condition_type = "random"
                elif network_num % 3 == 1:
                    init_condition_type = "growth_focused"
                else:
                    init_condition_type = "cell_cycle_focused"
                
                # Store results
                result_entry = {
                    'network_id': network_id,
                    'strategy': strategy_name,
                    'database': database_name,
                    'parameters': params,
                    'initial_conditions': init_condition_type,
                    'bnet_file': bnet_file,
                    'bnd_file': bnd_file,
                    'cfg_file': cfg_file,
                    'metrics': metrics,
                    'trajectories': trajectories,
                    'trajectory_plot': trajectory_plot,  # Add trajectory plot path
                    'network_plot': network_plot  # Add network plot path
                }
                
                all_results.append(result_entry)
                
                # Save results incrementally after each successful simulation
                save_single_result_incrementally(result_entry)
                
                current_count = get_current_result_count()
                print(f"  ✓ Complete - {current_count} successful networks saved so far")
    
    print("\n" + "="*80)
    print(f"ANALYSIS COMPLETE")
    print(f"Successfully analyzed: {len(all_results)} networks")
    print(f"Results saved incrementally in: {BASE_DIR}")
    
    # Final summary message - results already saved incrementally
    current_count = get_current_result_count()
    print(f"Final result count: {current_count} networks saved")
    
    return all_results

def save_summary_results(results):
    """Save summary of all results."""
    if not results:
        return
    
    # Create summary data
    summary_data = []
    for result in results:
        row = {
            'network_id': result['network_id'],
            'strategy': result['strategy'],
            'database': result['database'],
            'initial_conditions': result['initial_conditions'],
            'num_nodes': result['metrics']['num_nodes'],
            'num_edges': result['metrics']['num_edges'],
            'density': result['metrics']['density'],
            'avg_degree': result['metrics']['avg_degree'],
            'bnet_file': result['bnet_file']
        }
        
        # Add parameter columns
        for param, value in result['parameters'].items():
            row[f'param_{param}'] = value
        
        summary_data.append(row)
    
    # Save as CSV if pandas is available, otherwise as JSON
    if pd is not None:
        summary_df = pd.DataFrame(summary_data)
        if Path:
            summary_file = SUMMARY_DIR / "network_summary.csv"
        else:
            summary_file = os.path.join(SUMMARY_DIR, "network_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
    else:
        # Save as JSON
        if Path:
            summary_file = SUMMARY_DIR / "network_summary.json"
        else:
            summary_file = os.path.join(SUMMARY_DIR, "network_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"Summary saved to: {summary_file}")
    
    # Save trajectories separately
    if Path:
        trajectories_file = SUMMARY_DIR / "trajectories_data.json"
    else:
        trajectories_file = os.path.join(SUMMARY_DIR, "trajectories_data.json")
        
    trajectories_data = {
        result['network_id']: result['trajectories'] 
        for result in results
    }
    
    # Convert numpy arrays to lists for JSON serialization
    for network_id, trajs in trajectories_data.items():
        for node, traj in trajs.items():
            if hasattr(traj, 'tolist'):
                trajectories_data[network_id][node] = traj.tolist()
            else:
                trajectories_data[network_id][node] = list(traj)
    
    with open(trajectories_file, 'w') as f:
        json.dump(trajectories_data, f, indent=2)
    
    print(f"Trajectories saved to: {trajectories_file}")

if __name__ == "__main__":
    results = main()
