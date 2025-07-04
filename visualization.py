"""
Visualization module for the traffic simulation project.
Handles map plotting, route visualization, and congestion visualization.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import time
from matplotlib.lines import Line2D
from typing import Dict, List, Optional, Tuple, Any

from models import Vehicle
from routing import create_congestion_graph


def enhanced_visualize_congestion_map(G: nx.Graph, congestion_data: Dict[str, float], 
                                    vehicles: List[Vehicle], scenario: str = "Normal",
                                    notable_locations: Optional[Dict[str, int]] = None) -> str:
    """
    Enhanced visualization with clearer source/destination, notable locations, and routes.
    
    Args:
        G: NetworkX graph
        congestion_data: Dictionary of congestion values
        vehicles: List of vehicles to visualize
        scenario: Current scenario name
        notable_locations: Dictionary mapping location names to node IDs
        
    Returns:
        Path to saved map file
    """
    print("Creating visualization for current scenario...")
    
    # Create output directory
    maps_dir = os.path.join('london_simulation', 'scenario_maps')
    os.makedirs(maps_dir, exist_ok=True)
    
    # Create a figure for visualization
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Set up colors - Use RdYlGn_r for better visualization of congestion
    cmap = plt.cm.RdYlGn_r  # Red (high congestion) to Green (low congestion)
    
    # Set up colors for algorithms
    algo_colors = {
        'Dijkstra': 'blue',
        'A*': 'red',
        'Shortest Path': 'orange',
        'Shortest Path Congestion Aware': 'green'
    }
    
    # Get node positions from the graph
    node_positions = {}
    for node, data in G.nodes(data=True):
        node_positions[node] = (data['x'], data['y'])

    # Plot with congestion coloring
    norm = plt.Normalize(vmin=1.0, vmax=5.0)  # For 1-5 congestion scale (updated)
    
    # Track congestion statistics for verification
    congestion_values = []
    
    # Plot all edges with congestion-based coloring
    for u, v, k in G.edges(keys=True):
        edge_id = f"{u}_{v}_{k}"
        if edge_id in congestion_data:
            congestion = congestion_data[edge_id]
            congestion_values.append(congestion)
            vehicle_count = G[u][v][k].get('vehicle_count', 0)
            
            # Color based on congestion
            color = cmap(norm(congestion))
            
            # Line width based on both congestion and vehicle count
            linewidth = 0.5 + (congestion * 0.2) + (min(vehicle_count, 10) * 0.1)
            
            # Get coordinates
            if u in node_positions and v in node_positions:
                u_x, u_y = node_positions[u]
                v_x, v_y = node_positions[v]
                
                # Plot the edge
                ax.plot([u_x, v_x], [u_y, v_y], color=color, linewidth=linewidth, alpha=0.8)
    
    # Print congestion statistics for debugging
    if congestion_values:
        mean_congestion = np.mean(congestion_values)
        min_congestion = np.min(congestion_values)
        max_congestion = np.max(congestion_values)
        print(f"  Current map congestion - Mean: {mean_congestion:.2f}, Min: {min_congestion:.2f}, Max: {max_congestion:.2f}")
        print(f"  Total edges with congestion data: {len(congestion_values)} / {len(G.edges())}")
    else:
        print("  WARNING: No congestion values found for visualization!")
    
    # Plot routes for selected vehicles
    selected_vehicles = [v for v in vehicles if v.selected_for_analysis] if vehicles else []
    route_legend_elements = []
    
    if selected_vehicles:
        # Create congestion graph to get accurate travel times
        G_congestion = create_congestion_graph(G, congestion_data)
        
        for vehicle in selected_vehicles:
            if vehicle.paths and len(vehicle.paths) >= 1:
                # Calculate travel times for display
                travel_times = {}
                for algo_name, path in vehicle.paths.items():
                    if not path or len(path) < 2:
                        continue
                        
                    # Calculate total travel time for this path
                    total_time = 0
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if u in G_congestion and v in G_congestion[u]:
                            total_time += G_congestion[u][v]['travel_time']
                    
                    travel_times[algo_name] = total_time
                
                # Find the best algorithm based on travel time
                if travel_times:
                    best_algo = min(travel_times.items(), key=lambda x: x[1])[0]
                else:
                    best_algo = None
                
                # Plot each path with different color
                for algo_name, path in vehicle.paths.items():
                    if not path or len(path) < 2:
                        continue
                        
                    color = algo_colors.get(algo_name, 'purple')
                    linewidth = 2.5
                    style = '-'
                    alpha = 0.7
                    
                    # Highlight the best path
                    if algo_name == best_algo:
                        linewidth = 3.5
                        alpha = 1.0
                        label = f"Vehicle {vehicle.id}: {algo_name} (Best)"
                    else:
                        label = f"Vehicle {vehicle.id}: {algo_name}"
                    
                    # Create legend element if first vehicle with this algorithm
                    if algo_name not in [item.get_label().split(':')[-1].strip() for item in route_legend_elements]:
                        route_legend_elements.append(
                            Line2D([0], [0], color=color, linewidth=3, 
                                   label=f"{algo_name} algorithm")
                        )
                    
                    # Plot the path
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if u in node_positions and v in node_positions:
                            u_x, u_y = node_positions[u]
                            v_x, v_y = node_positions[v]
                            
                            # Plot the edge
                            ax.plot([u_x, v_x], [u_y, v_y], color=color, linestyle=style, 
                                  linewidth=linewidth, alpha=alpha, zorder=20)
    
    # Plot notable locations to ensure they're visible
    if notable_locations:
        for name, node_id in notable_locations.items():
            if node_id in node_positions:
                x, y = node_positions[node_id]
                # Mark notable locations with distinct symbols
                ax.scatter(x, y, s=80, marker='s', color='purple', alpha=0.7, edgecolor='black', zorder=10)
                # Add labels with small offset to prevent overlap
                ax.text(x, y+0.0001, name, fontsize=9, ha='center', va='bottom', 
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'), zorder=11)
    
    # Plot vehicles if provided
    if vehicles:
        # Only plot up to 100 vehicles for clarity
        plot_vehicles = vehicles[:100] if len(vehicles) > 100 else vehicles
        
        for vehicle in plot_vehicles:
            if vehicle.path and len(vehicle.path) > 1:
                # Get source and destination coordinates
                src = vehicle.path[0]
                dst = vehicle.path[-1]
                
                if src in node_positions and dst in node_positions:
                    src_x, src_y = node_positions[src]
                    dst_x, dst_y = node_positions[dst]
                    
                    # Mark source with a larger green dot
                    ax.scatter(src_x, src_y, s=100, c='lime', marker='o', edgecolor='black', alpha=0.8, zorder=12)
                    ax.text(src_x, src_y-0.0001, f"Start {vehicle.id}", fontsize=10, ha='center', va='top', zorder=13)
                    
                    # Mark destination with a larger red dot
                    ax.scatter(dst_x, dst_y, s=100, c='red', marker='o', edgecolor='black', alpha=0.8, zorder=12)
                    ax.text(dst_x, dst_y+0.0001, f"End {vehicle.id}", fontsize=10, ha='center', va='bottom', zorder=13)
                    
                    # Highlight vehicles selected for analysis with a circle
                    if vehicle.selected_for_analysis:
                        # Draw connecting line
                        ax.plot([src_x, dst_x], [src_y, dst_y], 'b--', linewidth=1.5, alpha=0.5, zorder=9)
                        # Add highlight ring
                        ax.scatter(src_x, src_y, s=150, facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8, zorder=11)
                        ax.scatter(dst_x, dst_y, s=150, facecolors='none', edgecolors='blue', linewidth=2, alpha=0.8, zorder=11)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Congestion Level (1-5)', fontsize=12)
    
    # Create scenario-specific descriptions
    scenario_descriptions = {
        "Normal": "Normal Traffic (Range: 1-4)",
        "Morning": "Morning Rush Hour (Range: 1.5-4)",
        "Evening": "Evening Rush Hour (Range: 2-4)",
        "Weekend": "Weekend Traffic (Range: 1-3)",
        "Special": "Special Event (Range: 1-5)"
    }
    
    # Title and styling
    vehicle_count = len(vehicles) if vehicles else 0
    selected_count = sum(1 for v in vehicles if v.selected_for_analysis) if vehicles else 0
    
    scenario_desc = scenario_descriptions.get(scenario, scenario)
    title = f'London Road Network - {scenario_desc}\n'
    
    # Add congestion statistics to title
    if congestion_values:
        title += f'Mean Congestion: {mean_congestion:.2f}, Min: {min_congestion:.2f}, Max: {max_congestion:.2f}\n'
        
    title += f'Congestion Map with {vehicle_count} Vehicles'
    if selected_count > 0:
        title += f' ({selected_count} selected for analysis)'
    
    plt.title(title, fontsize=16)
    
    # Add legend for map symbols (updated for 1-5 range)
    legend_elements = [
        Line2D([0], [0], color=cmap(norm(1.0)), linewidth=6, label='Low Congestion (1.0-2.0)'),
        Line2D([0], [0], color=cmap(norm(3.0)), linewidth=6, label='Medium Congestion (2.0-3.5)'),
        Line2D([0], [0], color=cmap(norm(4.5)), linewidth=6, label='High Congestion (3.5-5.0)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='purple', markersize=10, label='Notable Location'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label='Vehicle Start'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Vehicle End'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='blue', markersize=10, label='Selected for Analysis')
    ]
    
    # Add route algorithm colors to legend if any routes are displayed
    if route_legend_elements:
        legend_elements.extend(route_legend_elements)
    
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Remove axis
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure
    timestamp = int(time.time())
    map_file = os.path.join(maps_dir, f"congestion_map_{scenario.replace(' ', '_')}_{timestamp}.png")
    plt.savefig(map_file, dpi=300)
    print(f"  Saved map to {map_file}")
    
    # Show the figure
    plt.show()
    
    return map_file


def debug_edge_data_structure(G: nx.Graph, sample_edges: int = 5) -> None:
    """
    Debug helper to understand edge data structure.
    
    Args:
        G: NetworkX graph
        sample_edges: Number of edges to examine
    """
    print("=== Edge Data Structure Debug ===")
    
    edge_count = 0
    for u, v, k in G.edges(keys=True):
        if edge_count >= sample_edges:
            break
            
        print(f"\nEdge {u} -> {v} (key: {k}):")
        
        # Check G[u][v] structure
        if u in G and v in G[u]:
            edge_dict = G[u][v]
            print(f"  G[{u}][{v}] type: {type(edge_dict)}")
            print(f"  G[{u}][{v}] keys: {list(edge_dict.keys()) if isinstance(edge_dict, dict) else 'Not a dict'}")
            
            if isinstance(edge_dict, dict) and k in edge_dict:
                edge_data = edge_dict[k]
                print(f"  G[{u}][{v}][{k}] type: {type(edge_data)}")
                
                if isinstance(edge_data, dict):
                    print(f"  G[{u}][{v}][{k}] keys: {list(edge_data.keys())}")
                    if 'travel_time' in edge_data:
                        print(f"  travel_time: {edge_data['travel_time']} (type: {type(edge_data['travel_time'])})")
                elif hasattr(edge_data, 'get'):
                    print(f"  Edge data has 'get' method")
                else:
                    print(f"  Edge data value: {edge_data}")
        
        # Check G.edges access
        try:
            edge_attrs = G.edges[u, v, k]
            print(f"  G.edges[{u}, {v}, {k}] type: {type(edge_attrs)}")
            if isinstance(edge_attrs, dict):
                print(f"  G.edges[{u}, {v}, {k}] keys: {list(edge_attrs.keys())}")
                if 'travel_time' in edge_attrs:
                    print(f"  travel_time via G.edges: {edge_attrs['travel_time']} (type: {type(edge_attrs['travel_time'])})")
        except Exception as e:
            print(f"  G.edges access failed: {e}")
            
        edge_count += 1
    
    print(f"\nAnalyzed {edge_count} sample edges")


def test_edge_access_methods(G: nx.Graph) -> None:
    """
    Test different ways to access edge data.
    
    Args:
        G: NetworkX graph
    """
    print("=== Testing Edge Access Methods ===")
    
    # Get first edge for testing
    edges = list(G.edges(keys=True))
    if not edges:
        print("No edges to test")
        return
        
    u, v, k = edges[0]
    print(f"Testing edge: {u} -> {v} (key: {k})")
    
    # Method 1: G[u][v][k]
    try:
        edge_data = G[u][v][k]
        print(f"Method 1 - G[u][v][k]: {type(edge_data)} = {edge_data}")
        if hasattr(edge_data, 'get'):
            print(f"  Has 'get' method: {edge_data.get('travel_time', 'NOT_FOUND')}")
        else:
            print(f"  No 'get' method available")
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: G.edges[u, v, k]
    try:
        edge_attrs = G.edges[u, v, k]
        print(f"Method 2 - G.edges[u, v, k]: {type(edge_attrs)} = {edge_attrs}")
        if isinstance(edge_attrs, dict):
            print(f"  travel_time: {edge_attrs.get('travel_time', 'NOT_FOUND')}")
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: G.get_edge_data(u, v, k)
    try:
        edge_data = G.get_edge_data(u, v, k)
        print(f"Method 3 - G.get_edge_data(u, v, k): {type(edge_data)} = {edge_data}")
        if isinstance(edge_data, dict):
            print(f"  travel_time: {edge_data.get('travel_time', 'NOT_FOUND')}")
    except Exception as e:
        print(f"Method 3 failed: {e}")


def debug_congestion_flow(G: nx.Graph, congestion_data: Dict[str, float], 
                         vehicle_path: List[int], step_name: str = "Debug") -> None:
    """
    Debug function to trace congestion data through the system.
    
    Args:
        G: NetworkX graph
        congestion_data: Dictionary of congestion values
        vehicle_path: Path to debug
        step_name: Name of the debug step
    """
    print(f"\n=== {step_name} - Congestion Flow Debug ===")
    
    if not vehicle_path or len(vehicle_path) < 2:
        print("No valid vehicle path to debug")
        return
    
    print(f"Debugging path: {vehicle_path[:5]}...{vehicle_path[-5:] if len(vehicle_path) > 10 else vehicle_path}")
    
    # Check first few edges in the path
    for i in range(min(3, len(vehicle_path) - 1)):
        u, v = vehicle_path[i], vehicle_path[i + 1]
        print(f"\nEdge {u} -> {v}:")
        
        # Check vehicle count
        if u in G and v in G[u]:
            for k in G[u][v]:
                edge_id = f"{u}_{v}_{k}"
                vehicle_count = G[u][v][k].get('vehicle_count', 0)
                edge_congestion = G[u][v][k].get('congestion', 'NOT_SET')
                
                print(f"  Edge ID: {edge_id}")
                print(f"  Vehicle count: {vehicle_count}")
                print(f"  G[u][v][k]['congestion']: {edge_congestion}")
                
                # Check congestion_data
                congestion_data_value = congestion_data.get(edge_id, 'NOT_FOUND')
                print(f"  congestion_data['{edge_id}']: {congestion_data_value}")
                
                # Check if values match
                if edge_congestion != congestion_data_value:
                    print(f"  ⚠️  MISMATCH: Graph congestion != congestion_data")
                
                if vehicle_count > 5:
                    print(f"  📊 High vehicle count but congestion = {edge_congestion}")
                elif vehicle_count > 0:
                    print(f"  📊 Some vehicles, congestion = {edge_congestion}")
                else:
                    print(f"  📊 No vehicles on this edge")
                
                break  # Only check first key
        else:
            print(f"  ❌ Edge not found in graph")


def debug_travel_time_calculation(G: nx.Graph, congestion_data: Dict[str, float], 
                                sample_edges: int = 3) -> None:
    """
    Debug how travel times are being calculated.
    
    Args:
        G: NetworkX graph
        congestion_data: Dictionary of congestion values
        sample_edges: Number of edges to examine
    """
    print(f"\n=== Travel Time Calculation Debug ===")
    
    # Create congestion graph
    G_congestion = create_congestion_graph(G, congestion_data)
    
    print("Sample edges from congestion graph:")
    
    edge_count = 0
    for u, v, k in G.edges(keys=True):
        if edge_count >= sample_edges:
            break
            
        edge_id = f"{u}_{v}_{k}"
        print(f"\nEdge {edge_id}:")
        
        # Original graph data
        if u in G and v in G[u] and k in G[u][v]:
            orig_data = G[u][v][k]
            length = orig_data.get('length', 'NOT_FOUND')
            speed = orig_data.get('speed', 'NOT_FOUND')
            vehicle_count = orig_data.get('vehicle_count', 0)
            graph_congestion = orig_data.get('congestion', 'NOT_FOUND')
            
            print(f"  Original graph - Length: {length}, Speed: {speed}")
            print(f"  Vehicle count: {vehicle_count}")
            print(f"  Graph congestion: {graph_congestion}")
        
        # Congestion data
        congestion_value = congestion_data.get(edge_id, 'NOT_FOUND')
        print(f"  congestion_data value: {congestion_value}")
        
        # Congestion graph data
        if u in G_congestion and v in G_congestion[u] and k in G_congestion[u][v]:
            try:
                cong_data = G_congestion[u][v][k]
                if isinstance(cong_data, dict):
                    travel_time = cong_data.get('travel_time', 'NOT_FOUND')
                    base_travel_time = cong_data.get('base_travel_time', 'NOT_FOUND')
                    congestion = cong_data.get('congestion', 'NOT_FOUND')
                    
                    print(f"  Congestion graph - Travel time: {travel_time}")
                    print(f"  Base travel time: {base_travel_time}")
                    print(f"  Congestion level: {congestion}")
                else:
                    print(f"  Congestion graph data: {cong_data} (type: {type(cong_data)})")
            except Exception as e:
                print(f"  Error accessing congestion graph: {e}")
        else:
            print(f"  ❌ Edge not found in congestion graph")
            
        edge_count += 1


def quick_congestion_check(G: nx.Graph, congestion_data: Dict[str, float]) -> None:
    """
    Quick check of overall congestion state.
    
    Args:
        G: NetworkX graph
        congestion_data: Dictionary of congestion values
    """
    print(f"\n=== Quick Congestion Check ===")
    
    # Check congestion_data statistics
    if congestion_data:
        values = list(congestion_data.values())
        print(f"congestion_data: {len(values)} edges")
        print(f"  Min: {min(values):.2f}, Max: {max(values):.2f}, Avg: {sum(values)/len(values):.2f}")
    else:
        print("congestion_data is empty!")
    
    # Check vehicle counts
    total_vehicles = 0
    edges_with_vehicles = 0
    max_vehicle_count = 0
    
    for u, v, k in G.edges(keys=True):
        vehicle_count = G[u][v][k].get('vehicle_count', 0)
        if vehicle_count > 0:
            edges_with_vehicles += 1
            total_vehicles += vehicle_count
            max_vehicle_count = max(max_vehicle_count, vehicle_count)
    
    print(f"Vehicle distribution:")
    print(f"  Total vehicles on edges: {total_vehicles}")
    print(f"  Edges with vehicles: {edges_with_vehicles}")
    print(f"  Max vehicles on single edge: {max_vehicle_count}")
    
    if edges_with_vehicles == 0:
        print("  ⚠️  NO VEHICLES FOUND ON ANY EDGES!")
    elif max_vehicle_count > 10:
        print(f"  📊 High vehicle density detected")
    else:
        print(f"  📊 Moderate vehicle distribution")