"""
Congestion modeling and scenario management for the traffic simulation project.
Includes MM1 queuing model and congestion scenario generation.
"""

import numpy as np
import networkx as nx
import random
import time
import os
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from models import Vehicle, calculate_congestion_stats


def calculate_mm1_congestion(arrival_rate: float, service_rate: float, base_congestion: float = 1.0) -> float:
    """
    Calculate congestion using MM1 queuing model.
    
    Args:
        arrival_rate: Rate at which vehicles arrive (λ)
        service_rate: Rate at which vehicles depart (μ)
        base_congestion: Base congestion level of the road
        
    Returns:
        Congestion factor (1-10 scale)
    """
    # Check for valid service rate (must be greater than arrival rate)
    if arrival_rate >= service_rate:
        return 10.0  # Maximum congestion
    
    # Calculate utilization factor (ρ)
    utilization = arrival_rate / service_rate
    
    # Calculate average number of vehicles in the system (L)
    avg_vehicles = utilization / (1 - utilization)
    
    # Calculate average time in system (W)
    avg_time = 1 / (service_rate * (1 - utilization))
    
    # Calculate congestion factor based on MM1 model and base congestion
    congestion_factor = base_congestion * (1 + 2 * utilization + avg_vehicles / 3)
    
    # Ensure congestion is within 1-10 range
    congestion_factor = max(1.0, min(10.0, congestion_factor))
    
    return congestion_factor


def get_mm1_statistics(arrival_rate: float, service_rate: float) -> Dict[str, Any]:
    """
    Get detailed statistics from the MM1 queuing model.
    
    Args:
        arrival_rate: Rate at which vehicles arrive
        service_rate: Rate at which vehicles depart
        
    Returns:
        Dictionary with MM1 statistics
    """
    if arrival_rate >= service_rate:
        return {
            'utilization': float('inf'),
            'avg_vehicles_in_system': float('inf'),
            'avg_vehicles_in_queue': float('inf'),
            'avg_time_in_system': float('inf'),
            'avg_time_in_queue': float('inf'),
            'system_stable': False
        }
    
    # Calculate utilization factor (ρ)
    utilization = arrival_rate / service_rate
    
    # Calculate MM1 model statistics
    avg_vehicles_in_system = utilization / (1 - utilization)  # L
    avg_vehicles_in_queue = (utilization**2) / (1 - utilization)  # Lq
    avg_time_in_system = 1 / (service_rate * (1 - utilization))  # W
    avg_time_in_queue = utilization / (service_rate * (1 - utilization))  # Wq
    
    return {
        'utilization': utilization,
        'avg_vehicles_in_system': avg_vehicles_in_system,
        'avg_vehicles_in_queue': avg_vehicles_in_queue,
        'avg_time_in_system': avg_time_in_system,
        'avg_time_in_queue': avg_time_in_queue,
        'system_stable': True
    }


def create_random_hotspots(center_x: float, center_y: float, max_dist: float, 
                          num_hotspots: int = 3, radius_factor: float = 0.2) -> List[Tuple[float, float, float, float]]:
    """
    Create random congestion hotspots within the map.
    
    Args:
        center_x: X coordinate of map center
        center_y: Y coordinate of map center
        max_dist: Maximum distance from center
        num_hotspots: Number of hotspots to create
        radius_factor: Factor for hotspot radius calculation
        
    Returns:
        List of tuples (x, y, radius, intensity)
    """
    hotspots = []
    
    for _ in range(num_hotspots):
        # Random angle and distance from center
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0.1, 0.9) * max_dist
        
        # Convert to cartesian coordinates
        x = center_x + distance * np.cos(angle)
        y = center_y + distance * np.sin(angle)
        
        # Random radius and intensity
        radius = np.random.uniform(0.1, 0.3) * max_dist * radius_factor
        intensity = np.random.uniform(0.7, 1.0)
        
        hotspots.append((x, y, radius, intensity))
    
    return hotspots


def apply_consistent_congestion_scenario(G: nx.Graph, congestion_data: Dict[str, float], 
                                       scenario: str, base_congestion: Optional[Dict[str, float]] = None, 
                                       special_case_type: Optional[str] = None) -> Tuple[Dict[str, float], str]:
    """
    Apply a specific congestion scenario to the graph.
    
    Args:
        G: NetworkX graph
        congestion_data: Current congestion data
        scenario: Scenario name (Normal, Morning, Evening, Weekend, Special)
        base_congestion: Base congestion for reference
        special_case_type: Type of special case if scenario is Special
        
    Returns:
        Tuple of (updated_congestion_data, excel_file_path)
    """
    print(f"\nApplying '{scenario}' congestion scenario...")
    G.graph["scenario_name"] = scenario
    G.graph["special_case_type"] = special_case_type
    
    # Determine reference baseline
    if base_congestion is not None:
        print("  Using provided 'base_congestion' as reference.")
        reference_congestion = base_congestion.copy()
    else:
        print("  WARNING: 'base_congestion' not provided.")
        if not congestion_data:
            raise ValueError("Cannot determine reference congestion.")
        reference_congestion = congestion_data.copy()
    
    # Create a new random seed based on current time
    random_seed = int(time.time()) % (2**32)
    np.random.seed(random_seed)
    print(f"  Using random seed: {random_seed} for {scenario} scenario")
    
    # Get node positions for geographic-based congestion patterns
    node_positions = {}
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_positions[node] = (data['x'], data['y'])
    
    # Find the center and bounds of the map
    if node_positions:
        coords = np.array(list(node_positions.values()))
        center_x, center_y = np.mean(coords, axis=0)
        min_x, min_y = np.min(coords, axis=0)
        max_x, max_y = np.max(coords, axis=0)
        max_dist = max(max_x - min_x, max_y - min_y) / 2
    else:
        center_x, center_y = 0, 0
        max_dist = 0.05
    
    # Create random hotspots for this scenario
    if scenario == "Morning":
        num_hotspots = np.random.randint(3, 6)
        hotspots = create_random_hotspots(center_x, center_y, max_dist, num_hotspots)
    elif scenario == "Evening":
        num_hotspots = np.random.randint(4, 7)
        hotspots = create_random_hotspots(center_x, center_y, max_dist, num_hotspots)
    elif scenario == "Weekend":
        num_hotspots = np.random.randint(5, 9)
        hotspots = create_random_hotspots(center_x, center_y, max_dist, num_hotspots)
    elif scenario == "Special":
        num_hotspots = np.random.randint(1, 10)
        hotspots = create_random_hotspots(center_x, center_y, max_dist, num_hotspots)
    else:  # Normal
        num_hotspots = np.random.randint(2, 5)
        hotspots = create_random_hotspots(center_x, center_y, max_dist, num_hotspots)
    
    print(f"  Created {num_hotspots} random congestion hotspots")
    
    # Create a new congestion data dictionary
    scenario_result_congestion = {}
    edges_updated = 0
    
    for u, v, k in G.edges(keys=True):
        edge_id = f"{u}_{v}_{k}"
        
        # Get edge coordinates if available
        edge_x, edge_y = center_x, center_y
        if u in node_positions:
            u_x, u_y = node_positions[u]
            v_x, v_y = center_x, center_y
            if v in node_positions:
                v_x, v_y = node_positions[v]
            edge_x, edge_y = (u_x + v_x) / 2, (u_y + v_y) / 2
        
        # Calculate realistic baseline congestion based on scenario
        if scenario == "Normal":
            base_congestion = 1.0 + 3.0 * np.random.random()  # 1.0-4.0
        elif scenario == "Morning":
            base_congestion = 1.5 + 2.5 * np.random.random()  # 1.5-4.0 (slightly higher)
        elif scenario == "Evening":
            base_congestion = 2.0 + 2.0 * np.random.random()  # 2.0-4.0 (busier)
        elif scenario == "Weekend":
            base_congestion = 1.0 + 2.0 * np.random.random()  # 1.0-3.0 (lighter)
        elif scenario == "Special":
            base_congestion = 1.0 + 4.0 * np.random.random()  # 1.0-5.0 (variable)
        else:
            base_congestion = 5.0
        
        # Apply hotspot effects
        hotspot_effect = 0
        for hot_x, hot_y, radius, intensity in hotspots:
            # Calculate distance to hotspot
            dist = np.sqrt((edge_x - hot_x)**2 + (edge_y - hot_y)**2)
            
            # If within radius, apply effect with falloff
            if dist < radius:
                distance_factor = 1.0 - (dist / radius)
                
                # Calculate moderate effect based on scenario
                if scenario == "Morning":
                    effect = distance_factor * intensity * 1.0  # Reduced from 4.0
                elif scenario == "Evening":
                    effect = distance_factor * intensity * 1.2  # Reduced from 3.0
                elif scenario == "Weekend":
                    effect = distance_factor * intensity * 0.8  # Reduced from 5.0
                elif scenario == "Special":
                    effect = distance_factor * intensity * 1.5  # Reduced from 7.0
                else:  # Normal
                    effect = distance_factor * intensity * 0.5  # Reduced from 6.0
                
                hotspot_effect = max(hotspot_effect, effect)
        
        # Calculate final congestion
        new_congestion = base_congestion + hotspot_effect
        
        # Add some randomization
        new_congestion += np.random.normal(0, 0.5)
        
        # Ensure values are in a realistic 1-5 range for better performance
        new_congestion = max(1.0, min(5.0, new_congestion))
        
        # Update congestion
        scenario_result_congestion[edge_id] = new_congestion
        G[u][v][k]['congestion'] = new_congestion
        edges_updated += 1
    
    print(f"  Updated {edges_updated} edges with randomly distributed congestion values.")
    
    # Print statistics
    temp_vals = list(scenario_result_congestion.values())
    if temp_vals:
        print(f"  Congestion stats: Mean={np.mean(temp_vals):.2f}, Min={np.min(temp_vals):.2f}, Max={np.max(temp_vals):.2f}")
    
    # Export congestion data to Excel
    excel_file, df = export_congestion_to_excel(G, scenario_result_congestion, scenario)
    
    # Reset the random seed
    np.random.seed(42)
    
    return scenario_result_congestion, excel_file


def export_congestion_to_excel(G: nx.Graph, congestion_data: Dict[str, float], scenario: str) -> Tuple[str, pd.DataFrame]:
    """
    Export congestion data to an Excel file.
    
    Args:
        G: NetworkX graph
        congestion_data: Congestion data dictionary
        scenario: Scenario name
        
    Returns:
        Tuple of (excel_file_path, dataframe)
    """
    print(f"  Exporting congestion data for '{scenario}' scenario to Excel...")
    
    # Create output directory
    excel_dir = os.path.join('london_simulation', 'excel_data')
    os.makedirs(excel_dir, exist_ok=True)
    
    # Create a DataFrame to store the data
    data = []
    for u, v, k in G.edges(keys=True):
        edge_id = f"{u}_{v}_{k}"
        if edge_id in congestion_data:
            # Get edge data
            congestion = congestion_data[edge_id]
            length = G[u][v][k].get('length', 0)
            name = G[u][v][k].get('name', '')
            
            # Get geographic coordinates if available
            u_x, u_y = G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)
            v_x, v_y = G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)
            
            # Add to data list
            data.append({
                'Edge ID': edge_id,
                'Source Node': u,
                'Target Node': v,
                'Key': k,
                'Street Name': name,
                'Source X': u_x,
                'Source Y': u_y,
                'Target X': v_x,
                'Target Y': v_y,
                'Length (m)': length,
                'Congestion Level': congestion
            })
    
    # Create DataFrame and sort by congestion level
    df = pd.DataFrame(data)
    df = df.sort_values('Congestion Level', ascending=False)
    
    # Calculate congestion level distribution
    low = sum(1 for level in df['Congestion Level'] if level <= 3.0)
    medium = sum(1 for level in df['Congestion Level'] if 3.0 < level <= 6.0)
    high = sum(1 for level in df['Congestion Level'] if level > 6.0)
    total = len(df)
    
    print(f"  Congestion distribution:")
    print(f"    Low (1-3): {low} edges ({low/total*100:.1f}%)")
    print(f"    Medium (4-6): {medium} edges ({medium/total*100:.1f}%)")
    print(f"    High (7-10): {high} edges ({high/total*100:.1f}%)")
    
    # Save to Excel file
    timestamp = int(time.time())
    excel_file = os.path.join(excel_dir, f"congestion_{scenario.replace(' ', '_')}_{timestamp}.xlsx")
    df.to_excel(excel_file, index=False)
    
    print(f"  Saved to {excel_file}")
    return excel_file, df


def update_congestion_based_on_vehicles(G: nx.Graph, congestion_data: Dict[str, float], 
                                       base_congestion: Dict[str, float]) -> Dict[str, float]:
    """
    Update congestion based on vehicle counts using MM1 queuing model with fixed service rate logic.
    
    Args:
        G: NetworkX graph
        congestion_data: Current congestion data
        base_congestion: Base congestion data
        
    Returns:
        Updated congestion data
    """
    updated_edges = 0
    
    for u, v, k in G.edges(keys=True):
        edge_id = f"{u}_{v}_{k}"
        
        if edge_id in base_congestion:
            # Get the base congestion and vehicle count
            base_value = base_congestion[edge_id]
            vehicle_count = G[u][v][k].get('vehicle_count', 0)
            
            # Determine road capacity based on road type
            length = G[u][v][k].get('length', 100)
            
            # Base capacity and service rate based on road size
            if length > 300:  # Major road
                base_capacity = 20
                base_service_rate = 10.0
            elif length > 100:  # Medium road
                base_capacity = 10
                base_service_rate = 6.0
            else:  # Small road
                base_capacity = 5
                base_service_rate = 3.0
            
            # Higher congestion = LOWER service rate (more moderate impact)
            congestion_impact = base_value / 10.0
            degradation_factor = 0.1 + (0.4 * congestion_impact)  # Reduced impact
            adjusted_service_rate = base_service_rate * (1.0 - degradation_factor + 0.1)
            
            # Ensure service rate doesn't go below minimum threshold (increased minimum)
            min_service_rate = base_service_rate * 0.5  # Increased from 0.2 to 0.5
            adjusted_service_rate = max(min_service_rate, adjusted_service_rate)
            
            # Calculate arrival rate based on actual vehicle count (more moderate)
            if vehicle_count > 0:
                arrival_rate = vehicle_count * 0.2  # Reduced from 0.5 to 0.2
                
                # Check if system is overloaded
                if arrival_rate >= adjusted_service_rate:
                    # System is overloaded
                    new_congestion = 10.0
                    
                    # Store overload statistics
                    G[u][v][k]['mm1_stats'] = {
                        'utilization': float('inf'),
                        'avg_vehicles_in_system': float('inf'),
                        'avg_vehicles_in_queue': float('inf'),
                        'avg_time_in_system': float('inf'),
                        'avg_time_in_queue': float('inf'),
                        'system_stable': False,
                        'overloaded': True,
                        'vehicle_count': vehicle_count,
                        'base_capacity': base_capacity,
                        'base_service_rate': base_service_rate,
                        'adjusted_service_rate': adjusted_service_rate,
                        'arrival_rate': arrival_rate,
                        'degradation_factor': degradation_factor
                    }
                else:
                    # System is stable
                    new_congestion = calculate_mm1_congestion(arrival_rate, adjusted_service_rate, base_value)
                    
                    # Store MM1 statistics
                    mm1_stats = get_mm1_statistics(arrival_rate, adjusted_service_rate)
                    mm1_stats.update({
                        'overloaded': False,
                        'vehicle_count': vehicle_count,
                        'base_capacity': base_capacity,
                        'base_service_rate': base_service_rate,
                        'adjusted_service_rate': adjusted_service_rate,
                        'arrival_rate': arrival_rate,
                        'degradation_factor': degradation_factor
                    })
                    G[u][v][k]['mm1_stats'] = mm1_stats
            else:
                # No vehicles on this edge
                new_congestion = base_value
                G[u][v][k]['mm1_stats'] = None
            
            # Update congestion
            congestion_data[edge_id] = new_congestion
            G[u][v][k]['congestion'] = new_congestion
            updated_edges += 1
    
    print(f"Updated congestion for {updated_edges} edges using FIXED service rate logic")
    return congestion_data


def display_mm1_queueing_statistics(G: nx.Graph, edge_id: Optional[str] = None, 
                                   node1: Optional[int] = None, node2: Optional[int] = None) -> None:
    """
    Display detailed MM1 queuing statistics for an edge.
    
    Args:
        G: NetworkX graph
        edge_id: Edge ID string (format: node1_node2_key)
        node1: Source node (if edge_id not provided)
        node2: Target node (if edge_id not provided)
    """
    # If edge_id is not provided, try to find it from nodes
    if not edge_id and node1 is not None and node2 is not None:
        if node1 in G and node2 in G[node1]:
            k = list(G[node1][node2].keys())[0]
            edge_id = f"{node1}_{node2}_{k}"
    
    if not edge_id:
        print("No valid edge specified")
        return
    
    # Parse edge_id to get nodes and key
    parts = edge_id.split('_')
    if len(parts) != 3:
        print(f"Invalid edge ID format: {edge_id}")
        return
    
    u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
    
    # Check if edge exists
    if not (u in G and v in G[u] and k in G[u][v]):
        print(f"Edge {edge_id} not found in graph")
        return
    
    # Get MM1 statistics
    mm1_stats = G[u][v][k].get('mm1_stats')
    
    if not mm1_stats:
        vehicle_count = G[u][v][k].get('vehicle_count', 0)
        congestion = G[u][v][k].get('congestion', 1.0)
        
        print(f"\nMM1 Statistics for Edge {edge_id}:")
        print(f"  Vehicle count: {vehicle_count}")
        print(f"  Congestion level: {congestion:.2f}")
        print("  No MM1 queuing model statistics available (not enough vehicles)")
        return
    
    # Display MM1 statistics
    print(f"\nMM1 Queuing Statistics for Edge {edge_id}:")
    
    # Show overload status
    if mm1_stats.get('overloaded', False):
        print("  🔴 STATUS: SYSTEM OVERLOADED - TOO MANY VEHICLES!")
        print(f"  Vehicles on edge: {mm1_stats.get('vehicle_count', 0)}")
        print(f"  Base road capacity: {mm1_stats.get('base_capacity', 0)}")
        
        # Show service rate degradation details
        base_service = mm1_stats.get('base_service_rate', 0)
        adjusted_service = mm1_stats.get('adjusted_service_rate', 0)
        degradation = mm1_stats.get('degradation_factor', 0) * 100
        
        print(f"  Base service rate: {base_service:.2f} vehicles/time")
        print(f"  Degraded service rate: {adjusted_service:.2f} vehicles/time")
        print(f"  Service degradation: {degradation:.1f}% (due to base congestion)")
        print(f"  Arrival rate: {mm1_stats.get('arrival_rate', 0):.2f} vehicles/time")
        
    elif not mm1_stats['system_stable']:
        print("  ⚠️  WARNING: System is UNSTABLE - demand exceeds capacity!")
    else:
        print("  ✅ System is stable and operating normally")
        
        # Show service rate details for stable systems too
        if 'base_service_rate' in mm1_stats:
            base_service = mm1_stats.get('base_service_rate', 0)
            adjusted_service = mm1_stats.get('adjusted_service_rate', 0)
            degradation = mm1_stats.get('degradation_factor', 0) * 100
            
            print(f"  Base service rate: {base_service:.2f} vehicles/time")
            print(f"  Current service rate: {adjusted_service:.2f} vehicles/time")
            print(f"  Service degradation: {degradation:.1f}% (due to base congestion)")
    
    if mm1_stats['system_stable']:
        print(f"  Utilization (ρ): {mm1_stats['utilization']:.2f}")
        print(f"  Average vehicles in system (L): {mm1_stats['avg_vehicles_in_system']:.2f}")
        print(f"  Average vehicles in queue (Lq): {mm1_stats['avg_vehicles_in_queue']:.2f}")
        print(f"  Average time in system (W): {mm1_stats['avg_time_in_system']:.2f}")
        print(f"  Average time in queue (Wq): {mm1_stats['avg_time_in_queue']:.2f}")
    
    # Get additional edge information
    length = G[u][v][k].get('length', 0)
    congestion = G[u][v][k].get('congestion', 0)
    vehicle_count = G[u][v][k].get('vehicle_count', 0)
    
    print(f"\nAdditional Edge Information:")
    print(f"  Length: {length:.2f} meters")
    print(f"  Current congestion level: {congestion:.2f}")
    print(f"  Vehicle count: {vehicle_count}")


def print_service_rate_degradation_summary(G: nx.Graph, congestion_data: Dict[str, float]) -> None:
    """
    Print a summary of how service rates are being degraded across the network.
    
    Args:
        G: NetworkX graph
        congestion_data: Current congestion data
    """
    print("\n=== Service Rate Degradation Summary ===")
    
    degradation_stats = []
    for u, v, k in G.edges(keys=True):
        mm1_stats = G[u][v][k].get('mm1_stats')
        if mm1_stats and 'degradation_factor' in mm1_stats:
            degradation_stats.append({
                'edge_id': f"{u}_{v}_{k}",
                'base_service': mm1_stats.get('base_service_rate', 0),
                'adjusted_service': mm1_stats.get('adjusted_service_rate', 0),
                'degradation_pct': mm1_stats.get('degradation_factor', 0) * 100,
                'vehicle_count': mm1_stats.get('vehicle_count', 0),
                'overloaded': mm1_stats.get('overloaded', False)
            })
    
    if not degradation_stats:
        print("No service rate degradation data available")
        return
    
    # Sort by degradation percentage
    degradation_stats.sort(key=lambda x: x['degradation_pct'], reverse=True)
    
    print(f"Analyzed {len(degradation_stats)} edges with vehicle traffic:")
    print("\nTop 10 Most Degraded Roads:")
    print("Edge ID        | Base Rate | Current Rate | Degradation | Vehicles | Status")
    print("-" * 75)
    
    for i, stat in enumerate(degradation_stats[:10]):
        status = "OVERLOADED" if stat['overloaded'] else "Stable"
        print(f"{stat['edge_id']:<14} | {stat['base_service']:<9.1f} | {stat['adjusted_service']:<12.1f} | {stat['degradation_pct']:<11.1f}% | {stat['vehicle_count']:<8} | {status}")
    
    # Summary statistics
    avg_degradation = sum(s['degradation_pct'] for s in degradation_stats) / len(degradation_stats)
    overloaded_count = sum(1 for s in degradation_stats if s['overloaded'])
    
    print(f"\nSummary:")
    print(f"  Average service degradation: {avg_degradation:.1f}%")
    print(f"  Overloaded edges: {overloaded_count} out of {len(degradation_stats)}")
    print(f"  Network efficiency: {100 - avg_degradation:.1f}%")