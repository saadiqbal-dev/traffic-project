"""
Vehicle management module for the traffic simulation project.
Handles adding, tracking, and managing vehicles in the simulation.
"""

import random
import numpy as np
from typing import List, Dict, Optional, Tuple

from models import Vehicle, calculate_shortest_path, update_vehicle_counts_for_path
from routing import calculate_all_routes
from congestion import update_congestion_based_on_vehicles


def add_vehicle(G, vehicles: List[Vehicle], source: int, destination: int, 
               congestion_data: Optional[Dict[str, float]] = None, 
               calculate_routes: bool = False) -> Optional[Vehicle]:
    """
    Add a single vehicle with specific source and destination.
    
    Args:
        G: NetworkX graph
        vehicles: List of existing vehicles
        source: Source node
        destination: Destination node
        congestion_data: Optional congestion data for route calculation
        calculate_routes: Whether to calculate routes using all algorithms
        
    Returns:
        Created Vehicle object or None if failed
    """
    # Generate a unique ID for the vehicle
    vehicle_id = len(vehicles) + 1
    
    # Calculate the shortest path
    path = calculate_shortest_path(G, source, destination)
    
    if path is None:
        print(f"Could not add vehicle {vehicle_id}: No valid path")
        return None
    
    # Create the vehicle
    vehicle = Vehicle(vehicle_id, source, destination, path)
    
    # Calculate routes using all algorithms if requested
    if calculate_routes and congestion_data:
        vehicle = calculate_all_routes(G, vehicle, congestion_data)
    
    # Add vehicle to the collection
    vehicles.append(vehicle)
    
    # Update vehicle count on the path
    update_vehicle_counts_for_path(G, path, 1)
    
    print(f"Added vehicle {vehicle_id} from {source} to {destination} (path length: {len(path)} nodes)")
    return vehicle


def add_multiple_vehicles_manual(G, vehicles: List[Vehicle], source: int, destination: int, 
                                count: int, congestion_data: Optional[Dict[str, float]] = None,
                                calculate_routes: bool = False) -> int:
    """
    Add a specified number of vehicles with the same source and destination.
    
    Args:
        G: NetworkX graph
        vehicles: List of existing vehicles
        source: Source node
        destination: Destination node
        count: Number of vehicles to add
        congestion_data: Optional congestion data for route calculation
        calculate_routes: Whether to calculate routes using all algorithms
        
    Returns:
        Number of vehicles successfully added
    """
    print(f"Adding {count} vehicles from {source} to {destination}...")
    
    # Calculate the shortest path to check if valid
    path = calculate_shortest_path(G, source, destination)
    
    if path is None:
        print(f"Could not add vehicles: No valid path between {source} and {destination}")
        return 0
    
    added_count = 0
    
    for _ in range(count):
        # Generate a unique ID for the vehicle
        vehicle_id = len(vehicles) + 1
        
        # Create the vehicle
        vehicle = Vehicle(vehicle_id, source, destination, path)
        
        # Calculate routes using all algorithms if requested
        if calculate_routes and congestion_data:
            vehicle = calculate_all_routes(G, vehicle, congestion_data)
        
        # Add vehicle to the collection
        vehicles.append(vehicle)
        
        # Update vehicle count on the path
        update_vehicle_counts_for_path(G, path, 1)
        
        added_count += 1
    
    print(f"Successfully added {added_count} vehicles from {source} to {destination}")
    return added_count


def add_bulk_vehicles(G, vehicles: List[Vehicle], count: int, 
                     congestion_data: Optional[Dict[str, float]] = None,
                     calculate_routes: bool = False, 
                     notable_locations: Optional[Dict[str, int]] = None) -> int:
    """
    Add a specified number of vehicles with random source and destination.
    
    Args:
        G: NetworkX graph
        vehicles: List of existing vehicles
        count: Number of vehicles to add
        congestion_data: Optional congestion data for route calculation
        calculate_routes: Whether to calculate routes using all algorithms
        notable_locations: Optional notable locations to bias selection
        
    Returns:
        Number of vehicles successfully added
    """
    print(f"Adding {count} vehicles with random routes...")
    
    nodes = list(G.nodes())
    added_count = 0
    
    # If notable locations provided, use them with higher probability
    if notable_locations and len(notable_locations) >= 2:
        notable_nodes = list(notable_locations.values())
    else:
        notable_nodes = []
    
    for _ in range(count):
        # With 70% probability, use notable locations if available
        if notable_nodes and random.random() < 0.7:
            source = random.choice(notable_nodes)
            
            # Make sure destination is different from source
            possible_destinations = [n for n in notable_nodes if n != source]
            if possible_destinations:
                destination = random.choice(possible_destinations)
            else:
                # Fall back to random node if needed
                destination = random.choice(nodes)
                while destination == source:
                    destination = random.choice(nodes)
        else:
            # Random source and destination
            source = random.choice(nodes)
            destination = random.choice(nodes)
            while destination == source:
                destination = random.choice(nodes)
        
        # Add the vehicle
        vehicle = add_vehicle(G, vehicles, source, destination, congestion_data, calculate_routes)
        
        if vehicle:
            added_count += 1
    
    print(f"Successfully added {added_count} vehicles")
    return added_count


def add_stress_test_vehicles(G, vehicles: List[Vehicle], original_vehicle: Vehicle, 
                           congestion_data: Dict[str, float], 
                           original_congestion: Dict[str, float], 
                           count: int = 20) -> Tuple[int, List[Vehicle]]:
    """
    Add random vehicles along the path of an existing vehicle to stress test routing.
    
    Args:
        G: NetworkX graph
        vehicles: List of existing vehicles
        original_vehicle: Vehicle whose route to stress test
        congestion_data: Current congestion data
        original_congestion: Original congestion baseline
        count: Number of stress test vehicles to add
        
    Returns:
        Tuple of (vehicles_added, list_of_added_vehicles)
    """
    if not original_vehicle or not original_vehicle.path:
        print("Original vehicle has no valid path for stress testing")
        return 0, []
    
    print(f"Adding {count} stress test vehicles along the route of Vehicle {original_vehicle.id}...")
    
    # Get nodes along the original path
    path_nodes = original_vehicle.path.copy()
    
    # Save a copy of the original path for comparison
    original_path = original_vehicle.path.copy()
    
    # Ensure we have enough nodes for random placement
    if len(path_nodes) < 4:
        # Extend with neighbors
        extended_nodes = set(path_nodes)
        for node in path_nodes:
            extended_nodes.update(list(G.neighbors(node)))
        path_nodes = list(extended_nodes)
    
    added_count = 0
    added_vehicles = []
    
    # Add vehicles with sources and destinations along the path
    for _ in range(count):
        if len(path_nodes) >= 2:
            # Select random source and destination from path nodes
            source = random.choice(path_nodes)
            # Ensure destination is different and preferably creates congestion on original path
            dest_candidates = [n for n in path_nodes if n != source]
            destination = random.choice(dest_candidates)
            
            # Calculate path for this stress test vehicle
            path = calculate_shortest_path(G, source, destination)
            
            if path:
                vehicle_id = len(vehicles) + 1
                vehicle = Vehicle(vehicle_id, source, destination, path)
                vehicles.append(vehicle)
                update_vehicle_counts_for_path(G, path, 1)
                added_count += 1
                added_vehicles.append(vehicle)
    
    print(f"Added {added_count} stress test vehicles")
    
    # Update congestion based on all the new vehicles
    update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
    
    # Now recalculate the route for the original vehicle
    print(f"Recalculating route for Vehicle {original_vehicle.id} under new congestion conditions...")
    calculate_all_routes(G, original_vehicle, congestion_data)
    
    # Check if the path changed
    new_path = original_vehicle.path
    if new_path != original_path:
        print(f"Route changed due to increased congestion!")
        print(f"  Original path length: {len(original_path)} nodes")
        print(f"  New path length: {len(new_path)} nodes")
    else:
        print("Route remained the same despite increased congestion")
    
    # Detailed path comparison AFTER recalculating
    print("\nDetailed path comparison:")
    
    # Only show first few and last few nodes if paths are long
    if len(original_path) > 10:
        orig_display = f"{original_path[:5]} ... {original_path[-5:]}"
        new_display = f"{new_path[:5]} ... {new_path[-5:]}" if len(new_path) > 10 else str(new_path)
        print(f"Original path: {orig_display}")
        print(f"New path: {new_display}")
    else:
        print(f"Original path: {original_path}")
        print(f"New path: {new_path}")
    
    # Check how many nodes are different
    diff_count = sum(1 for x, y in zip(original_path, new_path) if x != y)
    if len(original_path) != len(new_path):
        diff_count += abs(len(original_path) - len(new_path))
        
    diff_percentage = (diff_count / max(len(original_path), len(new_path))) * 100
    print(f"Route difference: {diff_percentage:.1f}% of nodes are different")
    
    # Check if travel times actually changed
    if 'A*' in original_vehicle.travel_times:
        print(f"Travel time impact: {original_vehicle.travel_times['A*']:.2f}s (new) vs previous (recalculate to get old)")
    
    return added_count, added_vehicles


def select_from_notable_locations(notable_locations: Dict[str, int], prompt: str) -> Optional[int]:
    """
    Helper function to select a location from the notable locations.
    
    Args:
        notable_locations: Dictionary mapping location names to node IDs
        prompt: Prompt to display to user
        
    Returns:
        Selected node ID or None if invalid selection
    """
    print("\n" + prompt)
    for i, (name, _) in enumerate(notable_locations.items()):
        print(f"{i+1}. {name}")
    
    try:
        choice = int(input("Select location (number): "))
        if 1 <= choice <= len(notable_locations):
            selected_name = list(notable_locations.keys())[choice-1]
            selected_node = list(notable_locations.values())[choice-1]
            print(f"Selected: {selected_name} (Node: {selected_node})")
            return selected_node
        else:
            print("Invalid choice.")
            return None
    except ValueError:
        print("Invalid input. Please enter a number.")
        return None


def calculate_overall_congestion_metrics(G, congestion_data: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate detailed metrics about the current congestion state.
    
    Args:
        G: NetworkX graph
        congestion_data: Dictionary of congestion values
        
    Returns:
        Dictionary with congestion metrics
    """
    congestion_values = list(congestion_data.values())
    return {
        "mean": np.mean(congestion_values),
        "median": np.median(congestion_values),
        "min": np.min(congestion_values),
        "max": np.max(congestion_values),
        "std": np.std(congestion_values),
        "high_congestion_pct": sum(1 for v in congestion_values if v > 7.0) / len(congestion_values) * 100,
        "medium_congestion_pct": sum(1 for v in congestion_values if 3.0 < v <= 7.0) / len(congestion_values) * 100,
        "low_congestion_pct": sum(1 for v in congestion_values if v <= 3.0) / len(congestion_values) * 100
    }


def track_vehicle_congestion_impact(G, vehicles: List[Vehicle], 
                                  congestion_data: Dict[str, float], 
                                  base_congestion: Dict[str, float]) -> Dict:
    """
    Track the impact of vehicles on congestion and travel times.
    
    Args:
        G: NetworkX graph
        vehicles: List of vehicles
        congestion_data: Current congestion data
        base_congestion: Base congestion data
        
    Returns:
        Dictionary with impact analysis results
    """
    # Calculate metrics before any adjustments
    base_metrics = calculate_overall_congestion_metrics(G, base_congestion)
    current_metrics = calculate_overall_congestion_metrics(G, congestion_data)
    
    # Calculate percentage increase in congestion
    pct_increase_mean = ((current_metrics["mean"] - base_metrics["mean"]) / base_metrics["mean"]) * 100
    
    # Count affected roads
    total_edges = len(congestion_data)
    affected_edges = sum(1 for u, v, k in G.edges(keys=True) if G[u][v][k].get('vehicle_count', 0) > 0)
    affected_pct = (affected_edges / total_edges) * 100
    
    # Calculate average vehicle count on affected roads
    vehicle_counts = [G[u][v][k].get('vehicle_count', 0) for u, v, k in G.edges(keys=True) 
                     if G[u][v][k].get('vehicle_count', 0) > 0]
    avg_vehicle_count = np.mean(vehicle_counts) if vehicle_counts else 0
    
    # Calculate congestion impact by road type (simple method based on edge length)
    short_roads = []  # Local roads (shorter)
    medium_roads = []  # Collectors
    long_roads = []   # Arterials/major roads (longer)
    
    for u, v, k in G.edges(keys=True):
        length = G[u][v][k].get('length', 0)
        vehicle_count = G[u][v][k].get('vehicle_count', 0)
        edge_id = f"{u}_{v}_{k}"
        
        if length <= 100:  # Short roads (under 100m)
            short_roads.append((edge_id, vehicle_count))
        elif length <= 300:  # Medium roads (100-300m)
            medium_roads.append((edge_id, vehicle_count))
        else:  # Long roads (over 300m)
            long_roads.append((edge_id, vehicle_count))
    
    # Calculate average vehicle count by road type
    avg_vehicles_short = np.mean([vc for _, vc in short_roads]) if short_roads else 0
    avg_vehicles_medium = np.mean([vc for _, vc in medium_roads]) if medium_roads else 0 
    avg_vehicles_long = np.mean([vc for _, vc in long_roads]) if long_roads else 0
    
    # Calculate the impact on travel times for hypothetical routes
    travel_time_impacts = []
    
    # Sample a few vehicles to analyze
    sample_size = min(5, len(vehicles))
    if sample_size > 0:
        sample_vehicles = random.sample(vehicles, sample_size) if len(vehicles) > sample_size else vehicles
        
        for vehicle in sample_vehicles:
            if vehicle.path and len(vehicle.path) > 1:
                # Calculate travel time with base congestion
                base_time = 0
                current_time = 0
                
                for i in range(len(vehicle.path) - 1):
                    u, v = vehicle.path[i], vehicle.path[i+1]
                    if u in G and v in G[u]:
                        for k in G[u][v]:
                            edge_id = f"{u}_{v}_{k}"
                            
                            # Get edge data
                            length = G[u][v][k].get('length', 100)
                            speed = G[u][v][k].get('speed', 50)
                            base_travel_time = length / speed * 3.6  # Convert to seconds
                            
                            # Calculate times with different congestion
                            if edge_id in base_congestion and edge_id in congestion_data:
                                base_congestion_factor = base_congestion[edge_id]
                                current_congestion_factor = congestion_data[edge_id]
                                
                                base_time += base_travel_time * base_congestion_factor
                                current_time += base_travel_time * current_congestion_factor
                
                # Calculate percentage increase
                if base_time > 0:
                    time_increase_pct = ((current_time - base_time) / base_time) * 100
                    travel_time_impacts.append({
                        'vehicle_id': vehicle.id,
                        'base_time': base_time,
                        'current_time': current_time,
                        'increase_pct': time_increase_pct
                    })
    
    # Calculate average time impact
    avg_time_increase_pct = np.mean([impact['increase_pct'] for impact in travel_time_impacts]) if travel_time_impacts else 0
    
    # Generate the impact report
    impact_report = {
        'vehicle_count': len(vehicles),
        'affected_edges': affected_edges,
        'affected_pct': affected_pct,
        'avg_vehicle_count': avg_vehicle_count,
        'congestion_increase_pct': pct_increase_mean,
        'avg_vehicles_short_roads': avg_vehicles_short,
        'avg_vehicles_medium_roads': avg_vehicles_medium,
        'avg_vehicles_long_roads': avg_vehicles_long,
        'avg_time_increase_pct': avg_time_increase_pct,
        'travel_time_impacts': travel_time_impacts
    }
    
    return impact_report


def print_vehicle_impact_report(impact_report: Dict) -> None:
    """
    Print a formatted report of the vehicle impact on congestion and travel times.
    
    Args:
        impact_report: Dictionary with impact analysis results
    """
    print("\n=== Vehicle Impact on Congestion Report ===")
    print(f"Total vehicles: {impact_report['vehicle_count']}")
    print(f"Roads affected by vehicles: {impact_report['affected_edges']} ({impact_report['affected_pct']:.2f}%)")
    print(f"Average vehicles per affected road: {impact_report['avg_vehicle_count']:.2f}")
    print(f"Overall congestion increase: {impact_report['congestion_increase_pct']:.2f}%")
    
    print("\nVehicle distribution by road type:")
    print(f"  Small roads (<=100m): {impact_report['avg_vehicles_short_roads']:.2f} vehicles on average")
    print(f"  Medium roads (100-300m): {impact_report['avg_vehicles_medium_roads']:.2f} vehicles on average")
    print(f"  Major roads (>300m): {impact_report['avg_vehicles_long_roads']:.2f} vehicles on average")
    
    print("\nImpact on travel times:")
    print(f"  Average increase in travel time: {impact_report['avg_time_increase_pct']:.2f}%")
    
    if impact_report['travel_time_impacts']:
        print("\nSample vehicle travel time impacts:")
        for impact in impact_report['travel_time_impacts']:
            print(f"  Vehicle {impact['vehicle_id']}: {impact['increase_pct']:.2f}% increase " +
                 f"({impact['base_time']:.2f}s → {impact['current_time']:.2f}s)")
    
    print("\nCongestion Severity Summary:")
    if impact_report['avg_time_increase_pct'] < 5:
        print("  🟢 Low Impact: Minimal effect on traffic flow")
    elif impact_report['avg_time_increase_pct'] < 15:
        print("  🟡 Moderate Impact: Noticeable slowdown in affected areas")
    elif impact_report['avg_time_increase_pct'] < 30:
        print("  🟠 High Impact: Significant delays on major routes")
    else:
        print("  🔴 Severe Impact: Traffic congestion causing major delays")