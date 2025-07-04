"""
Routing algorithms for the traffic simulation project.
Includes A*, Shortest Path, and Shortest Path Congestion Aware algorithms.
Updated to use unified travel time calculation system for realistic and consistent results.
"""

import heapq
import time
import networkx as nx
import numpy as np
from typing import Tuple, Optional, List, Dict

from models import Vehicle, get_edge_travel_time, get_base_travel_time
from unified_travel_time import UnifiedTravelTimeCalculator, TravelTimeValidator


def enhanced_a_star_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Enhanced A* algorithm that finds optimal path with priority: Congestion > Travel Time > Distance.
    This algorithm considers congestion as the primary factor, then travel time, then distance.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    calc = UnifiedTravelTimeCalculator()
    
    # Get node positions for heuristic
    pos = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            pos[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            pos[node] = (0, 0)
    
    # Define multi-criteria heuristic function
    def multi_criteria_heuristic(n1: int, n2: int) -> Tuple[float, float, float]:
        """
        Returns (congestion_penalty, travel_time_estimate, distance_estimate)
        Priority: Congestion > Travel Time > Distance
        """
        # Distance estimate (Euclidean)
        if n1 in pos and n2 in pos:
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            distance_estimate = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        else:
            distance_estimate = 0
        
        # Estimate travel time based on average speed (rough heuristic)
        avg_speed_ms = 50 / 3.6  # 50 km/h in m/s
        travel_time_estimate = distance_estimate / avg_speed_ms if avg_speed_ms > 0 else 0
        
        # Congestion penalty estimate (assume moderate congestion for heuristic)
        congestion_penalty = 2.0  # Moderate congestion assumption
        
        return (congestion_penalty, travel_time_estimate, distance_estimate)
    
    # Initialize data structures with multi-criteria costs
    # g_score stores (congestion_cost, travel_time_cost, distance_cost)
    g_score = {node: (float('infinity'), float('infinity'), float('infinity')) for node in G.nodes()}
    g_score[start] = (0, 0, 0)
    
    # f_score combines g_score and heuristic
    f_score = {node: (float('infinity'), float('infinity'), float('infinity')) for node in G.nodes()}
    heuristic_start = multi_criteria_heuristic(start, end)
    f_score[start] = heuristic_start
    
    # Track previous nodes for path reconstruction
    previous = {node: None for node in G.nodes()}
    
    # Priority queue for nodes to visit - use tuple comparison for multi-criteria
    open_set = [(f_score[start], start)]
    closed_set = set()
    
    while open_set:
        # Get node with best multi-criteria score
        _, current_node = heapq.heappop(open_set)
        
        # If we've reached the end, we're done
        if current_node == end:
            break
        
        # Skip if already processed
        if current_node in closed_set:
            continue
            
        closed_set.add(current_node)
        
        # Check all neighbors
        for neighbor in G.neighbors(current_node):
            if neighbor in closed_set:
                continue
            
            # Calculate edge costs with priority: Congestion > Travel Time > Distance
            edge_congestion_cost = 0
            edge_travel_time = 0
            edge_distance = 0
            
            if current_node in G and neighbor in G[current_node]:
                edge_keys = list(G[current_node][neighbor].keys())
                if edge_keys:
                    k = edge_keys[0]  # Use first edge key
                    edge_data = G[current_node][neighbor][k]
                    
                    # Get congestion level (primary factor)
                    congestion_level = edge_data.get('congestion', 1.0)
                    edge_congestion_cost = congestion_level
                    
                    # Get travel time (secondary factor)
                    try:
                        edge_travel_time = calc.calculate_edge_travel_time(
                            G, current_node, neighbor, k, apply_congestion=True
                        )
                    except (KeyError, ValueError):
                        edge_travel_time = 60.0
                    
                    # Get distance (tertiary factor)
                    edge_distance = edge_data.get('length', 100)
                else:
                    edge_congestion_cost = 5.0  # High penalty for missing data
                    edge_travel_time = 60.0
                    edge_distance = 100
            else:
                edge_congestion_cost = 5.0  # High penalty for missing edge
                edge_travel_time = 60.0
                edge_distance = 100
            
            # Calculate new g_score (cumulative costs)
            current_g = g_score[current_node]
            tentative_g_score = (
                current_g[0] + edge_congestion_cost,      # Congestion (primary)
                current_g[1] + edge_travel_time,          # Travel time (secondary)
                current_g[2] + edge_distance              # Distance (tertiary)
            )
            
            # Update if this is a better path (lexicographic comparison)
            if tentative_g_score < g_score[neighbor]:
                previous[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                
                # Calculate f_score with heuristic
                heuristic_neighbor = multi_criteria_heuristic(neighbor, end)
                f_score[neighbor] = (
                    tentative_g_score[0] + heuristic_neighbor[0],
                    tentative_g_score[1] + heuristic_neighbor[1],
                    tentative_g_score[2] + heuristic_neighbor[2]
                )
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    if g_score[end][0] == float('infinity'):
        print(f"No path found from {start} to {end} using Enhanced A* algorithm")
        return None, float('inf'), 0
    
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    # Calculate final travel time using unified system
    final_travel_time = calc.calculate_path_travel_time(G, path, apply_congestion=True)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return path, final_travel_time, computation_time


def enhanced_dijkstra_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Enhanced Dijkstra's algorithm using unified travel time calculation system.
    Finds optimal path considering realistic congestion penalties.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    calc = UnifiedTravelTimeCalculator()
    
    # Initialize data structures
    distances = {node: float('infinity') for node in G.nodes()}
    distances[start] = 0
    
    # Track previous nodes for path reconstruction
    previous = {node: None for node in G.nodes()}
    
    # Priority queue for nodes to visit (distance, node)
    pq = [(0, start)]
    visited = set()
    
    while pq:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(pq)
        
        # If we've reached the end, we're done
        if current_node == end:
            break
        
        # Skip if already visited
        if current_node in visited:
            continue
        
        visited.add(current_node)
        
        # Check all neighbors
        for neighbor in G.neighbors(current_node):
            if neighbor in visited:
                continue
            
            # Use unified travel time calculation with congestion
            edge_travel_time = 0
            if current_node in G and neighbor in G[current_node]:
                edge_keys = list(G[current_node][neighbor].keys())
                if edge_keys:
                    k = edge_keys[0]  # Use first edge key
                    try:
                        edge_travel_time = calc.calculate_edge_travel_time(
                            G, current_node, neighbor, k, apply_congestion=True
                        )
                    except (KeyError, ValueError):
                        # Fallback to default time
                        edge_travel_time = 60.0
                else:
                    edge_travel_time = 60.0
            else:
                edge_travel_time = 60.0
            
            # Calculate new distance
            new_distance = distances[current_node] + edge_travel_time
            
            # Update if this is a better path
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (new_distance, neighbor))
    
    # Reconstruct path
    if distances[end] == float('infinity'):
        print(f"No path found from {start} to {end} using Enhanced Dijkstra's algorithm")
        return None, float('inf'), 0
    
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    # Calculate final travel time using unified system
    final_travel_time = calc.calculate_path_travel_time(G, path, apply_congestion=True)
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return path, final_travel_time, computation_time


def shortest_path_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Shortest Path Algorithm using unified travel time calculation system.
    Finds the shortest path without considering traffic/congestion.
    
    IMPORTANT: This algorithm always returns the SAME result for the same route,
    regardless of congestion, vehicle counts, or scenarios. It uses base travel times only.
    
    Args:
        G: NetworkX graph
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    calc = UnifiedTravelTimeCalculator()
    
    try:
        # Use NetworkX shortest path by length (ignoring congestion)
        path = nx.shortest_path(G, start, end, weight='length')
        
        # Calculate travel time WITHOUT congestion using unified system
        final_travel_time = calc.calculate_path_travel_time(G, path, apply_congestion=False)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return path, final_travel_time, computation_time
        
    except nx.NetworkXNoPath:
        print(f"No path found from {start} to {end} using Shortest Path algorithm")
        end_time = time.time()
        computation_time = end_time - start_time
        return None, float('inf'), computation_time
    except Exception as e:
        print(f"Error in shortest path calculation: {e}")
        end_time = time.time()
        computation_time = end_time - start_time
        return None, float('inf'), computation_time


def create_congestion_graph(G: nx.Graph, congestion_data: Dict[str, float], time_band: int = 0) -> nx.DiGraph:
    """
    Create a graph with travel times adjusted for congestion.
    
    Args:
        G: Original NetworkX graph
        congestion_data: Dictionary mapping edge_id to congestion factor
        time_band: Time band for congestion (unused in current implementation)
        
    Returns:
        New directed graph with congestion-adjusted travel times
    """
    G_congestion = nx.DiGraph()
    
    for u, v, k, data in G.edges(keys=True, data=True):
        edge_id = f"{u}_{v}_{k}"
        
        # Get base travel time (length/speed)
        length = data.get('length', 100)
        speed = data.get('speed', 50)
        base_time = length / speed * 3.6  # Convert to seconds
        
        # Apply congestion factor
        congestion_factor = congestion_data.get(edge_id, 1.0)

        if congestion_factor > 5.0:
            travel_time = base_time * (congestion_factor ** 2) / 5.0
        else:
            travel_time = base_time * congestion_factor
        
        # Add edge to the congestion graph
        G_congestion.add_edge(u, v, 
                             travel_time=travel_time, 
                             congestion=congestion_factor,
                             length=length,
                             base_travel_time=base_time)
    
    # Copy node attributes
    for node, data in G.nodes(data=True):
        if node in G_congestion.nodes():
            for key, value in data.items():
                G_congestion.nodes[node][key] = value
    
    return G_congestion


def shortest_path_congestion_aware_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Shortest Path Congestion Aware Algorithm using unified travel time calculation system.
    Finds the SAME path as shortest path but with congestion-adjusted travel times.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    calc = UnifiedTravelTimeCalculator()
    
    try:
        # Use NetworkX shortest path by length (same as shortest path algorithm)
        path = nx.shortest_path(G, start, end, weight='length')
        
        # Calculate travel time WITH congestion using unified system
        final_travel_time = calc.calculate_path_travel_time(G, path, apply_congestion=True)
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return path, final_travel_time, computation_time
        
    except nx.NetworkXNoPath:
        print(f"No path found from {start} to {end} using Shortest Path Congestion Aware algorithm")
        end_time = time.time()
        computation_time = end_time - start_time
        return None, float('inf'), computation_time
    except Exception as e:
        print(f"Error in shortest path congestion aware calculation: {e}")
        end_time = time.time()
        computation_time = end_time - start_time
        return None, float('inf'), computation_time


def calculate_all_routes(G: nx.Graph, vehicle: Vehicle, congestion_data: Dict[str, float]) -> Vehicle:
    """
    Calculate routes using A*, Shortest Path, and Shortest Path Congestion Aware algorithms for a specific vehicle.
    
    Args:
        G: NetworkX graph
        vehicle: Vehicle object to calculate routes for
        congestion_data: Dictionary of congestion values
        
    Returns:
        Updated vehicle object with calculated routes
    """
    print(f"Calculating routes for Vehicle {vehicle.id}...")
    
    source = vehicle.source
    destination = vehicle.destination
    
    # 1. Enhanced A* (use original graph, A* handles congestion internally)
    astar_path, astar_time, astar_comp_time = enhanced_a_star_algorithm(G, source, destination)
    if astar_path:
        vehicle.paths['A*'] = astar_path
        vehicle.travel_times['A*'] = astar_time
        vehicle.computation_times['A*'] = astar_comp_time
        
        # Calculate service rate for A*
        astar_service_rate = 1 / astar_comp_time if astar_comp_time > 0 else float('inf')
        vehicle.service_rates['A*'] = astar_service_rate
        
        print(f"  A*: {len(astar_path)} nodes, travel time: {astar_time:.2f}s, computation time: {astar_comp_time:.6f}s, service rate: {astar_service_rate:.2f} routes/sec")
    
    # 2. Shortest Path
    shortest_path, shortest_travel_time, shortest_comp_time = shortest_path_algorithm(G, source, destination)
    if shortest_path:
        vehicle.paths['Shortest Path'] = shortest_path
        vehicle.travel_times['Shortest Path'] = shortest_travel_time  # Now stores travel time consistently
        vehicle.computation_times['Shortest Path'] = shortest_comp_time
        
        # Calculate service rate for Shortest Path
        shortest_service_rate = 1 / shortest_comp_time if shortest_comp_time > 0 else float('inf')
        vehicle.service_rates['Shortest Path'] = shortest_service_rate
        
        print(f"  Shortest Path: {len(shortest_path)} nodes, base travel time: {shortest_travel_time:.2f}s (no congestion), computation time: {shortest_comp_time:.6f}s, service rate: {shortest_service_rate:.2f} routes/sec")
    
    # 3. Shortest Path Congestion Aware
    cong_aware_path, cong_aware_travel_time, cong_aware_comp_time = shortest_path_congestion_aware_algorithm(G, source, destination)
    if cong_aware_path:
        vehicle.paths['Shortest Path Congestion Aware'] = cong_aware_path
        vehicle.travel_times['Shortest Path Congestion Aware'] = cong_aware_travel_time
        vehicle.computation_times['Shortest Path Congestion Aware'] = cong_aware_comp_time
        
        # Calculate service rate for Shortest Path Congestion Aware
        cong_aware_service_rate = 1 / cong_aware_comp_time if cong_aware_comp_time > 0 else float('inf')
        vehicle.service_rates['Shortest Path Congestion Aware'] = cong_aware_service_rate
        
        print(f"  Shortest Path Congestion Aware: {len(cong_aware_path)} nodes, travel time: {cong_aware_travel_time:.2f}s, computation time: {cong_aware_comp_time:.6f}s, service rate: {cong_aware_service_rate:.2f} routes/sec")
    
    # Set the default path to A* (or the best available)
    if 'A*' in vehicle.paths:
        vehicle.path = vehicle.paths['A*']
    elif 'Shortest Path Congestion Aware' in vehicle.paths:
        vehicle.path = vehicle.paths['Shortest Path Congestion Aware']
    elif 'Shortest Path' in vehicle.paths:
        vehicle.path = vehicle.paths['Shortest Path']
    
    print(f"  A* and Shortest Path Congestion Aware consider congestion, Shortest Path ignores congestion completely")
    
    return vehicle
