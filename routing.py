"""
Routing algorithms for the traffic simulation project.
Includes A*, Shortest Path, and Shortest Path Congestion Aware algorithms.
"""

import heapq
import time
import networkx as nx
import numpy as np
from typing import Tuple, Optional, List, Dict

from models import Vehicle, get_edge_travel_time, get_base_travel_time


def enhanced_a_star_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Enhanced A* algorithm using direct distance as heuristic and better congestion handling.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    
    # Get node positions for heuristic
    pos = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            pos[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            pos[node] = (0, 0)
    
    # Define heuristic function (Euclidean distance)
    def heuristic(n1: int, n2: int) -> float:
        if n1 in pos and n2 in pos:
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        else:
            return 0
    
    # Initialize data structures
    g_score = {node: float('infinity') for node in G.nodes()}
    g_score[start] = 0
    
    f_score = {node: float('infinity') for node in G.nodes()}
    f_score[start] = heuristic(start, end)
    
    # Track previous nodes for path reconstruction
    previous = {node: None for node in G.nodes()}
    
    # Priority queue for nodes to visit (f_score, node)
    open_set = [(f_score[start], start)]
    closed_set = set()
    
    while open_set:
        # Get node with smallest f_score
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
                
            # Calculate BASE travel time (no congestion)
            travel_time = get_base_travel_time(G, current_node, neighbor)
            # Get congestion from first edge key (multigraph support)
            congestion = 1.0
            if current_node in G and neighbor in G[current_node]:
                edge_keys = list(G[current_node][neighbor].keys())
                if edge_keys:
                    edge_data = G[current_node][neighbor][edge_keys[0]]
                    if isinstance(edge_data, dict):
                        congestion = edge_data.get('congestion', 1.0)
            
            # Apply REALISTIC London congestion penalties
            if congestion > 4.0:
                penalty = travel_time * 3.5  # Heavy traffic: 71% speed reduction (stop-and-go traffic)
            elif congestion > 3.0:
                penalty = travel_time * 2.2  # Moderate traffic: 55% speed reduction (heavy congestion)
            elif congestion > 2.0:
                penalty = travel_time * 1.5  # Light traffic: 33% speed reduction (slow traffic)
            else:
                penalty = travel_time        # Free flow
            
            # Calculate new g_score
            tentative_g_score = g_score[current_node] + penalty
            
            # Update if this is a better path
            if tentative_g_score < g_score[neighbor]:
                previous[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    if g_score[end] == float('infinity'):
        print(f"No path found from {start} to {end} using Enhanced A* algorithm")
        return None, float('inf'), 0
    
    path = []
    current = end
    while current:
        path.append(current)
        current = previous[current]
    path.reverse()
    
    # Return the g_score as travel time since congestion was already applied during pathfinding
    # This avoids the DOUBLE PENALTY problem where congestion gets applied twice
    path_length = g_score[end]
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return path, path_length, computation_time


def enhanced_dijkstra_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Enhanced Dijkstra's algorithm that is more sensitive to high congestion values.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    
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
                
            # Calculate BASE travel time (no congestion)
            travel_time = get_base_travel_time(G, current_node, neighbor)
            congestion = G[current_node][neighbor].get('congestion', 1.0)
            
            # Apply REALISTIC London congestion penalties
            if congestion > 4.0:
                penalty = travel_time * 3.5  # Heavy traffic: 71% speed reduction (stop-and-go traffic)
            elif congestion > 3.0:
                penalty = travel_time * 2.2  # Moderate traffic: 55% speed reduction (heavy congestion)
            elif congestion > 2.0:
                penalty = travel_time * 1.5  # Light traffic: 33% speed reduction (slow traffic)
            else:
                penalty = travel_time        # Free flow
            
            # Calculate new distance
            new_distance = distances[current_node] + penalty
            
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
    
    # Calculate actual path travel time WITH congestion (the realistic time)
    path_length = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        base_time = get_edge_travel_time(G, u, v)
        # Get congestion from first edge key (multigraph support)
        congestion = 1.0
        if u in G and v in G[u]:
            edge_keys = list(G[u][v].keys())
            if edge_keys:
                edge_data = G[u][v][edge_keys[0]]
                if isinstance(edge_data, dict):
                    congestion = edge_data.get('congestion', 1.0)
        
        # Apply the same REALISTIC London congestion penalty used in pathfinding
        if congestion > 4.0:
            realistic_time = base_time * 3.5  # Heavy traffic: 71% speed reduction (stop-and-go traffic)
        elif congestion > 3.0:
            realistic_time = base_time * 2.2  # Moderate traffic: 55% speed reduction (heavy congestion)
        elif congestion > 2.0:
            realistic_time = base_time * 1.5  # Light traffic: 33% speed reduction (slow traffic)
        else:
            realistic_time = base_time        # Free flow
        
        path_length += realistic_time
    
    end_time = time.time()
    computation_time = end_time - start_time
    
    return path, path_length, computation_time


def shortest_path_algorithm(G: nx.Graph, start: int, end: int) -> Tuple[Optional[List[int]], float, float]:
    """
    Shortest Path Algorithm - finds the shortest path without considering traffic/congestion.
    This algorithm finds the shortest possible path from A to B ignoring all traffic conditions.
    
    IMPORTANT: This algorithm always returns the SAME result for the same route,
    regardless of congestion, vehicle counts, or scenarios. It uses original
    road speeds only, never modified values.
    
    Args:
        G: NetworkX graph
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    
    # Get node positions for heuristic
    pos = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            pos[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            pos[node] = (0, 0)
    
    # Define heuristic function (Euclidean distance)
    def heuristic(n1: int, n2: int) -> float:
        if n1 in pos and n2 in pos:
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        else:
            return 0
    
    # Implementation of Simple Shortest Path Search (ignores congestion completely)
    visited = set()
    queue = [(heuristic(start, end), start, [start], 0)]
    heapq.heapify(queue)
    
    while queue:
        _, current, path, cost = heapq.heappop(queue)
        
        if current == end:
            # Calculate base travel time WITHOUT any congestion effects
            base_travel_time = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                # Calculate proper base travel time from length/speed
                edge_data = None
                if u in G and v in G[u]:
                    edge_keys = list(G[u][v].keys())
                    if edge_keys:
                        edge_data = G[u][v][edge_keys[0]]
                
                if isinstance(edge_data, dict):
                    length = edge_data.get('length', 100)  # meters
                    # Use ORIGINAL maxspeed, not modified 'speed' field
                    original_speed = edge_data.get('maxspeed', 50)
                    
                    # Handle maxspeed being a list or string
                    if isinstance(original_speed, (list, tuple)):
                        original_speed = original_speed[0] if original_speed else 50
                    elif isinstance(original_speed, str):
                        try:
                            original_speed = float(original_speed.split()[0])  # Take first number
                        except (ValueError, IndexError):
                            original_speed = 50
                    
                    base_time = length / float(original_speed) * 3.6  # Convert to seconds
                else:
                    base_time = 60.0  # Default fallback
                
                base_travel_time += base_time
            
            end_time = time.time()
            computation_time = end_time - start_time
            return path, base_travel_time, computation_time
        
        if current in visited:
            continue
            
        visited.add(current)
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                # Calculate proper base travel time from length/speed (no congestion)
                edge_data = None
                if neighbor in G[current]:
                    edge_keys = list(G[current][neighbor].keys())
                    if edge_keys:
                        edge_data = G[current][neighbor][edge_keys[0]]
                
                if isinstance(edge_data, dict):
                    length = edge_data.get('length', 100)  # meters
                    # Use ORIGINAL maxspeed, not modified 'speed' field
                    original_speed = edge_data.get('maxspeed', 50)
                    
                    # Handle maxspeed being a list or string
                    if isinstance(original_speed, (list, tuple)):
                        original_speed = original_speed[0] if original_speed else 50
                    elif isinstance(original_speed, str):
                        try:
                            original_speed = float(original_speed.split()[0])  # Take first number
                        except (ValueError, IndexError):
                            original_speed = 50
                    
                    base_time = length / float(original_speed) * 3.6  # Convert to seconds
                else:
                    base_time = 60.0  # Default fallback
                
                # Calculate new cost with base travel time only
                new_cost = cost + base_time
                
                # Priority is purely based on heuristic distance to goal (greedy characteristic)
                priority = heuristic(neighbor, end)
                
                new_path = path + [neighbor]
                heapq.heappush(queue, (priority, neighbor, new_path, new_cost))
    
    # If no path is found
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"No path found from {start} to {end} using Shortest Path algorithm")
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
    Shortest Path Congestion Aware Algorithm - finds the SAME path as shortest path but with congestion-adjusted travel times.
    This algorithm runs exactly like the shortest path algorithm but returns travel times considering congestion.
    
    Args:
        G: NetworkX graph with congestion data
        start: Start node
        end: End node
        
    Returns:
        Tuple of (path, travel_time, computation_time)
    """
    start_time = time.time()
    
    # Get node positions for heuristic
    pos = {}
    for node in G.nodes():
        if 'x' in G.nodes[node] and 'y' in G.nodes[node]:
            pos[node] = (G.nodes[node]['x'], G.nodes[node]['y'])
        else:
            pos[node] = (0, 0)
    
    # Define heuristic function (Euclidean distance)
    def heuristic(n1: int, n2: int) -> float:
        if n1 in pos and n2 in pos:
            x1, y1 = pos[n1]
            x2, y2 = pos[n2]
            return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        else:
            return 0
    
    # Implementation of Simple Shortest Path Search (same as shortest path - ignores congestion in pathfinding)
    visited = set()
    queue = [(heuristic(start, end), start, [start], 0)]
    heapq.heapify(queue)
    
    while queue:
        _, current, path, cost = heapq.heappop(queue)
        
        if current == end:
            # Calculate travel time WITH congestion effects - EXACT SAME METHOD AS A*
            congestion_travel_time = 0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                base_time = get_edge_travel_time(G, u, v)
                # Get congestion from first edge key (multigraph support)
                congestion = 1.0
                if u in G and v in G[u]:
                    edge_keys = list(G[u][v].keys())
                    if edge_keys:
                        edge_data = G[u][v][edge_keys[0]]
                        if isinstance(edge_data, dict):
                            congestion = edge_data.get('congestion', 1.0)
                
                # Apply the same REALISTIC London congestion penalty used in A*
                if congestion > 4.0:
                    realistic_time = base_time * 3.5  # Heavy traffic: 71% speed reduction (stop-and-go traffic)
                elif congestion > 3.0:
                    realistic_time = base_time * 2.2  # Moderate traffic: 55% speed reduction (heavy congestion)
                elif congestion > 2.0:
                    realistic_time = base_time * 1.5  # Light traffic: 33% speed reduction (slow traffic)
                else:
                    realistic_time = base_time        # Free flow
                
                congestion_travel_time += realistic_time
            
            end_time = time.time()
            computation_time = end_time - start_time
            return path, congestion_travel_time, computation_time
        
        if current in visited:
            continue
            
        visited.add(current)
        
        for neighbor in G.neighbors(current):
            if neighbor not in visited:
                # Use get_edge_travel_time for consistency
                base_time = get_edge_travel_time(G, current, neighbor)
                
                # Calculate new cost with base travel time only (same as optimal path)
                new_cost = cost + base_time
                
                # Priority is purely based on heuristic distance to goal (greedy characteristic)
                priority = heuristic(neighbor, end)
                
                new_path = path + [neighbor]
                heapq.heappush(queue, (priority, neighbor, new_path, new_cost))
    
    # If no path is found
    end_time = time.time()
    computation_time = end_time - start_time
    print(f"No path found from {start} to {end} using Shortest Path Congestion Aware algorithm")
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