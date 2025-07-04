"""
Core data structures and classes for the traffic simulation project.
"""

import numpy as np
import networkx as nx
import osmnx as ox
import random
import time
from typing import Dict, List, Optional, Tuple, Any


class Vehicle:
    """Represents a vehicle in the traffic simulation with routing capabilities."""
    
    def __init__(self, id: int, source: int, destination: int, path: Optional[List[int]] = None):
        self.id = id
        self.source = source
        self.destination = destination
        self.path = path
        self.paths: Dict[str, List[int]] = {}
        self.travel_times: Dict[str, float] = {}
        self.computation_times: Dict[str, float] = {}
        self.service_rates: Dict[str, float] = {}
        self.current_position = 0
        self.completed = False
        self.selected_for_analysis = False

    def __str__(self):
        return f"Vehicle {self.id}: {self.source} -> {self.destination} (Paths: {len(self.paths)})"


def add_realistic_london_speeds(G: nx.Graph) -> nx.Graph:
    """
    Add realistic speed data to London street network using OSMnx methodology.
    Handles missing speed data with London-specific defaults.
    
    Args:
        G: NetworkX graph from OSMnx
        
    Returns:
        Graph with speed_kph attribute added to all edges
    """
    # London-specific speed defaults by highway type (km/h)
    # These reflect ACTUAL driving speeds, not posted speed limits
    london_highway_speeds = {
        'motorway': 65,        # M25, A40(M) - but capped for city sections
        'trunk': 45,           # A4, A40 - major roads with traffic
        'primary': 35,         # A1, A10 - main roads through London
        'secondary': 30,       # A roads, B roads
        'tertiary': 25,        # Local distributor roads
        'unclassified': 20,    # Minor roads
        'residential': 18,     # Residential streets
        'living_street': 15,   # Shared space, very slow
        'service': 15,         # Service roads, car parks
        'track': 12,           # Unpaved roads
        'path': 8,             # Footpaths accessible to vehicles
    }
    
    # Add speed data using OSMnx method with London-specific parameters
    try:
        G = ox.add_edge_speeds(G, hwy_speeds=london_highway_speeds, fallback=20)
    except Exception as e:
        print(f"Warning: Could not use ox.add_edge_speeds: {e}")
        print("Falling back to manual speed assignment...")
        
        # Manual fallback speed assignment
        for u, v, k, data in G.edges(keys=True, data=True):
            # Try to get existing speed data
            speed_kph = None
            
            # Check maxspeed first
            maxspeed = data.get('maxspeed')
            if maxspeed:
                try:
                    if isinstance(maxspeed, str):
                        if 'mph' in maxspeed.lower():
                            mph_val = float(maxspeed.lower().replace('mph', '').strip())
                            speed_kph = mph_val * 1.60934
                        else:
                            speed_kph = float(maxspeed)
                    elif isinstance(maxspeed, (int, float)):
                        speed_kph = float(maxspeed)
                except (ValueError, TypeError):
                    pass
            
            # If no maxspeed, use highway type
            if speed_kph is None:
                highway = data.get('highway', 'unclassified')
                if isinstance(highway, list):
                    highway = highway[0]
                speed_kph = london_highway_speeds.get(highway, 20)
            
            # Apply London traffic reality adjustments
            # Posted speeds are often higher than actual driving speeds
            if speed_kph > 50:  # Reduce highway speeds for London traffic
                speed_kph = min(speed_kph * 0.6, 35)  # Cap at 35 km/h
            elif speed_kph > 30:
                speed_kph = speed_kph * 0.75  # Reduce by 25% for traffic
            else:
                speed_kph = max(speed_kph * 0.9, 12)  # Slight reduction, min 12 km/h
            
            # Set the speed_kph attribute
            G[u][v][k]['speed_kph'] = round(speed_kph, 1)
    
    return G


def get_base_travel_time(G: nx.Graph, u: int, v: int, default: float = 60.0) -> float:
    """
    Get BASE travel time using proper OSM travel_time attribute.
    Falls back to length/speed calculation if travel_time not available.
    
    Args:
        G: NetworkX graph with OSM data
        u: Source node
        v: Target node
        default: Default travel time if not found
        
    Returns:
        Base travel time for the edge in seconds
    """
    try:
        if u in G and v in G[u]:
            edge_keys = list(G[u][v].keys())
            if edge_keys:
                edge_data = G[u][v][edge_keys[0]]
                
                if isinstance(edge_data, dict):
                    # Method 1: Use OSMnx calculated travel_time if available
                    travel_time = edge_data.get('travel_time')
                    if travel_time is not None:
                        return float(travel_time)
                    
                    # Method 2: Calculate from length and speed_kph
                    length = edge_data.get('length')
                    speed_kph = edge_data.get('speed_kph')
                    
                    if length is not None and speed_kph is not None and speed_kph > 0:
                        # travel_time = length(m) / speed(m/s) = length(m) / (speed_kph * 1000/3600)
                        travel_time = length / (speed_kph * 1000 / 3600)
                        return travel_time
                    
                    # Method 3: Fallback calculation
                    if length is not None:
                        # Use default London city speed if no speed data
                        fallback_speed = 20  # km/h
                        travel_time = length / (fallback_speed * 1000 / 3600)
                        return travel_time
        
    except Exception:
        pass
    
    return default


def get_edge_travel_time(G: nx.Graph, u: int, v: int, default: float = 60.0) -> float:
    """
    Robust method to get travel time using OSM data with comprehensive fallbacks.
    Prioritizes OSMnx travel_time attribute, then calculates from OSM speed data.
    
    Args:
        G: NetworkX graph with OSM data
        u: Source node
        v: Target node
        default: Default travel time if not found
        
    Returns:
        Travel time for the edge in seconds
    """
    try:
        # Method 1: Direct edge access with multiple keys
        if u in G and v in G[u]:
            edge_dict = G[u][v]
            
            if isinstance(edge_dict, dict):
                # Multi-edge graph - try each key
                for key in edge_dict.keys():
                    edge_data = edge_dict[key]
                    
                    if isinstance(edge_data, dict):
                        # Priority 1: OSMnx calculated travel_time
                        travel_time = edge_data.get('travel_time')
                        if travel_time is not None:
                            return float(travel_time)
                        
                        # Priority 2: Calculate from OSM length and speed_kph
                        length = edge_data.get('length')
                        speed_kph = edge_data.get('speed_kph')
                        if length is not None and speed_kph is not None and speed_kph > 0:
                            travel_time = length / (speed_kph * 1000 / 3600)
                            return travel_time
                        
                        # Priority 3: Calculate from length and maxspeed
                        maxspeed = edge_data.get('maxspeed')
                        if length is not None and maxspeed is not None:
                            try:
                                speed = None
                                if isinstance(maxspeed, str):
                                    if 'mph' in maxspeed.lower():
                                        mph_val = float(maxspeed.lower().replace('mph', '').strip())
                                        speed = mph_val * 1.60934  # Convert to km/h
                                    else:
                                        speed = float(maxspeed)
                                elif isinstance(maxspeed, (int, float)):
                                    speed = float(maxspeed)
                                
                                if speed and speed > 0:
                                    # Apply London traffic reduction
                                    realistic_speed = min(speed * 0.7, 35)  # Cap at 35 km/h
                                    realistic_speed = max(realistic_speed, 12)  # Min 12 km/h
                                    travel_time = length / (realistic_speed * 1000 / 3600)
                                    return travel_time
                            except (ValueError, TypeError):
                                pass
                        
                        # Priority 4: Use highway type for speed estimation
                        highway = edge_data.get('highway')
                        if length is not None and highway is not None:
                            highway_speeds = {
                                'motorway': 35, 'trunk': 30, 'primary': 25,
                                'secondary': 22, 'tertiary': 20, 'unclassified': 18,
                                'residential': 15, 'service': 12
                            }
                            if isinstance(highway, list):
                                highway = highway[0]
                            speed = highway_speeds.get(highway, 20)
                            travel_time = length / (speed * 1000 / 3600)
                            return travel_time
                    
                    elif isinstance(edge_data, (int, float)):
                        return float(edge_data)
            
            elif isinstance(edge_dict, (int, float)):
                return float(edge_dict)
        
        # Method 2: NetworkX edges access
        try:
            edges_data = G.edges[u, v]
            if isinstance(edges_data, dict):
                travel_time = edges_data.get('travel_time')
                if travel_time is not None:
                    return float(travel_time)
        except:
            pass
        
        # Method 3: Try with key=0 (common default)
        try:
            edge_data = G.edges[u, v, 0]
            if isinstance(edge_data, dict):
                travel_time = edge_data.get('travel_time')
                if travel_time is not None:
                    return float(travel_time)
        except:
            pass
        
        # Method 4: Use get_edge_data
        try:
            edge_data = G.get_edge_data(u, v)
            if isinstance(edge_data, dict):
                if len(edge_data) == 1:
                    # Single edge
                    key = list(edge_data.keys())[0]
                    data = edge_data[key]
                    if isinstance(data, dict):
                        travel_time = data.get('travel_time')
                        if travel_time is not None:
                            return float(travel_time)
        except:
            pass
            
    except Exception:
        pass
    
    return default


def get_edge_attribute(G: nx.Graph, u: int, v: int, attribute: str, default: Any) -> Any:
    """
    Get any attribute from edge with fallbacks.
    
    Args:
        G: NetworkX graph
        u: Source node
        v: Target node
        attribute: Attribute name to retrieve
        default: Default value if not found
        
    Returns:
        Attribute value or default
    """
    try:
        if u in G and v in G[u]:
            edge_dict = G[u][v]
            if isinstance(edge_dict, dict):
                for key in edge_dict.keys():
                    edge_data = edge_dict[key]
                    if isinstance(edge_data, dict):
                        value = edge_data.get(attribute)
                        if value is not None:
                            return value
        
        # Try NetworkX edges access
        edge_data = G.edges[u, v, 0]
        if isinstance(edge_data, dict):
            value = edge_data.get(attribute)
            if value is not None:
                return value
                
    except:
        pass
    
    return default


def safe_get_edge_data(G: nx.Graph, u: int, v: int, key: int, attribute: str, default: Any = 0) -> Any:
    """
    Safely get edge data attribute with fallbacks.
    
    Args:
        G: NetworkX graph
        u: Source node
        v: Target node
        key: Edge key
        attribute: Attribute name
        default: Default value
        
    Returns:
        Attribute value or default
    """
    try:
        # Method 1: Try G[u][v][key]
        if u in G and v in G[u] and key in G[u][v]:
            edge_data = G[u][v][key]
            if isinstance(edge_data, dict):
                return edge_data.get(attribute, default)
        
        # Method 2: Try G.edges
        edge_attrs = G.edges[u, v, key]
        if isinstance(edge_attrs, dict):
            return edge_attrs.get(attribute, default)
        elif isinstance(edge_attrs, (int, float)) and attribute == 'travel_time':
            return float(edge_attrs)
        
        # Method 3: Try get_edge_data
        edge_data = G.get_edge_data(u, v, key)
        if isinstance(edge_data, dict):
            return edge_data.get(attribute, default)
            
    except Exception:
        pass
    
    return default


def calculate_shortest_path(G: nx.Graph, source: int, destination: int) -> Optional[List[int]]:
    """
    Calculate the shortest path between source and destination.
    
    Args:
        G: NetworkX graph
        source: Source node
        destination: Destination node
        
    Returns:
        Path as list of nodes or None if no path exists
    """
    try:
        path = nx.shortest_path(G, source, destination, weight='length')
        return path
    except nx.NetworkXNoPath:
        print(f"No path found between {source} and {destination}")
        return None
    except Exception as e:
        print(f"Error calculating path: {e}")
        return None


def load_london_network() -> nx.Graph:
    """Load and return the London street network with proper OSM attributes."""
    print("Loading London street network...")
    
    # Define the London area - using City of London for a smaller, manageable area
    place_name = "City of London, UK"
    
    # Configure OSMnx to include comprehensive useful tags
    original_useful_tags_way = ox.settings.useful_tags_way
    ox.settings.useful_tags_way = [
        'access', 'bridge', 'highway', 'junction', 'lanes', 'maxspeed', 
        'name', 'oneway', 'service', 'surface', 'tunnel', 'width',
        'cycleway', 'footway', 'sidewalk', 'parking:lane', 'ref'
    ]
    
    try:
        G = ox.graph_from_place(
            place_name, 
            network_type='drive', 
            simplify=False,
            retain_all=False,
            truncate_by_edge=True
        )
    finally:
        # Restore original settings
        ox.settings.useful_tags_way = original_useful_tags_way
    
    # Check if the network is connected
    is_connected = nx.is_strongly_connected(G)
    if not is_connected:
        print("Extracting largest strongly connected component...")
        largest_component = max(nx.strongly_connected_components(G), key=len)
        G = G.subgraph(largest_component).copy()
    
    print(f"Network loaded with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    # Add proper speed data using OSMnx functions
    print("Processing OSM speed data...")
    G = add_realistic_london_speeds(G)
    
    # Add travel times based on proper OSM data
    print("Calculating travel times...")
    G = ox.add_edge_travel_times(G)
    
    print("✓ London network ready with realistic speed and travel time data")
    return G


def generate_initial_congestion(G: nx.Graph) -> Dict[str, float]:
    """
    Generate initial congestion data for all edges.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping edge_id to congestion level
    """
    print("Generating initial congestion data...")
    congestion_data = {}
    
    for u, v, k in G.edges(keys=True):
        # Generate realistic congestion value between 1.0 and 4.0 for Normal scenario
        congestion = 1.0 + 3.0 * np.random.random()
        
        # Store congestion value
        edge_id = f"{u}_{v}_{k}"
        congestion_data[edge_id] = congestion
        
        # Add congestion attribute to the edge in the graph
        G[u][v][k]['congestion'] = congestion
        
        # Initialize vehicle count on each edge
        G[u][v][k]['vehicle_count'] = 0
    
    return congestion_data


def calculate_congestion_stats(congestion_data: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate statistics for congestion data.
    
    Args:
        congestion_data: Dictionary of congestion values
        
    Returns:
        Dictionary with statistical measures
    """
    congestion_values = list(congestion_data.values())
    return {
        "mean": np.mean(congestion_values),
        "median": np.median(congestion_values),
        "min": np.min(congestion_values),
        "max": np.max(congestion_values),
        "std": np.std(congestion_values)
    }


def update_vehicle_counts_for_path(G: nx.Graph, path: List[int], increment: int = 1) -> None:
    """
    Update the vehicle count along a path.
    
    Args:
        G: NetworkX graph
        path: List of nodes representing the path
        increment: Number to add to vehicle count (can be negative)
    """
    # Get pairs of nodes representing edges in the path
    edges = list(zip(path[:-1], path[1:]))
    
    for u, v in edges:
        # Find all edges between these nodes (there might be multiple with different keys)
        if u in G and v in G[u]:
            for k in G[u][v]:
                # Update vehicle count
                current_count = G[u][v][k].get('vehicle_count', 0)
                G[u][v][k]['vehicle_count'] = current_count + increment


def create_evenly_distributed_notable_locations(G: nx.Graph) -> Dict[str, int]:
    """
    Create evenly distributed notable locations across the map.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary mapping location names to node IDs
    """
    # Fixed notable locations for consistency
    landmark_names = [
        "Central Business District",
        "Financial District", 
        "Shopping Center",
        "University Area",
        "Tourist Attraction",
        "Residential Zone A",
        "Residential Zone B",
        "Industrial Park",
        "Sports Arena",
        "Hospital Area"
    ]
    
    # For consistent node selection, use a fixed seed
    np.random.seed(42)
    
    # Get node positions
    node_positions = {}
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_positions[node] = (data['x'], data['y'])
    
    if not node_positions:
        print("Warning: No nodes with position data found.")
        # Fall back to random nodes
        all_nodes = list(G.nodes())
        notable_locations = {}
        for i, name in enumerate(landmark_names):
            if i < len(all_nodes):
                notable_locations[name] = all_nodes[i]
        return notable_locations
    
    # Convert to numpy arrays for easy manipulation
    nodes = list(node_positions.keys())
    positions = np.array(list(node_positions.values()))
    
    # Find the bounding box of the map
    min_x, min_y = positions.min(axis=0)
    max_x, max_y = positions.max(axis=0)
    
    # Divide the map into a 3x4 grid
    grid_size_x = (max_x - min_x) / 3
    grid_size_y = (max_y - min_y) / 4
    
    notable_locations = {}
    grid_cells_used = set()
    
    # Try to pick nodes from different grid cells to ensure even distribution
    for name in landmark_names:
        # Try multiple times to find a good cell
        for _ in range(20):
            # Pick a random grid cell that hasn't been used yet
            i = np.random.randint(0, 3)
            j = np.random.randint(0, 4)
            
            if (i, j) not in grid_cells_used:
                # Define the boundaries of this grid cell
                cell_min_x = min_x + i * grid_size_x
                cell_max_x = min_x + (i + 1) * grid_size_x
                cell_min_y = min_y + j * grid_size_y
                cell_max_y = min_y + (j + 1) * grid_size_y
                
                # Find nodes within this cell
                cell_nodes = []
                for node, (x, y) in node_positions.items():
                    if (cell_min_x <= x <= cell_max_x and 
                        cell_min_y <= y <= cell_max_y):
                        cell_nodes.append(node)
                
                if cell_nodes:
                    # Choose a random node from this cell
                    notable_locations[name] = np.random.choice(cell_nodes)
                    grid_cells_used.add((i, j))
                    break
        
        # If we still don't have a node for this landmark, pick a random one
        if name not in notable_locations and nodes:
            notable_locations[name] = np.random.choice(nodes)
    
    # Reset random seed for other operations
    np.random.seed()
    
    print("Created evenly distributed notable locations across the map")
    return notable_locations


def debug_osm_data_extraction(G: nx.Graph, num_samples: int = 10) -> None:
    """
    Debug and validate OSM data extraction for London network.
    
    Args:
        G: NetworkX graph with OSM data
        num_samples: Number of edges to sample for detailed analysis
    """
    print("🔍 OSM DATA EXTRACTION DEBUG REPORT")
    print("=" * 50)
    
    # Count edges with different attributes
    total_edges = G.number_of_edges()
    edges_with_travel_time = 0
    edges_with_speed_kph = 0
    edges_with_maxspeed = 0
    edges_with_highway = 0
    edges_with_length = 0
    
    highway_types = {}
    speed_values = []
    travel_times = []
    lengths = []
    
    print(f"Total edges in network: {total_edges}")
    print("\nAnalyzing edge attributes...")
    
    for u, v, k, data in G.edges(keys=True, data=True):
        if 'travel_time' in data:
            edges_with_travel_time += 1
            travel_times.append(data['travel_time'])
        
        if 'speed_kph' in data:
            edges_with_speed_kph += 1
            speed_values.append(data['speed_kph'])
        
        if 'maxspeed' in data:
            edges_with_maxspeed += 1
        
        if 'highway' in data:
            edges_with_highway += 1
            highway = data['highway']
            if isinstance(highway, list):
                highway = highway[0]
            highway_types[highway] = highway_types.get(highway, 0) + 1
        
        if 'length' in data:
            edges_with_length += 1
            lengths.append(data['length'])
    
    # Print statistics
    print(f"\n📊 ATTRIBUTE COVERAGE:")
    print(f"  travel_time: {edges_with_travel_time}/{total_edges} ({edges_with_travel_time/total_edges*100:.1f}%)")
    print(f"  speed_kph: {edges_with_speed_kph}/{total_edges} ({edges_with_speed_kph/total_edges*100:.1f}%)")
    print(f"  maxspeed: {edges_with_maxspeed}/{total_edges} ({edges_with_maxspeed/total_edges*100:.1f}%)")
    print(f"  highway: {edges_with_highway}/{total_edges} ({edges_with_highway/total_edges*100:.1f}%)")
    print(f"  length: {edges_with_length}/{total_edges} ({edges_with_length/total_edges*100:.1f}%)")
    
    # Highway type distribution
    print(f"\n🛣️  HIGHWAY TYPES:")
    for highway, count in sorted(highway_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        percentage = count / total_edges * 100
        print(f"  {highway}: {count} ({percentage:.1f}%)")
    
    # Speed statistics
    if speed_values:
        print(f"\n🚗 SPEED STATISTICS (km/h):")
        print(f"  Average: {np.mean(speed_values):.1f}")
        print(f"  Min: {np.min(speed_values):.1f}")
        print(f"  Max: {np.max(speed_values):.1f}")
        print(f"  Median: {np.median(speed_values):.1f}")
    
    # Travel time statistics
    if travel_times:
        print(f"\n⏱️  TRAVEL TIME STATISTICS (seconds):")
        print(f"  Average: {np.mean(travel_times):.1f}")
        print(f"  Min: {np.min(travel_times):.1f}")
        print(f"  Max: {np.max(travel_times):.1f}")
        print(f"  Median: {np.median(travel_times):.1f}")
    
    # Length statistics
    if lengths:
        print(f"\n📏 LENGTH STATISTICS (meters):")
        print(f"  Average: {np.mean(lengths):.1f}")
        print(f"  Min: {np.min(lengths):.1f}")
        print(f"  Max: {np.max(lengths):.1f}")
        print(f"  Median: {np.median(lengths):.1f}")
    
    # Sample detailed edge analysis
    print(f"\n🔬 DETAILED EDGE SAMPLES:")
    print("-" * 30)
    
    sample_edges = list(G.edges(keys=True, data=True))[:num_samples]
    for i, (u, v, k, data) in enumerate(sample_edges):
        print(f"\nEdge {i+1}: {u} → {v} (key: {k})")
        
        # Show relevant attributes
        attrs_to_show = ['length', 'highway', 'maxspeed', 'speed_kph', 'travel_time', 'lanes', 'surface']
        for attr in attrs_to_show:
            if attr in data:
                print(f"  {attr}: {data[attr]}")
        
        # Calculate and verify travel time
        base_time = get_base_travel_time(G, u, v)
        edge_time = get_edge_travel_time(G, u, v)
        print(f"  calculated base_time: {base_time:.2f}s")
        print(f"  calculated edge_time: {edge_time:.2f}s")
        
        # Speed check
        if 'length' in data and 'travel_time' in data and data['travel_time'] > 0:
            effective_speed = data['length'] / data['travel_time'] * 3.6  # km/h
            print(f"  effective speed: {effective_speed:.1f} km/h")
    
    print(f"\n✅ OSM data extraction analysis complete!")


def validate_london_travel_times(G: nx.Graph, num_routes: int = 5) -> None:
    """
    Validate that calculated travel times are realistic for London.
    
    Args:
        G: NetworkX graph
        num_routes: Number of sample routes to test
    """
    print("\n🚗 LONDON TRAVEL TIME VALIDATION")
    print("=" * 40)
    
    from routing import enhanced_a_star_algorithm
    
    nodes = list(G.nodes())
    realistic_count = 0
    
    for i in range(num_routes):
        start = random.choice(nodes)
        end = random.choice(nodes)
        
        if start != end:
            try:
                path, travel_time, comp_time = enhanced_a_star_algorithm(G, start, end)
                if path and len(path) > 10:
                    # Calculate actual distance
                    total_distance = 0
                    for j in range(len(path) - 1):
                        u, v = path[j], path[j+1]
                        if u in G and v in G[u]:
                            edge_keys = list(G[u][v].keys())
                            if edge_keys:
                                edge_data = G[u][v][edge_keys[0]]
                                if isinstance(edge_data, dict):
                                    total_distance += edge_data.get('length', 0)
                    
                    if total_distance > 0 and travel_time > 0:
                        distance_km = total_distance / 1000
                        time_hours = travel_time / 3600
                        avg_speed = distance_km / time_hours
                        
                        print(f"\nRoute {i+1}: {len(path)} nodes, {distance_km:.2f}km")
                        print(f"  Time: {travel_time/60:.1f} minutes")
                        print(f"  Speed: {avg_speed:.1f} km/h")
                        
                        # Check if realistic for London
                        if 8 <= avg_speed <= 40:  # Realistic London speed range
                            print(f"  ✅ Realistic for London traffic")
                            realistic_count += 1
                        elif avg_speed < 8:
                            print(f"  ⚠️  Too slow - may indicate traffic jam")
                        else:
                            print(f"  ❌ Too fast for London city driving")
            
            except Exception as e:
                print(f"Route {i+1}: Error - {e}")
    
    print(f"\n📊 VALIDATION SUMMARY:")
    print(f"  Realistic routes: {realistic_count}/{num_routes}")
    if realistic_count >= num_routes * 0.8:
        print(f"  ✅ Travel times appear realistic for London!")
    else:
        print(f"  ⚠️  Travel times may need adjustment")