"""
Test script to verify that algorithms are finding different paths according to specifications:
- A*: Optimal path considering Congestion > Travel Time > Distance
- Shortest Path: Shortest distance path only
- Shortest Path Congestion Aware: Same path as Shortest Path but with congestion applied
"""

import networkx as nx
from models import Vehicle
from routing import enhanced_a_star_algorithm, shortest_path_algorithm, shortest_path_congestion_aware_algorithm
from unified_travel_time import UnifiedTravelTimeCalculator

def create_test_graph_with_high_congestion():
    """Create a test graph with varying congestion levels to test path differences."""
    G = nx.MultiGraph()
    
    # Add nodes with coordinates
    nodes = [
        (1, {'x': 0, 'y': 0}),
        (2, {'x': 100, 'y': 0}),
        (3, {'x': 200, 'y': 0}),
        (4, {'x': 0, 'y': 100}),
        (5, {'x': 100, 'y': 100}),
        (6, {'x': 200, 'y': 100}),
        (7, {'x': 100, 'y': 200}),
    ]
    
    G.add_nodes_from(nodes)
    
    # Add edges with varying congestion levels to force different path choices
    edges = [
        # Direct route (short but high congestion)
        (1, 2, {'length': 100, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 8.0}),  # High congestion
        (2, 3, {'length': 100, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 9.0}),  # Very high congestion
        
        # Alternative route (longer but lower congestion)
        (1, 4, {'length': 100, 'speed_kph': 30, 'maxspeed': 30, 'congestion': 1.5}),  # Low congestion
        (4, 5, {'length': 100, 'speed_kph': 35, 'maxspeed': 35, 'congestion': 2.0}),  # Low congestion
        (5, 6, {'length': 100, 'speed_kph': 45, 'maxspeed': 45, 'congestion': 2.5}),  # Low congestion
        
        # Mixed routes
        (2, 5, {'length': 100, 'speed_kph': 40, 'maxspeed': 40, 'congestion': 6.0}),  # High congestion
        (3, 6, {'length': 100, 'speed_kph': 60, 'maxspeed': 60, 'congestion': 7.0}),  # High congestion
        (5, 7, {'length': 100, 'speed_kph': 40, 'maxspeed': 40, 'congestion': 3.0}),  # Moderate congestion
        
        # Diagonal shortcuts
        (4, 7, {'length': 141, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 1.0}),  # Low congestion diagonal
        (1, 5, {'length': 141, 'speed_kph': 45, 'maxspeed': 45, 'congestion': 4.0}),  # Moderate congestion diagonal
    ]
    
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    
    return G

def test_path_differences():
    """Test that algorithms find different paths according to specifications."""
    print("=== TESTING ALGORITHM PATH DIFFERENCES ===")
    
    G = create_test_graph_with_high_congestion()
    calc = UnifiedTravelTimeCalculator()
    
    # Test routes that should show clear differences
    test_routes = [
        (1, 3, "Direct route with high congestion vs alternative"),
        (1, 6, "Multiple path options with varying congestion"),
        (1, 7, "Long route with multiple alternatives"),
    ]
    
    for source, destination, description in test_routes:
        print(f"\n--- Testing route {source} -> {destination}: {description} ---")
        
        # Calculate paths using all algorithms
        astar_path, astar_time, astar_comp = enhanced_a_star_algorithm(G, source, destination)
        shortest_path, shortest_time, shortest_comp = shortest_path_algorithm(G, source, destination)
        cong_aware_path, cong_aware_time, cong_aware_comp = shortest_path_congestion_aware_algorithm(G, source, destination)
        
        # Display results
        print(f"A* Algorithm:")
        print(f"  Path: {astar_path}")
        print(f"  Travel time: {astar_time:.2f}s")
        print(f"  Path length: {len(astar_path)} nodes")
        
        print(f"Shortest Path Algorithm:")
        print(f"  Path: {shortest_path}")
        print(f"  Travel time: {shortest_time:.2f}s (no congestion)")
        print(f"  Path length: {len(shortest_path)} nodes")
        
        print(f"Shortest Path Congestion Aware:")
        print(f"  Path: {cong_aware_path}")
        print(f"  Travel time: {cong_aware_time:.2f}s (with congestion)")
        print(f"  Path length: {len(cong_aware_path)} nodes")
        
        # Verify specifications
        print(f"\nVerification:")
        
        # Check if Shortest Path and Congestion Aware use same path
        if shortest_path == cong_aware_path:
            print("  ✅ Shortest Path and Congestion Aware use SAME path")
        else:
            print("  ❌ Shortest Path and Congestion Aware use DIFFERENT paths (ERROR)")
        
        # Check if A* found a different path (when congestion is significant)
        if astar_path != shortest_path:
            print("  ✅ A* found DIFFERENT path (avoiding congestion)")
            
            # Calculate congestion impact
            shortest_with_congestion = calc.calculate_path_travel_time(G, shortest_path, apply_congestion=True)
            print(f"  📊 If shortest path had congestion: {shortest_with_congestion:.2f}s")
            print(f"  📊 A* path with congestion: {astar_time:.2f}s")
            
            if astar_time < shortest_with_congestion:
                print("  ✅ A* path is BETTER than shortest path with congestion")
            else:
                print("  ⚠️  A* path is not better (may need algorithm tuning)")
        else:
            print("  ⚠️  A* found SAME path as shortest (low congestion impact or limited alternatives)")
        
        # Analyze path details
        print(f"\nPath Analysis:")
        if astar_path:
            astar_congestion = analyze_path_congestion(G, astar_path)
            print(f"  A* path average congestion: {astar_congestion:.2f}")
        
        if shortest_path:
            shortest_congestion = analyze_path_congestion(G, shortest_path)
            print(f"  Shortest path average congestion: {shortest_congestion:.2f}")

def analyze_path_congestion(G, path):
    """Calculate average congestion level for a path."""
    if len(path) < 2:
        return 0
    
    total_congestion = 0
    edge_count = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if u in G and v in G[u]:
            edge_keys = list(G[u][v].keys())
            if edge_keys:
                k = edge_keys[0]
                congestion = G[u][v][k].get('congestion', 1.0)
                total_congestion += congestion
                edge_count += 1
    
    return total_congestion / edge_count if edge_count > 0 else 0

def test_edge_congestion_details():
    """Show detailed congestion information for edges."""
    print("\n=== EDGE CONGESTION DETAILS ===")
    
    G = create_test_graph_with_high_congestion()
    
    print("Edge congestion levels:")
    for u, v, k, data in G.edges(keys=True, data=True):
        congestion = data.get('congestion', 1.0)
        length = data.get('length', 0)
        speed = data.get('speed_kph', 0)
        
        # Classify congestion level
        if congestion <= 2.0:
            level = "LOW"
        elif congestion <= 5.0:
            level = "MODERATE"
        elif congestion <= 7.0:
            level = "HIGH"
        else:
            level = "SEVERE"
        
        print(f"  Edge {u}->{v}: {length}m, {speed}km/h, congestion={congestion:.1f} ({level})")

if __name__ == "__main__":
    print("Testing algorithm path selection according to specifications...")
    print("Expected behavior:")
    print("- A*: Should find optimal path considering Congestion > Travel Time > Distance")
    print("- Shortest Path: Should find shortest distance path (ignore congestion)")
    print("- Shortest Path Congestion Aware: Should use same path as Shortest Path but apply congestion")
    
    test_edge_congestion_details()
    test_path_differences()
    
    print("\n=== TEST COMPLETED ===")
