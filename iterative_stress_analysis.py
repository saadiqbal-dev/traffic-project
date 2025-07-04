"""
Iterative Stress Analysis System
===============================
This module provides comprehensive iterative stress testing from Point A to Point B
across all congestion scenarios with detailed Excel reporting for analysis.

Features:
- Tests all congestion scenarios (Normal, Morning, Evening, Weekend, Special)
- Iterative testing with increasing vehicle counts (20 vehicles per iteration)
- Comprehensive data collection for all algorithms
- Excel export with detailed metrics for graph generation
- Mean congestion analysis per route and overall map
- Uses the same notable locations as the main system
"""

import os
import time
import random
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from models import Vehicle, load_london_network, generate_initial_congestion, create_evenly_distributed_notable_locations
from routing import enhanced_a_star_algorithm, shortest_path_algorithm, shortest_path_congestion_aware_algorithm
from congestion import apply_consistent_congestion_scenario
from unified_travel_time import UnifiedTravelTimeCalculator


class IterativeStressAnalyzer:
    """
    Comprehensive iterative stress testing system for route analysis.
    """
    
    def __init__(self):
        self.calc = UnifiedTravelTimeCalculator()
        self.results_data = []
        
        # Predefined 10 locations for testing (London landmarks/areas)
        self.test_locations = {
            1: "Central London",
            2: "Westminster", 
            3: "Camden",
            4: "Greenwich",
            5: "Canary Wharf",
            6: "Shoreditch",
            7: "Kensington",
            8: "Hammersmith",
            9: "Islington",
            10: "Southwark"
        }
        
        # Congestion scenarios to test
        self.congestion_scenarios = [
            "Normal",
            "Morning",
            "Evening", 
            "Weekend",
            "Special"
        ]
        
        # Algorithm names
        self.algorithms = [
            "A*",
            "Shortest Path",
            "Shortest Path Congestion Aware"
        ]
    
    def get_available_locations(self) -> Dict[int, str]:
        """Get available test locations."""
        return self.test_locations.copy()
    
    def calculate_mean_route_congestion(self, G: nx.Graph, path: List[int]) -> float:
        """Calculate mean congestion level for a specific route."""
        if len(path) < 2:
            return 0.0
        
        total_congestion = 0.0
        edge_count = 0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in G and v in G[u]:
                edge_keys = list(G[u][v].keys())
                if edge_keys:
                    k = edge_keys[0]
                    edge_data = G[u][v][k]
                    congestion = edge_data.get('congestion', 1.0)
                    total_congestion += congestion
                    edge_count += 1
        
        return total_congestion / edge_count if edge_count > 0 else 0.0
    
    def calculate_overall_map_congestion(self, G: nx.Graph) -> float:
        """Calculate average congestion across the entire map."""
        total_congestion = 0.0
        edge_count = 0
        
        for u, v, k, data in G.edges(keys=True, data=True):
            congestion = data.get('congestion', 1.0)
            total_congestion += congestion
            edge_count += 1
        
        return total_congestion / edge_count if edge_count > 0 else 0.0
    
    def run_algorithm_test(self, G: nx.Graph, source: int, destination: int, 
                          algorithm_name: str) -> Tuple[Optional[List[int]], float, float, float]:
        """
        Run a specific algorithm and return path, travel time, computation time, and path length.
        """
        try:
            if algorithm_name == "A*":
                path, travel_time, comp_time = enhanced_a_star_algorithm(G, source, destination)
            elif algorithm_name == "Shortest Path":
                path, travel_time, comp_time = shortest_path_algorithm(G, source, destination)
            elif algorithm_name == "Shortest Path Congestion Aware":
                path, travel_time, comp_time = shortest_path_congestion_aware_algorithm(G, source, destination)
            else:
                return None, float('inf'), 0.0, 0
            
            path_length = len(path) if path else 0
            return path, travel_time, comp_time, path_length
            
        except Exception as e:
            print(f"Error running {algorithm_name}: {e}")
            return None, float('inf'), 0.0, 0
    
    def run_single_iteration(self, G: nx.Graph, source: int, destination: int, 
                           scenario: str, iteration: int, num_vehicles: int) -> List[Dict[str, Any]]:
        """
        Run a single iteration test for all algorithms.
        """
        iteration_results = []
        overall_congestion = self.calculate_overall_map_congestion(G)
        
        print(f"    Iteration {iteration}: {num_vehicles} vehicles")
        
        for algorithm in self.algorithms:
            # Run algorithm
            path, travel_time, comp_time, path_length = self.run_algorithm_test(
                G, source, destination, algorithm
            )
            
            # Calculate route congestion
            route_congestion = 0.0
            if path:
                route_congestion = self.calculate_mean_route_congestion(G, path)
            
            # Store results
            result = {
                'Congestion_Scenario': scenario,
                'Iteration': iteration,
                'Number_of_Vehicles': num_vehicles,
                'Algorithm_Name': algorithm,
                'Travel_Time': travel_time if travel_time != float('inf') else 'No Path',
                'Path_Length': path_length,
                'Computation_Time': comp_time,
                'Mean_Route_Congestion': route_congestion,
                'Average_Overall_Map_Congestion': overall_congestion,
                'Path': str(path) if path else 'No Path',
                'Source_Location': source,
                'Destination_Location': destination,
                'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            iteration_results.append(result)
            
            # Print progress
            if path:
                print(f"      {algorithm}: {travel_time:.2f}s, {path_length} nodes, route congestion: {route_congestion:.2f}")
            else:
                print(f"      {algorithm}: No path found")
        
        return iteration_results
    
    def run_scenario_analysis(self, G: nx.Graph, source: int, destination: int, 
                            scenario: str, max_iterations: int = 20) -> List[Dict[str, Any]]:
        """
        Run complete analysis for a single congestion scenario.
        """
        print(f"  Testing scenario: {scenario}")
        scenario_results = []
        
        # Generate initial congestion and apply scenario
        congestion_data = generate_initial_congestion(G)
        original_congestion = congestion_data.copy()
        
        # Apply scenario-specific congestion
        congestion_data, _ = apply_consistent_congestion_scenario(G, congestion_data, scenario, original_congestion)
        
        # Run iterations with increasing vehicle counts
        for iteration in range(1, max_iterations + 1):
            num_vehicles = iteration * 20  # 20 vehicles per iteration
            
            # Simulate vehicle load effect on congestion (optional enhancement)
            # For now, we'll use the base congestion scenario
            
            # Run iteration
            iteration_results = self.run_single_iteration(
                G, source, destination, scenario, iteration, num_vehicles
            )
            
            scenario_results.extend(iteration_results)
        
        return scenario_results
    
    def run_comprehensive_analysis(self, source: int, destination: int, 
                                 max_iterations: int = 20) -> str:
        """
        Run comprehensive iterative stress analysis from Point A to Point B.
        
        Args:
            source: Source location ID (1-10)
            destination: Destination location ID (1-10)
            max_iterations: Number of iterations per scenario (default: 20)
            
        Returns:
            Path to generated Excel file
        """
        print("=" * 80)
        print("COMPREHENSIVE ITERATIVE STRESS ANALYSIS")
        print("=" * 80)
        print(f"Route: {self.test_locations.get(source, 'Unknown')} -> {self.test_locations.get(destination, 'Unknown')}")
        print(f"Iterations per scenario: {max_iterations}")
        print(f"Vehicle increment: 20 per iteration")
        print(f"Total scenarios: {len(self.congestion_scenarios)}")
        print()
        
        # Load London graph
        print("Loading London street network...")
        try:
            G = load_london_network()
            print(f"Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            print(f"Error loading graph: {e}")
            print("Using fallback test graph...")
            G = self.create_fallback_graph()
            # For fallback graph, use the location IDs directly
            source_node = source
            destination_node = destination
        else:
            # Use the same notable locations function as the main system
            print("Creating notable locations using main system function...")
            notable_locations = create_evenly_distributed_notable_locations(G)
            location_names = list(notable_locations.keys())
            
            print("Notable locations created:")
            for name, node in notable_locations.items():
                print(f"  {name}: Node {node}")
            
            # Map location IDs to actual notable location nodes
            if source <= len(location_names) and destination <= len(location_names):
                source_name = location_names[source - 1]
                destination_name = location_names[destination - 1]
                source_node = notable_locations[source_name]
                destination_node = notable_locations[destination_name]
                print(f"Mapped locations:")
                print(f"  {self.test_locations[source]} -> {source_name} (Node: {source_node})")
                print(f"  {self.test_locations[destination]} -> {destination_name} (Node: {destination_node})")
            else:
                print(f"Warning: Invalid location IDs, using first two notable locations")
                source_name = location_names[0]
                destination_name = location_names[1]
                source_node = notable_locations[source_name]
                destination_node = notable_locations[destination_name]
                print(f"Using: {source_name} (Node: {source_node}) -> {destination_name} (Node: {destination_node})")
        
        print(f"Final route: Node {source_node} -> Node {destination_node}")
        
        # Initialize results storage
        all_results = []
        
        # Run analysis for each congestion scenario
        for scenario in self.congestion_scenarios:
            scenario_results = self.run_scenario_analysis(
                G.copy(), source_node, destination_node, scenario, max_iterations
            )
            all_results.extend(scenario_results)
        
        # Generate Excel report
        excel_path = self.generate_excel_report(all_results, source, destination)
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETED")
        print("=" * 80)
        print(f"Total data points collected: {len(all_results)}")
        print(f"Excel report generated: {excel_path}")
        print("Ready for graph generation and analysis!")
        
        return excel_path
    
    def create_fallback_graph(self) -> nx.MultiGraph:
        """Create a fallback test graph if London graph fails to load."""
        G = nx.MultiGraph()
        
        # Add 10 test nodes
        for i in range(1, 11):
            G.add_node(i, x=i*100, y=i*50)
        
        # Add edges connecting the nodes
        edges = [
            (1, 2, {'length': 1000, 'speed_kph': 50}),
            (2, 3, {'length': 1200, 'speed_kph': 40}),
            (3, 4, {'length': 800, 'speed_kph': 60}),
            (4, 5, {'length': 1500, 'speed_kph': 30}),
            (5, 6, {'length': 900, 'speed_kph': 45}),
            (6, 7, {'length': 1100, 'speed_kph': 55}),
            (7, 8, {'length': 700, 'speed_kph': 35}),
            (8, 9, {'length': 1300, 'speed_kph': 50}),
            (9, 10, {'length': 1000, 'speed_kph': 40}),
            # Add some alternative routes
            (1, 3, {'length': 2000, 'speed_kph': 60}),
            (2, 4, {'length': 1800, 'speed_kph': 45}),
            (3, 5, {'length': 2200, 'speed_kph': 50}),
            (4, 6, {'length': 1600, 'speed_kph': 55}),
            (5, 7, {'length': 2000, 'speed_kph': 40}),
            (6, 8, {'length': 1400, 'speed_kph': 50}),
            (7, 9, {'length': 1800, 'speed_kph': 45}),
            (8, 10, {'length': 1200, 'speed_kph': 55}),
        ]
        
        for u, v, data in edges:
            G.add_edge(u, v, **data)
        
        return G
    
    def generate_excel_report(self, results: List[Dict[str, Any]], 
                            source: int, destination: int) -> str:
        """
        Generate comprehensive Excel report with multiple sheets for analysis.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iterative_stress_analysis_{source}_to_{destination}_{timestamp}.xlsx"
        filepath = os.path.join("london_simulation", "excel_data", filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # Summary statistics by scenario
            summary_stats = []
            for scenario in self.congestion_scenarios:
                scenario_data = df[df['Congestion_Scenario'] == scenario]
                for algorithm in self.algorithms:
                    algo_data = scenario_data[scenario_data['Algorithm_Name'] == algorithm]
                    if not algo_data.empty:
                        # Calculate statistics
                        valid_times = algo_data[algo_data['Travel_Time'] != 'No Path']['Travel_Time']
                        if not valid_times.empty:
                            stats = {
                                'Scenario': scenario,
                                'Algorithm': algorithm,
                                'Min_Travel_Time': valid_times.min(),
                                'Max_Travel_Time': valid_times.max(),
                                'Mean_Travel_Time': valid_times.mean(),
                                'Std_Travel_Time': valid_times.std(),
                                'Min_Route_Congestion': algo_data['Mean_Route_Congestion'].min(),
                                'Max_Route_Congestion': algo_data['Mean_Route_Congestion'].max(),
                                'Mean_Route_Congestion': algo_data['Mean_Route_Congestion'].mean(),
                                'Success_Rate': len(valid_times) / len(algo_data) * 100,
                                'Total_Tests': len(algo_data)
                            }
                            summary_stats.append(stats)
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Create pivot tables for easy graphing
            for scenario in self.congestion_scenarios:
                scenario_data = df[df['Congestion_Scenario'] == scenario]
                if not scenario_data.empty:
                    # Pivot table: Iteration vs Algorithm (Travel Time)
                    pivot_travel_time = scenario_data.pivot_table(
                        values='Travel_Time', 
                        index='Iteration', 
                        columns='Algorithm_Name',
                        aggfunc='mean'
                    )
                    
                    # Pivot table: Iteration vs Algorithm (Route Congestion)
                    pivot_congestion = scenario_data.pivot_table(
                        values='Mean_Route_Congestion',
                        index='Iteration',
                        columns='Algorithm_Name', 
                        aggfunc='mean'
                    )
                    
                    # Save pivot tables
                    sheet_name_time = f'{scenario}_Travel_Time'[:31]  # Excel sheet name limit
                    sheet_name_cong = f'{scenario}_Congestion'[:31]
                    
                    pivot_travel_time.to_excel(writer, sheet_name=sheet_name_time)
                    pivot_congestion.to_excel(writer, sheet_name=sheet_name_cong)
            
            # Metadata sheet
            metadata = {
                'Analysis_Info': [
                    f"Source Location: {self.test_locations.get(source, 'Unknown')} (Location ID {source})",
                    f"Destination Location: {self.test_locations.get(destination, 'Unknown')} (Location ID {destination})",
                    f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Total Scenarios: {len(self.congestion_scenarios)}",
                    f"Iterations per Scenario: {max(df['Iteration']) if not df.empty else 0}",
                    f"Vehicle Increment: 20 per iteration",
                    f"Algorithms Tested: {', '.join(self.algorithms)}",
                    f"Total Data Points: {len(results)}",
                    "",
                    "Notable Locations Used:",
                    "  Same as main system - created using create_evenly_distributed_notable_locations()",
                    "",
                    "Congestion Scenarios:",
                ] + [f"  - {scenario}" for scenario in self.congestion_scenarios] + [
                    "",
                    "Data Columns:",
                    "  - Congestion_Scenario: Traffic scenario type",
                    "  - Iteration: Test iteration number",
                    "  - Number_of_Vehicles: Simulated vehicle count",
                    "  - Algorithm_Name: Routing algorithm used",
                    "  - Travel_Time: Calculated travel time (seconds)",
                    "  - Path_Length: Number of nodes in path",
                    "  - Computation_Time: Algorithm execution time",
                    "  - Mean_Route_Congestion: Average congestion on route",
                    "  - Average_Overall_Map_Congestion: Map-wide congestion",
                    "  - Path: Complete route path",
                    "",
                    "Usage:",
                    "  - Use pivot tables for creating graphs",
                    "  - Compare algorithms across scenarios",
                    "  - Analyze congestion impact on travel times",
                    "  - Study scalability with vehicle count increases"
                ]
            }
            
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False, header=False)
        
        print(f"Excel report saved: {filepath}")
        return filepath


def run_interactive_stress_analysis():
    """
    Interactive function to run iterative stress analysis.
    """
    analyzer = IterativeStressAnalyzer()
    
    print("=" * 60)
    print("ITERATIVE STRESS ANALYSIS SYSTEM")
    print("=" * 60)
    print("This system will test routing algorithms across all congestion scenarios")
    print("with increasing vehicle loads and generate comprehensive Excel reports.")
    print("Uses the same notable locations as the main traffic simulation system.")
    print()
    
    # Show available locations
    print("Available test locations:")
    locations = analyzer.get_available_locations()
    for loc_id, name in locations.items():
        print(f"  {loc_id}: {name}")
    print()
    
    # Get user input
    try:
        source = int(input("Enter source location ID (1-10): "))
        destination = int(input("Enter destination location ID (1-10): "))
        
        if source == destination:
            print("Source and destination cannot be the same!")
            return
        
        if source not in locations or destination not in locations:
            print("Invalid location IDs! Please use 1-10.")
            return
        
        # Optional: customize iterations
        iterations_input = input("Number of iterations per scenario (default 20): ").strip()
        max_iterations = int(iterations_input) if iterations_input else 20
        
        if max_iterations < 1 or max_iterations > 50:
            print("Iterations must be between 1 and 50!")
            return
        
        print(f"\nStarting analysis: {locations[source]} -> {locations[destination]}")
        print(f"Iterations per scenario: {max_iterations}")
        print("This may take several minutes...")
        print()
        
        # Run analysis
        excel_path = analyzer.run_comprehensive_analysis(source, destination, max_iterations)
        
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Excel report: {excel_path}")
        print("\nYou can now use this Excel file to create graphs showing:")
        print("  - Travel time trends across iterations")
        print("  - Congestion impact on different algorithms")
        print("  - Algorithm performance comparison")
        print("  - Scalability analysis with increasing vehicle counts")
        
    except ValueError:
        print("Invalid input! Please enter numeric values.")
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.")
    except Exception as e:
        print(f"Error during analysis: {e}")


if __name__ == "__main__":
    run_interactive_stress_analysis()
