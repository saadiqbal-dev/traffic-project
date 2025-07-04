"""
Main entry point for the London Traffic Simulation project.
Provides an interactive menu system for running traffic simulations.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import osmnx as ox
from typing import Dict, List, Optional

# Import all modules
from models import (
    Vehicle, load_london_network, generate_initial_congestion, 
    calculate_congestion_stats, create_evenly_distributed_notable_locations
)
from routing import calculate_all_routes
from congestion import (
    apply_consistent_congestion_scenario, update_congestion_based_on_vehicles,
    display_mm1_queueing_statistics, print_service_rate_degradation_summary
)
from visualization import (
    enhanced_visualize_congestion_map, debug_edge_data_structure, 
    test_edge_access_methods
)
from vehicle_management import (
    add_vehicle, add_multiple_vehicles_manual, add_bulk_vehicles,
    add_stress_test_vehicles, select_from_notable_locations,
    track_vehicle_congestion_impact, print_vehicle_impact_report
)
from analysis import (
    print_algorithm_comparison_table, simple_algorithm_comparison_table,
    print_travel_time_analysis_summary, recalculate_routes_for_selected_vehicles,
    export_analysis_to_excel, export_impact_report_to_excel, open_excel_file
)
from stress_testing import run_iterative_stress_test

# Advanced ML imports
try:
    from advanced_ml_integration import AdvancedMLManager, handle_advanced_ml_option
    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Advanced ML not available: {e}")
    ADVANCED_ML_AVAILABLE = False

# A* ML System imports
try:
    from astar_ml_system import AStarDataCollector, AStarPredictor, launch_gui
    ASTAR_ML_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  A* ML System not available: {e}")
    ASTAR_ML_AVAILABLE = False

# Set up environment
np.random.seed(42)

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.all_oneway = True

# Create output directories
output_dir = 'london_simulation'
excel_dir = os.path.join(output_dir, 'excel_data')
maps_dir = os.path.join(output_dir, 'scenario_maps')

os.makedirs(output_dir, exist_ok=True)
os.makedirs(excel_dir, exist_ok=True)
os.makedirs(maps_dir, exist_ok=True)


def run_interactive_congestion_simulation():
    """Run the interactive congestion simulation with vehicles and route analysis."""
    try:
        from IPython.display import clear_output
    except ImportError:
        def clear_output(wait=True):
            """Fallback clear_output function."""
            pass
    
    print("Starting London congestion simulation with vehicles and route analysis...")
    
    # Load the London network
    G = load_london_network()
    
    # Generate initial congestion
    congestion_data = generate_initial_congestion(G)
    
    # Save original congestion for reset
    original_congestion = congestion_data.copy()
    
    # Create evenly distributed notable locations
    notable_locations = create_evenly_distributed_notable_locations(G)
    print("\nNotable locations created:")
    for name, node in notable_locations.items():
        print(f"  {name}: Node {node}")
    
    # Initialize vehicles list
    vehicles = []
    
    # Initialize Advanced ML Manager
    ml_manager = None
    if ADVANCED_ML_AVAILABLE:
        try:
            ml_manager = AdvancedMLManager()
            print("🚀 Advanced ML system ready! Use AML options for cutting-edge features.")
        except Exception as e:
            print(f"⚠️  Advanced ML initialization failed: {e}")
            ml_manager = None
    
    # Define scenarios
    scenarios = ["Normal", "Morning", "Evening", "Weekend", "Special"]
    special_case_types = ["Sports Event", "Concert", "Holiday", "Roadwork", "Weather Event", "Random"]
    
    # Initial visualization with normal traffic
    current_scenario = "Normal"
    excel_file = None
    congestion_data, excel_file = apply_consistent_congestion_scenario(G, congestion_data, current_scenario, original_congestion)
    enhanced_visualize_congestion_map(G, congestion_data, vehicles, current_scenario, notable_locations)
    
    # Print congestion statistics
    stats = calculate_congestion_stats(congestion_data)
    print("\nCurrent congestion statistics:")
    for key, value in stats.items():
        print(f"  {key.capitalize()}: {value:.4f}")
    
    # Track the latest Excel file for each scenario
    excel_files = {scenario: None for scenario in scenarios}
    excel_files[current_scenario] = excel_file
    
    # Track the latest impact report
    latest_impact_report = None
    latest_impact_excel = None
    
    # Interactive loop
    while True:
        print("\n=== London Traffic Simulation Menu ===")
        print("Congestion Scenarios:")
        for i, scenario in enumerate(scenarios):
            print(f"{i+1}. Apply {scenario} congestion scenario")
        
        print("\nVehicle Management:")
        print("A. Add a single vehicle with custom source/destination")
        print("B. Add 50 random vehicles")
        print("C. Add 100 random vehicles")
        print("D. Add 200 random vehicles")
        print("E. Remove all vehicles")
        print("S. Add multiple vehicles between specific source/destination")
        
        print("\nRoute Analysis:")
        print("F. Select a vehicle for route analysis")
        print("G. Deselect all vehicles from analysis")
        print("H. Calculate routes for selected vehicles")
        print("I. Compare algorithms under current congestion")
        
        print("\nCongestion Impact Analysis:")
        print("L. Calculate and display vehicle impact on congestion")
        print("M. Export detailed impact report to Excel")
        
        print("\nData Management:")
        print("J. Open latest Excel file for current scenario")
        print("K. Generate new Excel file without changing scenario")
        print("N. Open latest impact report Excel file")
        print("T. View travel time analysis summary")
        print("DEBUG. Debug edge data structure")
        print("W. View network-wide service rate degradation summary")
        
        print("\nData Export:")
        print("X. Export comprehensive analysis to Excel")
        
        print("\nTraffic Analysis:")
        print("Y. View MM1 queuing statistics for an edge")
        print("Z. Run iterative stress test on a selected vehicle")
        
        print("\nA* Machine Learning (Simplified System):")
        print("ASML1. Collect A* training data")
        print("ASML2. Train A* prediction model")
        print("ASML3. Launch A* ML GUI")
        print("ASML4. Quick A* ML demo")
        
        print("\nMachine Learning (Legacy):")
        print("ML1. Collect ML training data")
        print("ML2. Train ML models")
        print("ML3. Run ML prediction demo")
        print("ML4. Compare ML vs actual for selected vehicle")
        print("ML5. Collect congestion prediction data")
        print("ML6. Train congestion prediction models")
        print("ML7. Predict network congestion")
        print("ML8. Collect route optimization data")
        print("ML9. Train route optimization models")
        print("ML10. AI-powered route recommendation")
        
        print("\nAdvanced ML (GNN & Deep Learning):")
        print("AML1. Initialize Advanced ML System")
        print("AML2. Train GNN & Advanced ML Pipeline")
        print("AML3. GNN-Enhanced Route Calculation")
        print("AML4. Predict Network Congestion with GNN")
        print("AML5. Advanced Algorithm Comparison")
        print("AML6. Generate ML Insights Report")
        print("AML7. Test Advanced ML Features")
        
        print("\nOther Options:")
        print("R. Reset to original congestion")
        print("0. Exit simulation")
        
        choice = input("\nSelect an option (letter/number): ").strip().upper()
        
        if choice == '0':
            print("Exiting simulation...")
            break
            
        # Handle reset option
        elif choice == 'R':
            print("Resetting to original congestion...")
            congestion_data = original_congestion.copy()
            for u, v, k in G.edges(keys=True):
                edge_id = f"{u}_{v}_{k}"
                if edge_id in original_congestion:
                    G[u][v][k]['congestion'] = original_congestion[edge_id]
                    G[u][v][k]['vehicle_count'] = 0
            vehicles = []
            current_scenario = "Normal"
            
        # Handle vehicle management options
        elif choice == 'A':
            # Add a single vehicle
            source = select_from_notable_locations(notable_locations, "Select source location:")
            if source:
                destination = select_from_notable_locations(notable_locations, "Select destination location:")
                if destination and source != destination:
                    # Ask whether to calculate routes
                    calc_routes = input("Calculate routes using all algorithms? (y/n): ").strip().lower() == 'y'
                    
                    vehicle = add_vehicle(G, vehicles, source, destination, congestion_data, calc_routes)
                    if vehicle:
                        # Update congestion based on vehicles
                        update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
                        
                        # Ask if the vehicle should be selected for analysis
                        select_for_analysis = input("Select this vehicle for route analysis? (y/n): ").strip().lower() == 'y'
                        if select_for_analysis:
                            vehicle.selected_for_analysis = True
                            print(f"Vehicle {vehicle.id} selected for analysis")
                        
                        # Offer stress testing
                        stress_test = input("Would you like to stress test this route with 20 random vehicles? (y/n): ").strip().lower() == 'y'
                        if stress_test:
                            # First, visualize the current route
                            enhanced_visualize_congestion_map(G, congestion_data, vehicles, current_scenario, notable_locations)
                            print("\nCurrent route before stress testing:")
                            try:
                                print_algorithm_comparison_table(G, vehicle, congestion_data)
                            except Exception as e:
                                print(f"Warning: Complex comparison failed, using simple version: {e}")
                                simple_algorithm_comparison_table(G, vehicle, congestion_data)
                            
                            # Add stress test vehicles
                            added_count, added_vehicles = add_stress_test_vehicles(G, vehicles, vehicle, congestion_data, original_congestion)
                            
                            # Visualize again to show the change
                            enhanced_visualize_congestion_map(G, congestion_data, vehicles, current_scenario, notable_locations)
                            print("\nRecalculated route after adding stress test vehicles:")
                            try:
                                print_algorithm_comparison_table(G, vehicle, congestion_data)
                            except Exception as e:
                                print(f"Warning: Complex comparison failed, using simple version: {e}")
                                simple_algorithm_comparison_table(G, vehicle, congestion_data)
                else:
                    print("Invalid destination or same as source.")
            else:
                print("Invalid source selection.")
                
        elif choice == 'Z':
            # Run iterative stress test
            if not vehicles:
                print("No vehicles available for stress testing")
                continue
                
            print("\nSelect a vehicle for iterative stress testing:")
            for i, v in enumerate(vehicles):
                print(f"{i+1}. Vehicle {v.id}: {v.source} -> {v.destination}")
            
            try:
                vehicle_idx = int(input("Select vehicle (number): ")) - 1
                if 0 <= vehicle_idx < len(vehicles):
                    # Calculate routes if not already done
                    if not vehicles[vehicle_idx].paths:
                        print("Calculating routes before stress testing...")
                        calculate_all_routes(G, vehicles[vehicle_idx], congestion_data)
                    
                    # Run the stress test
                    run_iterative_stress_test(G, vehicles, vehicles[vehicle_idx], 
                                             congestion_data, original_congestion, current_scenario)
                else:
                    print("Invalid vehicle number")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == 'B':
            # Add 50 vehicles
            add_bulk_vehicles(G, vehicles, 50, congestion_data, False, notable_locations)
            update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
            
            # Calculate and display the impact
            impact_report = track_vehicle_congestion_impact(G, vehicles, congestion_data, original_congestion)
            latest_impact_report = impact_report
            print_vehicle_impact_report(impact_report)
            
        elif choice == 'DEBUG':
            debug_edge_data_structure(G, 3)
            test_edge_access_methods(G)
            
        elif choice == 'C':
            # Add 100 vehicles
            add_bulk_vehicles(G, vehicles, 100, congestion_data, False, notable_locations)
            update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
            
            # Calculate and display the impact
            impact_report = track_vehicle_congestion_impact(G, vehicles, congestion_data, original_congestion)
            latest_impact_report = impact_report
            print_vehicle_impact_report(impact_report)
            
        elif choice == 'D':
            # Add 200 vehicles
            add_bulk_vehicles(G, vehicles, 200, congestion_data, False, notable_locations)
            update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
            
            # Calculate and display the impact
            impact_report = track_vehicle_congestion_impact(G, vehicles, congestion_data, original_congestion)
            latest_impact_report = impact_report
            print_vehicle_impact_report(impact_report)
            
        elif choice == 'E':
            # Remove all vehicles
            print(f"Removing all {len(vehicles)} vehicles...")
            
            # Reset vehicle counts on all edges
            for u, v, k in G.edges(keys=True):
                G[u][v][k]['vehicle_count'] = 0
            
            vehicles = []
            
            # Restore original congestion before applying scenario
            congestion_data = original_congestion.copy()
            for u, v, k in G.edges(keys=True):
                edge_id = f"{u}_{v}_{k}"
                if edge_id in original_congestion:
                    G[u][v][k]['congestion'] = original_congestion[edge_id]
            
            # Re-apply current scenario without vehicle influence
            congestion_data, excel_file = apply_consistent_congestion_scenario(G, congestion_data, 
                                                 current_scenario, original_congestion)
            excel_files[current_scenario] = excel_file
            
            # Reset impact report
            latest_impact_report = None
            
        # Handle route analysis options
        elif choice == 'F':
            # Select a vehicle for analysis
            if not vehicles:
                print("No vehicles available to select")
                continue
                
            print("\nAvailable vehicles:")
            for i, v in enumerate(vehicles):
                status = " (Selected)" if v.selected_for_analysis else ""
                print(f"{i+1}. Vehicle {v.id}: {v.source} -> {v.destination}{status}")
            
            try:
                vehicle_idx = int(input("Select vehicle (number): ")) - 1
                if 0 <= vehicle_idx < len(vehicles):
                    vehicles[vehicle_idx].selected_for_analysis = True
                    print(f"Vehicle {vehicles[vehicle_idx].id} selected for analysis")
                    
                    # Ask whether to calculate routes if not already done
                    if not vehicles[vehicle_idx].paths:
                        calc_routes = input("Calculate routes using all algorithms now? (y/n): ").strip().lower() == 'y'
                        if calc_routes:
                            calculate_all_routes(G, vehicles[vehicle_idx], congestion_data)
                else:
                    print("Invalid vehicle number")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == 'G':
            # Deselect all vehicles
            selected_count = sum(1 for v in vehicles if v.selected_for_analysis)
            if selected_count > 0:
                print(f"Deselecting {selected_count} vehicles from analysis")
                for v in vehicles:
                    v.selected_for_analysis = False
            else:
                print("No vehicles were selected for analysis")
                
        elif choice == 'T':
            # View travel time analysis summary
            if vehicles:
                print_travel_time_analysis_summary(G, vehicles, congestion_data)
            else:
                print("No vehicles available for travel time analysis")

        elif choice == 'W':
            # View network-wide service rate degradation
            if vehicles:
                print_service_rate_degradation_summary(G, congestion_data)
            else:
                print("No vehicles available for service rate analysis")

        elif choice == 'X':
            # Export comprehensive analysis to Excel
            selected_v = None
            if vehicles:
                # Ask if user wants to include detailed analysis for a specific vehicle
                include_detail = input("Include detailed analysis for a specific vehicle? (y/n): ").strip().lower() == 'y'
                if include_detail:
                    print("\nAvailable vehicles:")
                    for i, v in enumerate(vehicles):
                        status = " (Selected)" if v.selected_for_analysis else ""
                        print(f"{i+1}. Vehicle {v.id}: {v.source} -> {v.destination}{status}")
                    
                    try:
                        vehicle_idx = int(input("Select vehicle for detailed analysis (number): ")) - 1
                        if 0 <= vehicle_idx < len(vehicles):
                            selected_v = vehicles[vehicle_idx]
                        else:
                            print("Invalid vehicle number, proceeding without detailed analysis")
                    except ValueError:
                        print("Invalid input, proceeding without detailed analysis")
            
            # Export the analysis
            excel_file = export_analysis_to_excel(G, vehicles, congestion_data, current_scenario, selected_v)
            
            # Ask if the user wants to open it
            if input("Open the Excel file now? (y/n): ").strip().lower() == 'y':
                open_excel_file(excel_file)
                
        elif choice == 'H':
            # Calculate routes for selected vehicles
            selected_vehicles = [v for v in vehicles if v.selected_for_analysis]
            if selected_vehicles:
                for v in selected_vehicles:
                    calculate_all_routes(G, v, congestion_data)
                print(f"Calculated routes for {len(selected_vehicles)} selected vehicles")
            else:
                print("No vehicles selected for analysis")
                
        elif choice == 'I':
            # Compare algorithms under current congestion
            recalculate_routes_for_selected_vehicles(G, vehicles, congestion_data, current_scenario)
        
        # Handle impact analysis options
        elif choice == 'L':
            # Calculate and display vehicle impact on congestion
            if vehicles:
                impact_report = track_vehicle_congestion_impact(G, vehicles, congestion_data, original_congestion)
                latest_impact_report = impact_report
                print_vehicle_impact_report(impact_report)
            else:
                print("No vehicles available for impact analysis")
        
        elif choice == 'M':
            # Export detailed impact report to Excel
            if vehicles and latest_impact_report:
                latest_impact_excel = export_impact_report_to_excel(G, vehicles, congestion_data, 
                                                        original_congestion, current_scenario, latest_impact_report)
                
                # Ask if the user wants to open it
                if input("Open the impact report Excel file now? (y/n): ").strip().lower() == 'y':
                    open_excel_file(latest_impact_excel)
            else:
                print("No impact report available. Please add vehicles and calculate impact first (option L)")
        
        # Handle data management options
        elif choice == 'J':
            # Open latest Excel file for current scenario
            excel_file = excel_files.get(current_scenario)
            if excel_file and os.path.exists(excel_file):
                print(f"Opening Excel file for {current_scenario} scenario: {excel_file}")
                open_excel_file(excel_file)
            else:
                print(f"No Excel file available for {current_scenario} scenario.")
        
        elif choice == 'K':
            # Generate new Excel file without changing scenario
            from congestion import export_congestion_to_excel
            print(f"Generating new Excel file for {current_scenario} scenario...")
            excel_file, _ = export_congestion_to_excel(G, congestion_data, current_scenario)
            excel_files[current_scenario] = excel_file
            print(f"Generated new Excel file: {excel_file}")
            
            # Ask if the user wants to open it
            if input("Open the Excel file now? (y/n): ").strip().lower() == 'y':
                open_excel_file(excel_file)
        
        elif choice == 'N':
            # Open latest impact report Excel file
            if latest_impact_excel and os.path.exists(latest_impact_excel):
                print(f"Opening impact report Excel file: {latest_impact_excel}")
                open_excel_file(latest_impact_excel)
            else:
                print("No impact report Excel file available. Generate one first (option M)")
        
        # Handle MM1 queuing statistics
        elif choice == 'Y':
            # View MM1 queuing statistics for an edge
            print("\nSelect method to identify edge:")
            print("1. Select source and destination nodes")
            print("2. Enter edge ID directly")
            
            try:
                method = int(input("Select method (1 or 2): "))
                
                if method == 1:
                    # Select nodes
                    source = select_from_notable_locations(notable_locations, "Select source node:")
                    if source:
                        dest = select_from_notable_locations(notable_locations, "Select destination node:")
                        if dest and source != dest:
                            display_mm1_queueing_statistics(G, node1=source, node2=dest)
                        else:
                            print("Invalid destination selection.")
                    else:
                        print("Invalid source selection.")
                elif method == 2:
                    # Enter edge ID directly
                    edge_id = input("Enter edge ID (format: node1_node2_key): ")
                    display_mm1_queueing_statistics(G, edge_id=edge_id)
                else:
                    print("Invalid method selection.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif choice == 'S':
            # Add multiple vehicles between specific source/destination
            source = select_from_notable_locations(notable_locations, "Select source location:")
            if source:
                destination = select_from_notable_locations(notable_locations, "Select destination location:")
                if destination and source != destination:
                    try:
                        count = int(input("How many vehicles to add (1-20): "))
                        if 1 <= count <= 20:
                            # Ask whether to calculate routes
                            calc_routes = input("Calculate routes using all algorithms? (y/n): ").strip().lower() == 'y'
                            
                            add_multiple_vehicles_manual(G, vehicles, source, destination, count, congestion_data, calc_routes)
                            
                            # Update congestion based on vehicles
                            update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
                            
                            # Calculate and display the impact
                            impact_report = track_vehicle_congestion_impact(G, vehicles, congestion_data, original_congestion)
                            latest_impact_report = impact_report
                            print_vehicle_impact_report(impact_report)
                        else:
                            print("Invalid number of vehicles. Please enter a value between 1 and 20.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                else:
                    print("Invalid destination or same as source.")
            else:
                print("Invalid source selection.")
        
        # Handle A* Machine Learning options
        elif choice == 'ASML1':
            # Collect A* training data
            print("\n🚀 Starting A* ML Data Collection...")
            if not ASTAR_ML_AVAILABLE:
                print("❌ A* ML System not available.")
                continue
                
            try:
                import os
                collector = AStarDataCollector()
                samples_per_scenario = int(input("Number of samples per scenario (recommended: 30-50): ") or "30")
                
                print(f"This will collect A* data for:")
                print(f"  - {len(collector.scenarios)} scenarios")
                print(f"  - {len(collector.vehicle_counts)} vehicle count variations")
                print(f"  - {samples_per_scenario} samples per scenario")
                
                confirm = input("Proceed with A* data collection? (y/n): ").strip().lower()
                if confirm == 'y':
                    data_file = collector.collect_astar_data(samples_per_scenario)
                    print(f"✅ A* data collection complete! Data saved to: {data_file}")
                else:
                    print("Data collection cancelled.")
                    
            except Exception as e:
                print(f"❌ Error during A* data collection: {e}")
        
        elif choice == 'ASML2':
            # Train A* prediction model
            print("\n🔥 Starting A* Model Training...")
            if not ASTAR_ML_AVAILABLE:
                print("❌ A* ML System not available.")
                continue
                
            try:
                # Explicitly import required modules to avoid scope issues
                import os
                import time
                
                # Find latest data file
                data_dir = os.path.join('london_simulation', 'astar_ml_data')
                if not os.path.exists(data_dir):
                    print("❌ No A* training data found. Run ASML1 first.")
                    continue
                    
                data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
                if not data_files:
                    print("❌ No A* training data files found. Run ASML1 first.")
                    continue
                    
                latest_data = os.path.join(data_dir, sorted(data_files)[-1])
                print(f"Using training data: {os.path.basename(latest_data)}")
                
                predictor = AStarPredictor()
                metrics = predictor.train(latest_data)
                
                # Save model
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_file = os.path.join(predictor.model_dir, f"astar_model_{timestamp}.pkl")
                predictor.save_model(model_file)
                
                print("✅ A* model training complete!")
                print(f"   Model saved to: {model_file}")
                    
            except Exception as e:
                print(f"❌ Error during A* model training: {e}")
        
        elif choice == 'ASML3':
            # Launch A* ML GUI
            print("\n🖥️  Launching A* ML GUI...")
            if not ASTAR_ML_AVAILABLE:
                print("❌ A* ML System not available.")
                continue
                
            try:
                launch_gui()
            except Exception as e:
                print(f"❌ Error launching A* ML GUI: {e}")
        
        elif choice == 'ASML4':
            # Quick A* ML demo
            print("\n🚀 Running Quick A* ML Demo...")
            if not ASTAR_ML_AVAILABLE:
                print("❌ A* ML System not available.")
                continue
                
            try:
                import os
                # Quick data collection
                print("Step 1: Collecting sample data...")
                collector = AStarDataCollector()
                data_file = collector.collect_astar_data(10)  # Small sample for demo
                
                # Quick training
                print("Step 2: Training model...")
                predictor = AStarPredictor()
                metrics = predictor.train(data_file)
                
                # Quick prediction
                print("Step 3: Making sample prediction...")
                locations = list(notable_locations.keys())
                if len(locations) >= 2:
                    result = predictor.predict_optimal_route(
                        locations[0], locations[1], 12, "Normal"
                    )
                    
                    print(f"\n✅ Demo Results:")
                    print(f"   Route: {locations[0]} → {locations[1]}")
                    print(f"   Predicted time: {result['predicted_time']:.1f}s")
                    print(f"   Actual A* time: {result['actual_time']:.1f}s")
                    print(f"   Accuracy: {100 - result['accuracy']:.1f}%")
                else:
                    print("❌ Not enough locations for demo.")
                    
            except Exception as e:
                print(f"❌ Error during A* ML demo: {e}")
        
        # Handle Machine Learning options
        elif choice == 'ML1':
            # Collect ML training data
            print("\n🚀 Starting ML Data Collection...")
            try:
                from ml_data_collector import MLDataCollector
                
                collector = MLDataCollector()
                samples_per_config = int(input("Number of samples per configuration (recommended: 10-20): ") or "10")
                
                print(f"This will collect data for:")
                print(f"  - {len(collector.scenarios)} scenarios")
                print(f"  - {len(collector.vehicle_counts)} vehicle count variations")
                print(f"  - {samples_per_config} samples each")
                print(f"  - Estimated total: {len(collector.scenarios) * len(collector.vehicle_counts) * samples_per_config} samples")
                
                confirm = input("Proceed with data collection? (y/n): ").strip().lower()
                if confirm == 'y':
                    df = collector.collect_all_data(samples_per_config)
                    print(f"✅ Data collection complete! Collected {len(df)} samples.")
                else:
                    print("Data collection cancelled.")
                    
            except ImportError:
                print("❌ ML modules not available. Check ml_data_collector.py")
            except Exception as e:
                print(f"❌ Error during data collection: {e}")
        
        elif choice == 'ML2':
            # Train ML models
            print("\n🔥 Starting ML Model Training...")
            try:
                from ml_models import MLTrainingPipeline
                
                # Check for training data
                import os
                data_dir = 'london_simulation'
                if os.path.exists(data_dir):
                    data_files = [f for f in os.listdir(data_dir) 
                                if f.startswith('ml_training_data_') and f.endswith('.csv')]
                    
                    if data_files:
                        latest_data = sorted(data_files)[-1]
                        data_path = os.path.join(data_dir, latest_data)
                        
                        print(f"📂 Using training data: {latest_data}")
                        
                        epochs = int(input("Number of training epochs (recommended: 50-100): ") or "50")
                        
                        pipeline = MLTrainingPipeline()
                        results = pipeline.train_both_models(data_path, epochs=epochs)
                        
                        print("✅ ML model training complete!")
                    else:
                        print("❌ No training data found. Run ML1 first to collect data.")
                else:
                    print("❌ Simulation data directory not found.")
                    
            except ImportError as e:
                print(f"❌ ML training not available: {e}")
                print("Install TensorFlow: pip install tensorflow")
            except Exception as e:
                print(f"❌ Error during model training: {e}")
        
        elif choice == 'ML3':
            # Run ML prediction demo
            print("\n🧪 Running ML Prediction Demo...")
            try:
                from ml_integration import run_ml_demo
                
                run_ml_demo(G, congestion_data, notable_locations, current_scenario, len(vehicles))
                
            except ImportError:
                print("❌ ML integration modules not available.")
            except Exception as e:
                print(f"❌ Error during ML demo: {e}")
        
        elif choice == 'ML4':
            # Compare ML vs actual for selected vehicle
            print("\n🔬 ML vs Actual Comparison")
            print(f"DEBUG: Current vehicles count: {len(vehicles)}")
            
            if not vehicles:
                print("❌ No vehicles available. Add vehicles first.")
                continue
            
            print("Select a vehicle to test ML predictions:")
            for i, v in enumerate(vehicles):
                print(f"{i+1}. Vehicle {v.id}: {v.source} -> {v.destination}")
            
            try:
                user_input = input("Select vehicle (number): ").strip()
                print(f"DEBUG: User input: '{user_input}'")
                
                if not user_input:
                    print("❌ No input provided.")
                    continue
                
                vehicle_idx = int(user_input) - 1
                print(f"DEBUG: Vehicle index: {vehicle_idx}")
                
                if 0 <= vehicle_idx < len(vehicles):
                    print("DEBUG: Valid vehicle selected, importing MLPredictor...")
                    from ml_integration import MLPredictor
                    
                    print("DEBUG: Creating predictor...")
                    predictor = MLPredictor()
                    
                    print("DEBUG: Loading models...")
                    if predictor.load_latest_models():
                        print("DEBUG: Models loaded, running comparison...")
                        comparison = predictor.compare_prediction_vs_actual(
                            G, vehicles[vehicle_idx], congestion_data, current_scenario, len(vehicles)
                        )
                        print("DEBUG: Comparison completed successfully!")
                    else:
                        print("❌ Could not load ML models. Train models first (ML2).")
                else:
                    print(f"❌ Invalid vehicle number. Please enter 1-{len(vehicles)}")
            except ValueError as e:
                print(f"❌ Invalid input. Please enter a number. Error: {e}")
            except ImportError as e:
                print(f"❌ ML integration modules not available. Error: {e}")
            except Exception as e:
                print(f"❌ Error during ML comparison: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == 'ML5':
            # Collect congestion prediction data
            print("\n🔮 Starting Congestion Prediction Data Collection...")
            try:
                from congestion_prediction import CongestionDataCollector
                
                collector = CongestionDataCollector()
                samples_per_config = int(input("Number of samples per configuration (recommended: 10-15): ") or "10")
                
                print(f"This will collect data for:")
                print(f"  - {len(collector.scenarios)} scenarios")
                print(f"  - {len(collector.vehicle_counts)} vehicle count variations")
                print(f"  - {samples_per_config} samples each")
                print(f"  - Estimated total: {len(collector.scenarios) * len(collector.vehicle_counts) * samples_per_config} configurations")
                
                confirm = input("Proceed with congestion data collection? (y/n): ").strip().lower()
                if confirm == 'y':
                    df = collector.collect_congestion_data(samples_per_config)
                    print(f"✅ Congestion data collection complete! Collected {len(df)} samples.")
                else:
                    print("Data collection cancelled.")
                    
            except ImportError:
                print("❌ Congestion prediction modules not available. Check congestion_prediction.py")
            except Exception as e:
                print(f"❌ Error during congestion data collection: {e}")
        
        elif choice == 'ML6':
            # Train congestion prediction models
            print("\n🧠 Training Congestion Prediction Models...")
            try:
                from congestion_prediction import CongestionPredictionModel
                import os
                
                # Check for training data
                data_dir = 'london_simulation'
                if os.path.exists(data_dir):
                    data_files = [f for f in os.listdir(data_dir) 
                                if f.startswith('congestion_prediction_data_') and f.endswith('.csv')]
                    
                    if data_files:
                        latest_data = sorted(data_files)[-1]
                        data_path = os.path.join(data_dir, latest_data)
                        
                        print(f"📂 Using training data: {latest_data}")
                        
                        # Train Random Forest model
                        rf_model = CongestionPredictionModel("random_forest")
                        rf_metrics = rf_model.train(pd.read_csv(data_path))
                        
                        # Train Gradient Boosting model
                        gb_model = CongestionPredictionModel("gradient_boosting")
                        gb_metrics = gb_model.train(pd.read_csv(data_path))
                        
                        # Save models
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        rf_path = os.path.join(rf_model.models_dir, f"congestion_random_forest_{timestamp}.pkl")
                        gb_path = os.path.join(gb_model.models_dir, f"congestion_gradient_boosting_{timestamp}.pkl")
                        
                        rf_model.save_model(rf_path)
                        gb_model.save_model(gb_path)
                        
                        print("✅ Congestion prediction models training complete!")
                        print(f"Random Forest - R²: {rf_metrics['r2_score']:.4f}")
                        print(f"Gradient Boosting - R²: {gb_metrics['r2_score']:.4f}")
                    else:
                        print("❌ No congestion training data found. Run ML5 first to collect data.")
                else:
                    print("❌ Simulation data directory not found.")
                    
            except ImportError as e:
                print(f"❌ Congestion prediction not available: {e}")
            except Exception as e:
                print(f"❌ Error during congestion model training: {e}")
        
        elif choice == 'ML7':
            # Predict network congestion
            print("\n🔮 AI-Powered Network Congestion Prediction...")
            try:
                from congestion_prediction import CongestionPredictor
                
                predictor = CongestionPredictor()
                
                if predictor.load_models():
                    # Get prediction parameters
                    print("Select prediction scenario:")
                    for i, scenario in enumerate(scenarios):
                        print(f"{i+1}. {scenario}")
                    
                    try:
                        scenario_idx = int(input("Select scenario (number): ")) - 1
                        if 0 <= scenario_idx < len(scenarios):
                            pred_scenario = scenarios[scenario_idx]
                            
                            pred_vehicle_count = int(input(f"Vehicle count for prediction (current: {len(vehicles)}): ") or str(len(vehicles)))
                            pred_hour = int(input("Hour of day (0-23): ") or "12")
                            pred_weather = input("Weather (clear/rain/fog): ").strip().lower() or "clear"
                            
                            # Predict congestion
                            predicted_congestion = predictor.predict_network_congestion(
                                G, congestion_data, pred_scenario, pred_vehicle_count, pred_hour, pred_weather
                            )
                            
                            # Apply predictions to network (temporarily)
                            original_current = congestion_data.copy()
                            for edge_id, pred_level in predicted_congestion.items():
                                congestion_data[edge_id] = pred_level
                                
                                # Update graph
                                parts = edge_id.split('_')
                                if len(parts) >= 3:
                                    u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
                                    if u in G and v in G[u] and k in G[u][v]:
                                        G[u][v][k]['congestion'] = pred_level
                            
                            print(f"✅ Applied AI congestion predictions for {pred_scenario} scenario")
                            print(f"   Vehicle count: {pred_vehicle_count}, Hour: {pred_hour}, Weather: {pred_weather}")
                            
                            # Ask if user wants to revert
                            if input("Keep predictions or revert to original? (keep/revert): ").strip().lower() == 'revert':
                                congestion_data.update(original_current)
                                for edge_id, orig_level in original_current.items():
                                    parts = edge_id.split('_')
                                    if len(parts) >= 3:
                                        u, v, k = int(parts[0]), int(parts[1]), int(parts[2])
                                        if u in G and v in G[u] and k in G[u][v]:
                                            G[u][v][k]['congestion'] = orig_level
                                print("Reverted to original congestion levels.")
                        else:
                            print("Invalid scenario selection.")
                    except ValueError:
                        print("Invalid input. Please enter numbers.")
                else:
                    print("❌ Could not load congestion prediction models. Train models first (ML6).")
                    
            except ImportError:
                print("❌ Congestion prediction modules not available.")
            except Exception as e:
                print(f"❌ Error during congestion prediction: {e}")
        
        elif choice == 'ML8':
            # Collect route optimization data
            print("\n🛣️ Starting Route Optimization Data Collection...")
            try:
                from route_optimization import RouteDataCollector
                
                collector = RouteDataCollector()
                samples_per_config = int(input("Number of samples per configuration (recommended: 10-15): ") or "10")
                
                print(f"This will collect data for:")
                print(f"  - {len(collector.scenarios)} scenarios")
                print(f"  - {len(collector.vehicle_counts)} vehicle count variations")
                print(f"  - {samples_per_config} samples each")
                print(f"  - Route comparisons: A* vs Shortest Path algorithms")
                
                confirm = input("Proceed with route optimization data collection? (y/n): ").strip().lower()
                if confirm == 'y':
                    df = collector.collect_route_optimization_data(samples_per_config)
                    print(f"✅ Route optimization data collection complete! Collected {len(df)} samples.")
                else:
                    print("Data collection cancelled.")
                    
            except ImportError:
                print("❌ Route optimization modules not available. Check route_optimization.py")
            except Exception as e:
                print(f"❌ Error during route optimization data collection: {e}")
        
        elif choice == 'ML9':
            # Train route optimization models
            print("\n🚀 Training Route Optimization Models...")
            try:
                from route_optimization import RouteOptimizationModel
                import os
                
                # Check for training data
                data_dir = 'london_simulation'
                if os.path.exists(data_dir):
                    data_files = [f for f in os.listdir(data_dir) 
                                if f.startswith('route_optimization_data_') and f.endswith('.csv')]
                    
                    if data_files:
                        latest_data = sorted(data_files)[-1]
                        data_path = os.path.join(data_dir, latest_data)
                        
                        print(f"📂 Using training data: {latest_data}")
                        
                        epochs = int(input("Number of training epochs (recommended: 50-100): ") or "50")
                        
                        # Train algorithm selection model
                        print("\n🎯 Training Algorithm Selection Model...")
                        selector_model = RouteOptimizationModel("algorithm_selector")
                        selector_metrics = selector_model.train(pd.read_csv(data_path), epochs=epochs)
                        
                        # Train travel time prediction model
                        print("\n⏱️ Training Travel Time Prediction Model...")
                        time_model = RouteOptimizationModel("travel_time_predictor")
                        time_metrics = time_model.train(pd.read_csv(data_path), epochs=epochs)
                        
                        # Save models
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        selector_path = os.path.join(selector_model.models_dir, f"algorithm_selector_{timestamp}")
                        time_path = os.path.join(time_model.models_dir, f"travel_time_predictor_{timestamp}")
                        
                        selector_model.save_model(selector_path)
                        time_model.save_model(time_path)
                        
                        print("✅ Route optimization models training complete!")
                        print(f"Algorithm Selector - Accuracy: {selector_metrics['accuracy']:.4f}")
                        print(f"Travel Time Predictor - RMSE: {time_metrics['rmse']:.2f}s")
                    else:
                        print("❌ No route optimization training data found. Run ML8 first to collect data.")
                else:
                    print("❌ Simulation data directory not found.")
                    
            except ImportError as e:
                print(f"❌ Route optimization training not available: {e}")
                print("Install TensorFlow: pip install tensorflow")
            except Exception as e:
                print(f"❌ Error during route optimization model training: {e}")
        
        elif choice == 'ML10':
            # AI-powered route recommendation
            print("\n🤖 AI-Powered Route Recommendation System")
            
            if not vehicles:
                print("❌ No vehicles available. Add vehicles first.")
                continue
            
            try:
                from route_optimization import RouteOptimizationModel, RouteDataCollector
                
                print("Select a vehicle for AI route analysis:")
                for i, v in enumerate(vehicles):
                    print(f"{i+1}. Vehicle {v.id}: {v.source} -> {v.destination}")
                
                vehicle_idx = int(input("Select vehicle (number): ")) - 1
                if 0 <= vehicle_idx < len(vehicles):
                    selected_vehicle = vehicles[vehicle_idx]
                    
                    # Load route optimization model
                    print("Loading AI route optimization models...")
                    
                    # Check for trained models
                    models_dir = os.path.join('london_simulation', 'route_models')
                    if os.path.exists(models_dir):
                        model_files = [f for f in os.listdir(models_dir) if f.endswith('_data.pkl')]
                        selector_files = [f for f in model_files if 'algorithm_selector' in f]
                        
                        if selector_files:
                            latest_selector = sorted(selector_files)[-1]
                            selector_path = os.path.join(models_dir, latest_selector.replace('_data.pkl', ''))
                            
                            # Load model
                            ai_model = RouteOptimizationModel("algorithm_selector")
                            ai_model.load_model(selector_path)
                            
                            # Extract features for the selected vehicle
                            collector = RouteDataCollector()
                            features = collector.extract_route_features(
                                G, selected_vehicle, congestion_data, current_scenario, len(vehicles)
                            )
                            
                            # Get AI recommendation
                            recommended_algo, confidence = ai_model.predict_best_algorithm(features)
                            
                            print(f"\n🤖 AI Route Recommendation for Vehicle {selected_vehicle.id}:")
                            print(f"   Route: {selected_vehicle.source} -> {selected_vehicle.destination}")
                            print(f"   Recommended Algorithm: {recommended_algo.upper()}")
                            print(f"   Confidence: {confidence:.2%}")
                            
                            # Calculate actual routes for comparison
                            if not selected_vehicle.paths:
                                print("Calculating actual routes for comparison...")
                                calculate_all_routes(G, selected_vehicle, congestion_data)
                            
                            # Compare with actual results
                            if selected_vehicle.travel_times:
                                astar_time = selected_vehicle.travel_times.get('A*', float('inf'))
                                shortest_time = selected_vehicle.travel_times.get('Shortest Path', float('inf'))
                                
                                actual_best = 'astar' if astar_time < shortest_time else 'shortest_path'
                                ai_correct = recommended_algo == actual_best
                                
                                print(f"\n📊 Results Comparison:")
                                print(f"   A* Travel Time: {astar_time:.1f}s")
                                print(f"   Shortest Path Travel Time: {shortest_time:.1f}s")
                                print(f"   Actual Best: {actual_best.upper()}")
                                print(f"   AI Prediction: {'✅ CORRECT' if ai_correct else '❌ INCORRECT'}")
                                
                                if ai_correct:
                                    improvement = abs(astar_time - optimal_time)
                                    print(f"   Time Savings: {improvement:.1f}s by choosing {recommended_algo.upper()}")
                        else:
                            print("❌ No trained algorithm selector models found. Run ML9 first.")
                    else:
                        print("❌ No route optimization models directory found. Train models first (ML9).")
                else:
                    print("Invalid vehicle selection.")
                    
            except ImportError:
                print("❌ Route optimization modules not available.")
            except Exception as e:
                print(f"❌ Error during AI route recommendation: {e}")
        
        # Handle Advanced ML options
        elif choice.startswith('AML') and ADVANCED_ML_AVAILABLE and ml_manager:
            try:
                handle_advanced_ml_option(choice, G, vehicles, congestion_data, current_scenario, ml_manager)
            except Exception as e:
                print(f"❌ Advanced ML option failed: {e}")
        
        elif choice.startswith('AML'):
            print("❌ Advanced ML not available. Please install dependencies:")
            print("   pip install torch torch-geometric")
                
        # Handle scenario selection
        else:
            try:
                scenario_idx = int(choice) - 1
                if 0 <= scenario_idx < len(scenarios):
                    current_scenario = scenarios[scenario_idx]
                    
                    # Handle special case selection
                    special_case_type = None
                    if current_scenario == "Special":
                        print("\nSelect Special Case type:")
                        for i, case_type in enumerate(special_case_types):
                            print(f"{i+1}. {case_type}")
                        
                        try:
                            case_idx = int(input("Select type (number): ")) - 1
                            if 0 <= case_idx < len(special_case_types):
                                special_case_type = special_case_types[case_idx]
                            else:
                                print("Invalid choice. Using Random special case.")
                                special_case_type = "Random"
                        except ValueError:
                            print("Invalid input. Using Random special case.")
                            special_case_type = "Random"
                    
                    # Apply scenario and generate new Excel file
                    congestion_data, excel_file = apply_consistent_congestion_scenario(G, congestion_data, 
                                                     current_scenario, original_congestion, special_case_type)
                    excel_files[current_scenario] = excel_file
                    
                    # Ask whether to recalculate routes for selected vehicles
                    selected_count = sum(1 for v in vehicles if v.selected_for_analysis)
                    if selected_count > 0:
                        recalc = input(f"Recalculate routes for {selected_count} selected vehicles? (y/n): ").strip().lower() == 'y'
                        if recalc:
                            recalculate_routes_for_selected_vehicles(G, vehicles, congestion_data, current_scenario)
                else:
                    print("Invalid choice. Please try again.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a valid option.")
                continue
        
        # Visualize the updated congestion
        try:
            clear_output(wait=True)  # Clear previous output (works in Jupyter notebooks)
        except:
            pass  # If not in Jupyter, just continue
            
        enhanced_visualize_congestion_map(G, congestion_data, vehicles, current_scenario, notable_locations)
        
        # Print updated congestion statistics
        stats = calculate_congestion_stats(congestion_data)
        print("\nCurrent congestion statistics:")
        for key, value in stats.items():
            print(f"  {key.capitalize()}: {value:.4f}")
        
        # Display vehicle count
        selected_count = sum(1 for v in vehicles if v.selected_for_analysis)
        print(f"\nCurrent vehicle count: {len(vehicles)}")
        if selected_count > 0:
            print(f"Vehicles selected for analysis: {selected_count}")


if __name__ == "__main__":
    print("London Traffic Simulation - Interactive Mode")
    print("=" * 50)
    
    # Check dependencies
    try:
        import osmnx
        import networkx
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        print("✓ All required dependencies are available")
    except ImportError as e:
        print(f"✗ Missing required dependency: {e}")
        print("\nPlease install required packages:")
        print("pip install osmnx networkx matplotlib pandas numpy openpyxl scikit-learn")
        sys.exit(1)
    
    try:
        run_interactive_congestion_simulation()
    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your installation and try again.")
        sys.exit(1)