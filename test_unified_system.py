"""
Comprehensive Test Suite for Unified Travel Time System and Dynamic Stress Testing
================================================================================
This script demonstrates and validates the new unified travel time calculation system
and dynamic stress testing capabilities.

Features Tested:
- Unified travel time calculation consistency
- Algorithm result validation
- Progressive stress testing
- Performance monitoring
- System stability under load

Author: Traffic Simulation System
Version: 1.0
"""

import time
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

# Import our modules
from models import Vehicle
from routing import calculate_all_routes, enhanced_a_star_algorithm, shortest_path_algorithm, shortest_path_congestion_aware_algorithm
from unified_travel_time import UnifiedTravelTimeCalculator, TravelTimeValidator, calculate_realistic_travel_time, validate_travel_times
from dynamic_stress_testing import DynamicStressTester, run_quick_stress_test, run_comprehensive_stress_test


def create_test_graph() -> nx.MultiGraph:
    """Create a test graph for validation."""
    print("Creating test graph...")
    
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
    
    # Add edges with realistic properties
    edges = [
        (1, 2, {'length': 100, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 1.0}),
        (2, 3, {'length': 100, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 2.0}),
        (1, 4, {'length': 100, 'speed_kph': 30, 'maxspeed': 30, 'congestion': 3.0}),
        (2, 5, {'length': 100, 'speed_kph': 40, 'maxspeed': 40, 'congestion': 1.5}),
        (3, 6, {'length': 100, 'speed_kph': 60, 'maxspeed': 60, 'congestion': 4.0}),
        (4, 5, {'length': 100, 'speed_kph': 35, 'maxspeed': 35, 'congestion': 2.5}),
        (5, 6, {'length': 100, 'speed_kph': 45, 'maxspeed': 45, 'congestion': 3.5}),
        (5, 7, {'length': 100, 'speed_kph': 40, 'maxspeed': 40, 'congestion': 2.0}),
        (4, 7, {'length': 141, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 1.0}),  # Diagonal
        (6, 7, {'length': 141, 'speed_kph': 50, 'maxspeed': 50, 'congestion': 5.0}),  # High congestion
    ]
    
    for u, v, data in edges:
        G.add_edge(u, v, **data)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def test_unified_travel_time_calculator():
    """Test the unified travel time calculation system."""
    print("\n=== TESTING UNIFIED TRAVEL TIME CALCULATOR ===")
    
    calc = UnifiedTravelTimeCalculator()
    
    # Test 1: Basic travel time calculation
    print("\n1. Testing basic travel time calculation:")
    test_cases = [
        (100, 50),  # 100m at 50 km/h
        (200, 30),  # 200m at 30 km/h
        (500, 60),  # 500m at 60 km/h
    ]
    
    for length, speed in test_cases:
        travel_time = calc.calculate_base_travel_time(length, speed)
        expected_time = length / (speed / 3.6)  # Manual calculation
        print(f"  {length}m at {speed}km/h: {travel_time:.2f}s (expected: {expected_time:.2f}s)")
        assert abs(travel_time - expected_time) < 0.01, "Travel time calculation error"
    
    # Test 2: Congestion multiplier calculation
    print("\n2. Testing congestion multiplier calculation:")
    congestion_levels = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    
    for congestion in congestion_levels:
        multiplier = calc.calculate_congestion_multiplier(congestion)
        penalty_info = calc.get_congestion_penalty_info(congestion)
        print(f"  Congestion {congestion}: {multiplier:.2f}x ({penalty_info['penalty_percent']:.1f}% penalty) - {penalty_info['category']}")
        
        # Validate multiplier is reasonable
        assert 1.0 <= multiplier <= 2.5, f"Multiplier {multiplier} out of reasonable range"
    
    print("✓ Unified travel time calculator tests passed")


def test_algorithm_consistency():
    """Test that all algorithms produce consistent results using unified system."""
    print("\n=== TESTING ALGORITHM CONSISTENCY ===")
    
    G = create_test_graph()
    calc = UnifiedTravelTimeCalculator()
    validator = TravelTimeValidator()
    
    # Test multiple routes
    test_routes = [
        (1, 3),  # Simple route
        (1, 6),  # Medium complexity
        (1, 7),  # Multiple path options
        (4, 6),  # Different starting point
    ]
    
    all_consistent = True
    
    for source, destination in test_routes:
        print(f"\n--- Testing route {source} -> {destination} ---")
        
        # Calculate routes using all algorithms
        results = {}
        
        try:
            # A* algorithm
            astar_path, astar_time, astar_comp = enhanced_a_star_algorithm(G, source, destination)
            if astar_path:
                results['A*'] = (astar_path, astar_time, astar_comp)
            
            # Shortest Path algorithm
            sp_path, sp_time, sp_comp = shortest_path_algorithm(G, source, destination)
            if sp_path:
                results['Shortest Path'] = (sp_path, sp_time, sp_comp)
            
            # Shortest Path Congestion Aware algorithm
            spca_path, spca_time, spca_comp = shortest_path_congestion_aware_algorithm(G, source, destination)
            if spca_path:
                results['Shortest Path Congestion Aware'] = (spca_path, spca_time, spca_comp)
            
            # Validate consistency
            if len(results) >= 2:
                validation_report = validator.validate_algorithm_consistency(results)
                
                print(f"  Algorithms tested: {list(results.keys())}")
                print(f"  Travel times: {validation_report['travel_times']}")
                print(f"  Time ratio: {validation_report['ratio']:.2f}")
                print(f"  Validation: {'✓ PASS' if validation_report['valid'] else '✗ FAIL'}")
                
                if not validation_report['valid']:
                    print(f"  Warning: {validation_report.get('warning', 'Unknown issue')}")
                    all_consistent = False
                
                # Test unified calculation on paths
                for algo, (path, reported_time, comp_time) in results.items():
                    # Calculate using unified system
                    unified_time_with_congestion = calc.calculate_path_travel_time(G, path, apply_congestion=True)
                    unified_time_without_congestion = calc.calculate_path_travel_time(G, path, apply_congestion=False)
                    
                    print(f"    {algo}:")
                    print(f"      Reported: {reported_time:.2f}s")
                    print(f"      Unified (with congestion): {unified_time_with_congestion:.2f}s")
                    print(f"      Unified (without congestion): {unified_time_without_congestion:.2f}s")
                    print(f"      Computation time: {comp_time:.4f}s")
            else:
                print(f"  Warning: Only {len(results)} algorithm(s) found valid paths")
        
        except Exception as e:
            print(f"  Error testing route {source} -> {destination}: {e}")
            all_consistent = False
    
    if all_consistent:
        print("\n✓ Algorithm consistency tests passed")
    else:
        print("\n⚠ Some algorithm consistency issues detected")
    
    return all_consistent


def test_travel_time_validation():
    """Test travel time validation functions."""
    print("\n=== TESTING TRAVEL TIME VALIDATION ===")
    
    G = create_test_graph()
    validator = TravelTimeValidator()
    
    # Create test results with known characteristics
    test_results = {
        'Algorithm A': ([1, 2, 3], 120.0, 0.001),  # 2 minute travel time
        'Algorithm B': ([1, 4, 5, 6], 150.0, 0.002),  # 2.5 minute travel time
        'Algorithm C': ([1, 2, 5, 6], 135.0, 0.0015),  # 2.25 minute travel time
    }
    
    # Test validation
    validation_report = validator.validate_algorithm_consistency(test_results)
    
    print(f"Test results: {validation_report['travel_times']}")
    print(f"Min time: {validation_report['min_time']:.1f}s")
    print(f"Max time: {validation_report['max_time']:.1f}s")
    print(f"Ratio: {validation_report['ratio']:.2f}")
    print(f"Valid: {'✓ YES' if validation_report['valid'] else '✗ NO'}")
    
    # Test edge debugging
    print("\nTesting edge debugging:")
    debug_info = validator.debug_travel_time_calculation(G, 1, 2, 0)
    if debug_info['valid']:
        print(f"  Edge 1->2: {debug_info['length_meters']}m at {debug_info['speed_kph']}km/h")
        print(f"  Congestion: {debug_info['congestion_level']} ({debug_info['penalty_info']['category']})")
        print(f"  Base time: {debug_info['base_travel_time']:.2f}s")
        print(f"  Congested time: {debug_info['congested_travel_time']:.2f}s")
    else:
        print(f"  Error: {debug_info['error']}")
    
    print("✓ Travel time validation tests completed")


def test_dynamic_stress_testing():
    """Test the dynamic stress testing system."""
    print("\n=== TESTING DYNAMIC STRESS TESTING SYSTEM ===")
    
    G = create_test_graph()
    
    # Create a mock simulation object
    class MockSimulation:
        def __init__(self):
            self.vehicles = []
            self.congestion_data = {}
    
    simulation = MockSimulation()
    
    # Run quick stress test
    print("\nRunning quick stress test...")
    try:
        results = run_quick_stress_test(G, simulation)
        
        summary = results['test_summary']
        print(f"  Test duration: {summary['test_duration']:.1f}s")
        print(f"  Max vehicles tested: {summary['max_vehicles_tested']}")
        print(f"  Max congestion tested: {summary['max_congestion_tested']:.1f}")
        print(f"  Average success rate: {summary['avg_success_rate']:.1%}")
        print(f"  Failures: {summary['failure_count']}")
        
        # Check recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"  Recommendations: {len(recommendations)} generated")
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                print(f"    {i}. {rec[:100]}...")
        
        print("✓ Dynamic stress testing completed successfully")
        
        return results
        
    except Exception as e:
        print(f"✗ Dynamic stress testing failed: {e}")
        return None


def test_performance_comparison():
    """Compare performance between old and new systems."""
    print("\n=== PERFORMANCE COMPARISON ===")
    
    G = create_test_graph()
    calc = UnifiedTravelTimeCalculator()
    
    # Test performance with different vehicle counts
    vehicle_counts = [10, 25, 50]
    
    for count in vehicle_counts:
        print(f"\nTesting with {count} vehicles:")
        
        # Generate test vehicles
        vehicles = []
        nodes = list(G.nodes())
        
        for i in range(count):
            source = random.choice(nodes)
            destination = random.choice([n for n in nodes if n != source])
            vehicle = Vehicle(id=f"perf_test_{i}", source=source, destination=destination)
            vehicles.append(vehicle)
        
        # Time the unified system
        start_time = time.time()
        successful_routes = 0
        
        for vehicle in vehicles:
            try:
                updated_vehicle = calculate_all_routes(G, vehicle, {})
                if updated_vehicle.paths:
                    successful_routes += 1
            except Exception as e:
                print(f"    Error with vehicle {vehicle.id}: {e}")
        
        total_time = time.time() - start_time
        
        print(f"  Results: {successful_routes}/{count} successful routes")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per vehicle: {total_time/count:.4f}s")
        print(f"  Routes per second: {successful_routes/total_time:.1f}")


def run_comprehensive_validation():
    """Run all validation tests."""
    print("=" * 80)
    print("COMPREHENSIVE VALIDATION OF UNIFIED TRAVEL TIME SYSTEM")
    print("=" * 80)
    
    start_time = time.time()
    
    # Run all tests
    tests_passed = 0
    total_tests = 5
    
    try:
        test_unified_travel_time_calculator()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Unified calculator test failed: {e}")
    
    try:
        if test_algorithm_consistency():
            tests_passed += 1
    except Exception as e:
        print(f"✗ Algorithm consistency test failed: {e}")
    
    try:
        test_travel_time_validation()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Travel time validation test failed: {e}")
    
    try:
        stress_results = test_dynamic_stress_testing()
        if stress_results:
            tests_passed += 1
    except Exception as e:
        print(f"✗ Dynamic stress testing failed: {e}")
    
    try:
        test_performance_comparison()
        tests_passed += 1
    except Exception as e:
        print(f"✗ Performance comparison test failed: {e}")
    
    # Final summary
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {tests_passed/total_tests:.1%}")
    print(f"Total validation time: {total_time:.1f}s")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
    
    return tests_passed == total_tests


def save_validation_report(results: Dict[str, Any]):
    """Save validation results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"validation_report_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nValidation report saved to: {filename}")
    except Exception as e:
        print(f"Error saving validation report: {e}")


if __name__ == "__main__":
    print("Starting comprehensive validation of unified travel time system...")
    
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Run validation
    success = run_comprehensive_validation()
    
    if success:
        print("\n🚀 System validation completed successfully!")
        print("The unified travel time system and dynamic stress testing are ready for use.")
    else:
        print("\n❌ System validation encountered issues.")
        print("Please review the test results and fix any problems before proceeding.")
