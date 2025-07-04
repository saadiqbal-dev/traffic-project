"""
Dynamic Stress Testing System
============================
This module provides comprehensive stress testing capabilities for the traffic simulation system.
It progressively increases system load and congestion to identify breaking points and performance limits.

Key Features:
- Progressive load testing (vehicles, congestion, complexity)
- Real-time performance monitoring
- Automatic failure detection and recovery
- Detailed performance analytics
- System stability validation

Author: Traffic Simulation System
Version: 1.0
"""

import time
import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any, Callable
import threading
import queue
import json
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from models import Vehicle
from routing import calculate_all_routes
from unified_travel_time import UnifiedTravelTimeCalculator, TravelTimeValidator


class StressTestMetrics:
    """Tracks and analyzes stress test performance metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics for a new test."""
        self.start_time = time.time()
        self.vehicle_counts = []
        self.congestion_levels = []
        self.computation_times = []
        self.memory_usage = []
        self.success_rates = []
        self.travel_time_ratios = []
        self.algorithm_performance = {}
        self.failure_points = []
        self.timestamps = []
        
    def record_iteration(self, vehicle_count: int, avg_congestion: float, 
                        computation_time: float, success_rate: float,
                        travel_time_ratio: float, algorithm_times: Dict[str, float]):
        """Record metrics for a single test iteration."""
        current_time = time.time() - self.start_time
        
        self.timestamps.append(current_time)
        self.vehicle_counts.append(vehicle_count)
        self.congestion_levels.append(avg_congestion)
        self.computation_times.append(computation_time)
        self.success_rates.append(success_rate)
        self.travel_time_ratios.append(travel_time_ratio)
        
        # Record algorithm-specific performance
        for algo, comp_time in algorithm_times.items():
            if algo not in self.algorithm_performance:
                self.algorithm_performance[algo] = []
            self.algorithm_performance[algo].append(comp_time)
    
    def record_failure(self, vehicle_count: int, congestion_level: float, 
                      error_type: str, error_message: str):
        """Record a system failure point."""
        self.failure_points.append({
            'timestamp': time.time() - self.start_time,
            'vehicle_count': vehicle_count,
            'congestion_level': congestion_level,
            'error_type': error_type,
            'error_message': error_message
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive test summary."""
        if not self.timestamps:
            return {'error': 'No data recorded'}
        
        return {
            'test_duration': self.timestamps[-1],
            'max_vehicles_tested': max(self.vehicle_counts) if self.vehicle_counts else 0,
            'max_congestion_tested': max(self.congestion_levels) if self.congestion_levels else 0,
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
            'max_computation_time': max(self.computation_times) if self.computation_times else 0,
            'min_success_rate': min(self.success_rates) if self.success_rates else 0,
            'avg_success_rate': np.mean(self.success_rates) if self.success_rates else 0,
            'max_travel_time_ratio': max(self.travel_time_ratios) if self.travel_time_ratios else 0,
            'failure_count': len(self.failure_points),
            'algorithm_performance': {
                algo: {
                    'avg_time': np.mean(times),
                    'max_time': max(times),
                    'min_time': min(times)
                } for algo, times in self.algorithm_performance.items()
            },
            'failure_points': self.failure_points
        }


class DynamicStressTester:
    """
    Main stress testing engine that progressively increases system load
    and monitors performance under extreme conditions.
    """
    
    def __init__(self, G: nx.Graph, simulation: Any = None):
        self.G = G
        self.simulation = simulation
        self.metrics = StressTestMetrics()
        self.calc = UnifiedTravelTimeCalculator()
        self.validator = TravelTimeValidator()
        
        # Test configuration
        self.max_vehicles = 1000
        self.max_congestion = 10.0
        self.timeout_seconds = 300  # 5 minutes max per test
        self.failure_threshold = 0.5  # Stop if success rate drops below 50%
        
        # Performance thresholds
        self.max_computation_time = 30.0  # seconds
        self.max_travel_time_ratio = 5.0  # max ratio between algorithms
        
    def generate_test_vehicles(self, count: int) -> List[Vehicle]:
        """Generate vehicles for stress testing."""
        vehicles = []
        nodes = list(self.G.nodes())
        
        if len(nodes) < 2:
            raise ValueError("Graph must have at least 2 nodes for testing")
        
        for i in range(count):
            # Select random source and destination
            source = random.choice(nodes)
            destination = random.choice([n for n in nodes if n != source])
            
            vehicle = Vehicle(
                id=f"stress_test_{i}",
                source=source,
                destination=destination
            )
            vehicles.append(vehicle)
        
        return vehicles
    
    def apply_stress_congestion(self, base_congestion: float, variation: float = 0.3):
        """Apply stress-level congestion to the graph."""
        for u, v, k, data in self.G.edges(keys=True, data=True):
            # Add random variation to congestion
            congestion_variation = random.uniform(-variation, variation)
            stress_congestion = max(1.0, base_congestion + congestion_variation)
            
            # Cap at maximum congestion level
            stress_congestion = min(stress_congestion, self.max_congestion)
            
            # Apply to edge
            self.G[u][v][k]['congestion'] = stress_congestion
    
    def run_single_stress_test(self, vehicle_count: int, congestion_level: float) -> Dict[str, Any]:
        """Run a single stress test iteration."""
        print(f"\n--- Stress Test: {vehicle_count} vehicles, congestion {congestion_level:.1f} ---")
        
        test_start = time.time()
        results = {
            'vehicle_count': vehicle_count,
            'congestion_level': congestion_level,
            'success': False,
            'computation_time': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'algorithm_times': {},
            'travel_time_validation': None,
            'error': None
        }
        
        try:
            # Generate test vehicles
            vehicles = self.generate_test_vehicles(vehicle_count)
            
            # Apply stress congestion
            self.apply_stress_congestion(congestion_level)
            
            # Track algorithm performance
            algorithm_times = {'A*': [], 'Shortest Path': [], 'Shortest Path Congestion Aware': []}
            successful_routes = 0
            failed_routes = 0
            all_results = {}
            
            # Test each vehicle
            for vehicle in vehicles:
                try:
                    # Calculate routes with timeout protection
                    vehicle_start = time.time()
                    updated_vehicle = calculate_all_routes(self.G, vehicle, {})
                    vehicle_time = time.time() - vehicle_start
                    
                    # Check for timeout
                    if vehicle_time > self.max_computation_time:
                        print(f"  Vehicle {vehicle.id}: TIMEOUT ({vehicle_time:.2f}s)")
                        failed_routes += 1
                        continue
                    
                    # Validate results
                    if updated_vehicle.paths:
                        successful_routes += 1
                        
                        # Record algorithm times
                        for algo in algorithm_times.keys():
                            if algo in updated_vehicle.computation_times:
                                algorithm_times[algo].append(updated_vehicle.computation_times[algo])
                        
                        # Store results for validation
                        vehicle_results = {}
                        for algo in ['A*', 'Shortest Path', 'Shortest Path Congestion Aware']:
                            if algo in updated_vehicle.paths and algo in updated_vehicle.travel_times:
                                vehicle_results[algo] = (
                                    updated_vehicle.paths[algo],
                                    updated_vehicle.travel_times[algo],
                                    updated_vehicle.computation_times.get(algo, 0)
                                )
                        
                        if vehicle_results:
                            all_results[vehicle.id] = vehicle_results
                    else:
                        failed_routes += 1
                        print(f"  Vehicle {vehicle.id}: No valid paths found")
                
                except Exception as e:
                    failed_routes += 1
                    print(f"  Vehicle {vehicle.id}: ERROR - {str(e)}")
                
                # Check for early termination
                total_time = time.time() - test_start
                if total_time > self.timeout_seconds:
                    print(f"  Test TIMEOUT after {total_time:.1f}s")
                    break
            
            # Calculate results
            total_computation_time = time.time() - test_start
            success_rate = successful_routes / len(vehicles) if vehicles else 0
            
            # Validate travel time consistency
            travel_time_validation = None
            if all_results:
                # Sample a few results for validation
                sample_results = dict(list(all_results.items())[:min(5, len(all_results))])
                validation_reports = []
                
                for vehicle_id, vehicle_results in sample_results.items():
                    validation = self.validator.validate_algorithm_consistency(vehicle_results)
                    validation_reports.append(validation)
                
                # Aggregate validation results
                if validation_reports:
                    travel_time_validation = {
                        'valid_count': sum(1 for r in validation_reports if r.get('valid', False)),
                        'total_count': len(validation_reports),
                        'avg_ratio': np.mean([r.get('ratio', 1.0) for r in validation_reports]),
                        'max_ratio': max([r.get('ratio', 1.0) for r in validation_reports])
                    }
            
            # Update results
            results.update({
                'success': True,
                'computation_time': total_computation_time,
                'successful_routes': successful_routes,
                'failed_routes': failed_routes,
                'success_rate': success_rate,
                'algorithm_times': {
                    algo: {
                        'avg': np.mean(times) if times else 0,
                        'max': max(times) if times else 0,
                        'count': len(times)
                    } for algo, times in algorithm_times.items()
                },
                'travel_time_validation': travel_time_validation
            })
            
            print(f"  Results: {successful_routes}/{len(vehicles)} successful ({success_rate:.1%})")
            print(f"  Total time: {total_computation_time:.2f}s")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"  CRITICAL ERROR: {str(e)}")
        
        return results
    
    def run_progressive_stress_test(self, 
                                  vehicle_steps: List[int] = None,
                                  congestion_steps: List[float] = None) -> Dict[str, Any]:
        """
        Run progressive stress test with increasing load.
        
        Args:
            vehicle_steps: List of vehicle counts to test
            congestion_steps: List of congestion levels to test
            
        Returns:
            Comprehensive test results
        """
        if vehicle_steps is None:
            vehicle_steps = [10, 25, 50, 100, 200, 500, 1000]
        
        if congestion_steps is None:
            congestion_steps = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        
        print("=== DYNAMIC STRESS TESTING STARTED ===")
        print(f"Vehicle steps: {vehicle_steps}")
        print(f"Congestion steps: {congestion_steps}")
        
        self.metrics.reset()
        test_results = []
        
        # Test each combination
        for congestion in congestion_steps:
            for vehicles in vehicle_steps:
                result = self.run_single_stress_test(vehicles, congestion)
                test_results.append(result)
                
                # Record metrics
                if result['success']:
                    travel_time_ratio = 1.0
                    if result['travel_time_validation']:
                        travel_time_ratio = result['travel_time_validation'].get('max_ratio', 1.0)
                    
                    avg_algo_time = np.mean([
                        data['avg'] for data in result['algorithm_times'].values()
                        if data['count'] > 0
                    ]) if result['algorithm_times'] else 0
                    
                    self.metrics.record_iteration(
                        vehicles, congestion, result['computation_time'],
                        result['success_rate'], travel_time_ratio,
                        {algo: data['avg'] for algo, data in result['algorithm_times'].items()}
                    )
                else:
                    self.metrics.record_failure(
                        vehicles, congestion, 'test_failure', 
                        result.get('error', 'Unknown error')
                    )
                
                # Check failure threshold
                if result['success'] and result['success_rate'] < self.failure_threshold:
                    print(f"\n!!! FAILURE THRESHOLD REACHED !!!")
                    print(f"Success rate {result['success_rate']:.1%} < {self.failure_threshold:.1%}")
                    break
            
            # Early termination check
            if test_results and not test_results[-1]['success']:
                print(f"\n!!! CRITICAL FAILURE - STOPPING TESTS !!!")
                break
        
        # Generate final report
        summary = self.metrics.get_summary()
        
        final_results = {
            'test_summary': summary,
            'individual_results': test_results,
            'recommendations': self._generate_recommendations(summary, test_results)
        }
        
        print("\n=== STRESS TESTING COMPLETED ===")
        self._print_summary(summary)
        
        return final_results
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance recommendations based on test results."""
        recommendations = []
        
        # Performance recommendations
        if summary['max_computation_time'] > 10.0:
            recommendations.append(
                f"High computation times detected (max: {summary['max_computation_time']:.1f}s). "
                "Consider optimizing algorithms or implementing caching."
            )
        
        if summary['min_success_rate'] < 0.8:
            recommendations.append(
                f"Low success rates detected (min: {summary['min_success_rate']:.1%}). "
                "System may need better error handling or timeout management."
            )
        
        if summary['max_travel_time_ratio'] > 3.0:
            recommendations.append(
                f"High travel time variance between algorithms (max ratio: {summary['max_travel_time_ratio']:.1f}). "
                "Review unified travel time calculation consistency."
            )
        
        # Algorithm-specific recommendations
        algo_performance = summary.get('algorithm_performance', {})
        if algo_performance:
            slowest_algo = max(algo_performance.keys(), 
                             key=lambda x: algo_performance[x]['avg_time'])
            fastest_algo = min(algo_performance.keys(), 
                             key=lambda x: algo_performance[x]['avg_time'])
            
            if (algo_performance[slowest_algo]['avg_time'] > 
                algo_performance[fastest_algo]['avg_time'] * 2):
                recommendations.append(
                    f"{slowest_algo} is significantly slower than {fastest_algo}. "
                    "Consider algorithm optimization or load balancing."
                )
        
        # Failure analysis
        if summary['failure_count'] > 0:
            recommendations.append(
                f"{summary['failure_count']} failures detected. "
                "Review error logs and implement better error recovery."
            )
        
        return recommendations
    
    def _print_summary(self, summary: Dict[str, Any]):
        """Print formatted test summary."""
        print(f"\nTest Duration: {summary['test_duration']:.1f}s")
        print(f"Max Vehicles Tested: {summary['max_vehicles_tested']}")
        print(f"Max Congestion Tested: {summary['max_congestion_tested']:.1f}")
        print(f"Average Computation Time: {summary['avg_computation_time']:.2f}s")
        print(f"Max Computation Time: {summary['max_computation_time']:.2f}s")
        print(f"Average Success Rate: {summary['avg_success_rate']:.1%}")
        print(f"Min Success Rate: {summary['min_success_rate']:.1%}")
        print(f"Failures: {summary['failure_count']}")
        
        if summary['algorithm_performance']:
            print("\nAlgorithm Performance:")
            for algo, perf in summary['algorithm_performance'].items():
                print(f"  {algo}: avg={perf['avg_time']:.4f}s, max={perf['max_time']:.4f}s")
    
    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save test results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stress_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_performance_plots(self, results: Dict[str, Any], save_plots: bool = True):
        """Create visualization plots of test results."""
        try:
            individual_results = results.get('individual_results', [])
            if not individual_results:
                print("No data available for plotting")
                return
            
            # Extract data for plotting
            vehicle_counts = [r['vehicle_count'] for r in individual_results if r['success']]
            congestion_levels = [r['congestion_level'] for r in individual_results if r['success']]
            computation_times = [r['computation_time'] for r in individual_results if r['success']]
            success_rates = [r['success_rate'] for r in individual_results if r['success']]
            
            if not vehicle_counts:
                print("No successful tests to plot")
                return
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Computation Time vs Vehicle Count
            ax1.scatter(vehicle_counts, computation_times, c=congestion_levels, cmap='Reds', alpha=0.7)
            ax1.set_xlabel('Vehicle Count')
            ax1.set_ylabel('Computation Time (s)')
            ax1.set_title('Computation Time vs Vehicle Count')
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Success Rate vs Vehicle Count
            ax2.scatter(vehicle_counts, success_rates, c=congestion_levels, cmap='Reds', alpha=0.7)
            ax2.set_xlabel('Vehicle Count')
            ax2.set_ylabel('Success Rate')
            ax2.set_title('Success Rate vs Vehicle Count')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
            
            # Plot 3: Computation Time vs Congestion Level
            ax3.scatter(congestion_levels, computation_times, c=vehicle_counts, cmap='Blues', alpha=0.7)
            ax3.set_xlabel('Congestion Level')
            ax3.set_ylabel('Computation Time (s)')
            ax3.set_title('Computation Time vs Congestion Level')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Success Rate vs Congestion Level
            scatter = ax4.scatter(congestion_levels, success_rates, c=vehicle_counts, cmap='Blues', alpha=0.7)
            ax4.set_xlabel('Congestion Level')
            ax4.set_ylabel('Success Rate')
            ax4.set_title('Success Rate vs Congestion Level')
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 1.1)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax4, label='Vehicle Count')
            
            plt.tight_layout()
            
            if save_plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"stress_test_plots_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"Plots saved to: {plot_filename}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating plots: {e}")


# Convenience functions for easy testing

def run_quick_stress_test(G: nx.Graph, simulation: Any = None) -> Dict[str, Any]:
    """Run a quick stress test with default parameters."""
    tester = DynamicStressTester(G, simulation)
    return tester.run_progressive_stress_test(
        vehicle_steps=[10, 25, 50, 100],
        congestion_steps=[1.0, 3.0, 5.0]
    )

def run_comprehensive_stress_test(G: nx.Graph, simulation: Any = None) -> Dict[str, Any]:
    """Run a comprehensive stress test with extensive parameters."""
    tester = DynamicStressTester(G, simulation)
    return tester.run_progressive_stress_test(
        vehicle_steps=[10, 25, 50, 100, 200, 500, 1000],
        congestion_steps=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    )

def run_extreme_stress_test(G: nx.Graph, simulation: Any = None) -> Dict[str, Any]:
    """Run an extreme stress test to find absolute system limits."""
    tester = DynamicStressTester(G, simulation)
    tester.max_vehicles = 5000
    tester.timeout_seconds = 600  # 10 minutes
    
    return tester.run_progressive_stress_test(
        vehicle_steps=[100, 250, 500, 1000, 2000, 5000],
        congestion_steps=[1.0, 5.0, 10.0, 15.0, 20.0]
    )


# Module constants
DEFAULT_VEHICLE_STEPS = [10, 25, 50, 100, 200, 500]
DEFAULT_CONGESTION_STEPS = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
MAX_COMPUTATION_TIME_THRESHOLD = 30.0
MIN_SUCCESS_RATE_THRESHOLD = 0.5
MAX_TRAVEL_TIME_RATIO_THRESHOLD = 5.0

# Version info
__version__ = "1.0.0"
__author__ = "Traffic Simulation System"
