"""
Unified Travel Time Calculation System
=====================================
This module provides a single, consistent travel time calculation system
used by all routing algorithms to ensure realistic and symmetrical results.

Key Features:
- Physics-based travel time calculation (Time = Distance / Speed)
- Research-based congestion penalties (max 2.5x multiplier)
- Consistent results across all algorithms
- Support for dynamic congestion updates

Author: Traffic Simulation System
Version: 1.0
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import time


class UnifiedTravelTimeCalculator:
    """
    Single source of truth for travel time calculations across all algorithms.
    Ensures symmetrical and realistic results throughout the system.
    """
    
    @staticmethod
    def calculate_base_travel_time(length_meters: float, speed_kph: float) -> float:
        """
        Calculate base travel time using proper physics formula.
        
        Formula: Time = Distance / Speed
        Units: seconds = meters / (km/h converted to m/s)
        
        Args:
            length_meters: Edge length in meters
            speed_kph: Speed in kilometers per hour
            
        Returns:
            Travel time in seconds
            
        Raises:
            ValueError: If inputs are invalid
        """
        if length_meters < 0:
            raise ValueError(f"Length cannot be negative: {length_meters}")
        if speed_kph <= 0:
            return float('inf')  # Infinite time for zero/negative speed
        
        # Convert km/h to m/s: speed_kph / 3.6
        speed_ms = speed_kph / 3.6
        
        # Time = Distance / Speed
        travel_time_seconds = length_meters / speed_ms
        
        return travel_time_seconds
    
    @staticmethod
    def calculate_congestion_multiplier(congestion_level: float) -> float:
        """
        Calculate realistic congestion penalty multiplier.
        
        Research-based penalties that don't create extreme differences:
        - Congestion 1-2: 1.0-1.1x (0-10% increase)
        - Congestion 3-4: 1.1-1.3x (10-30% increase)  
        - Congestion 5-6: 1.3-1.6x (30-60% increase)
        - Congestion 7-8: 1.6-2.0x (60-100% increase)
        - Congestion 9-10: 2.0-2.5x (100-150% increase)
        
        Args:
            congestion_level: Congestion level (1-10 scale)
            
        Returns:
            Penalty multiplier (1.0 = no penalty, 2.5 = 150% increase)
        """
        if congestion_level <= 1.0:
            return 1.0
        
        # Cap congestion at reasonable maximum
        congestion_level = min(congestion_level, 10.0)
        
        # Piecewise linear function for realistic penalties
        if congestion_level <= 2.0:
            # Light traffic: 0-10% penalty
            return 1.0 + (congestion_level - 1.0) * 0.1
        elif congestion_level <= 4.0:
            # Moderate traffic: 10-30% penalty
            return 1.1 + (congestion_level - 2.0) * 0.1
        elif congestion_level <= 6.0:
            # Heavy traffic: 30-60% penalty
            return 1.3 + (congestion_level - 4.0) * 0.15
        elif congestion_level <= 8.0:
            # Severe traffic: 60-100% penalty
            return 1.6 + (congestion_level - 6.0) * 0.2
        else:
            # Gridlock: 100-150% penalty (capped at 2.5x)
            return min(2.0 + (congestion_level - 8.0) * 0.25, 2.5)
    
    @staticmethod
    def calculate_edge_travel_time(G: nx.Graph, u: int, v: int, k: int, 
                                 apply_congestion: bool = True) -> float:
        """
        Calculate travel time for a specific edge.
        Used by ALL algorithms to ensure consistency.
        
        Args:
            G: NetworkX graph
            u: Source node
            v: Target node  
            k: Edge key
            apply_congestion: Whether to apply congestion penalties
            
        Returns:
            Travel time in seconds
            
        Raises:
            KeyError: If edge doesn't exist
            ValueError: If edge data is invalid
        """
        try:
            edge_data = G[u][v][k]
        except KeyError:
            raise KeyError(f"Edge {u}->{v} (key {k}) not found in graph")
        
        # Get base parameters with defaults
        length = edge_data.get('length', 100)  # meters
        speed_kph = edge_data.get('speed_kph', 30)  # km/h
        
        # Validate parameters
        if length <= 0:
            length = 100  # Default length
        if speed_kph <= 0:
            speed_kph = 30  # Default speed
        
        # Calculate base travel time
        base_time = UnifiedTravelTimeCalculator.calculate_base_travel_time(length, speed_kph)
        
        if not apply_congestion:
            return base_time
        
        # Apply congestion penalty
        congestion = edge_data.get('congestion', 1.0)
        multiplier = UnifiedTravelTimeCalculator.calculate_congestion_multiplier(congestion)
        
        return base_time * multiplier
    
    @staticmethod
    def calculate_path_travel_time(G: nx.Graph, path: List[int], 
                                 apply_congestion: bool = True) -> float:
        """
        Calculate total travel time for a complete path.
        
        Args:
            G: NetworkX graph
            path: List of node IDs representing the path
            apply_congestion: Whether to apply congestion penalties
            
        Returns:
            Total travel time in seconds
            
        Raises:
            ValueError: If path is invalid
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_time = 0.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            
            # Handle multigraph - get first available edge
            if u in G and v in G[u]:
                edge_keys = list(G[u][v].keys())
                if edge_keys:
                    k = edge_keys[0]  # Use first edge key
                    try:
                        edge_time = UnifiedTravelTimeCalculator.calculate_edge_travel_time(
                            G, u, v, k, apply_congestion
                        )
                        total_time += edge_time
                    except (KeyError, ValueError) as e:
                        print(f"Warning: Error calculating time for edge {u}->{v}: {e}")
                        # Use default time for problematic edges
                        total_time += 60.0  # 1 minute default
                else:
                    print(f"Warning: No edge keys found for {u}->{v}")
                    total_time += 60.0
            else:
                print(f"Warning: Edge {u}->{v} not found in graph")
                total_time += 60.0
        
        return total_time
    
    @staticmethod
    def validate_travel_time_result(travel_time: float, path_length: int) -> bool:
        """
        Validate that calculated travel time is reasonable.
        
        Args:
            travel_time: Calculated travel time in seconds
            path_length: Number of nodes in path
            
        Returns:
            True if travel time seems reasonable, False otherwise
        """
        if travel_time <= 0:
            return False
        
        # Reasonable bounds: 10 seconds to 2 hours per edge on average
        min_time_per_edge = 10  # seconds
        max_time_per_edge = 7200  # 2 hours
        
        edges_count = max(1, path_length - 1)
        avg_time_per_edge = travel_time / edges_count
        
        return min_time_per_edge <= avg_time_per_edge <= max_time_per_edge
    
    @staticmethod
    def get_congestion_penalty_info(congestion_level: float) -> Dict[str, Any]:
        """
        Get detailed information about congestion penalty.
        
        Args:
            congestion_level: Congestion level (1-10 scale)
            
        Returns:
            Dictionary with penalty information
        """
        multiplier = UnifiedTravelTimeCalculator.calculate_congestion_multiplier(congestion_level)
        penalty_percent = (multiplier - 1.0) * 100
        
        # Determine congestion category
        if congestion_level <= 2.0:
            category = "Light Traffic"
            description = "Free flowing traffic with minimal delays"
        elif congestion_level <= 4.0:
            category = "Moderate Traffic"
            description = "Some congestion, noticeable but manageable delays"
        elif congestion_level <= 6.0:
            category = "Heavy Traffic"
            description = "Significant congestion, substantial delays"
        elif congestion_level <= 8.0:
            category = "Severe Congestion"
            description = "Very heavy traffic, major delays expected"
        else:
            category = "Gridlock"
            description = "Stop-and-go traffic, extreme delays"
        
        return {
            'congestion_level': congestion_level,
            'multiplier': multiplier,
            'penalty_percent': penalty_percent,
            'category': category,
            'description': description
        }


class TravelTimeValidator:
    """
    Validates travel time calculations and provides debugging information.
    """
    
    @staticmethod
    def validate_algorithm_consistency(results: Dict[str, Tuple[List[int], float, float]]) -> Dict[str, Any]:
        """
        Validate that algorithm results are consistent and realistic.
        
        Args:
            results: Dictionary mapping algorithm names to (path, travel_time, comp_time) tuples
            
        Returns:
            Validation report
        """
        if not results:
            return {'valid': False, 'error': 'No results to validate'}
        
        travel_times = {algo: time for algo, (path, time, comp) in results.items()}
        
        # Check for reasonable travel times
        min_time = min(travel_times.values())
        max_time = max(travel_times.values())
        
        # Maximum acceptable ratio between algorithms
        max_acceptable_ratio = 3.0
        
        if min_time <= 0:
            return {'valid': False, 'error': 'Invalid travel time (≤ 0)'}
        
        ratio = max_time / min_time
        
        validation_report = {
            'valid': ratio <= max_acceptable_ratio,
            'travel_times': travel_times,
            'min_time': min_time,
            'max_time': max_time,
            'ratio': ratio,
            'max_acceptable_ratio': max_acceptable_ratio,
            'algorithms_count': len(results)
        }
        
        if not validation_report['valid']:
            validation_report['warning'] = f"Travel time ratio ({ratio:.2f}) exceeds acceptable limit ({max_acceptable_ratio})"
        
        return validation_report
    
    @staticmethod
    def debug_travel_time_calculation(G: nx.Graph, u: int, v: int, k: int) -> Dict[str, Any]:
        """
        Debug travel time calculation for a specific edge.
        
        Args:
            G: NetworkX graph
            u: Source node
            v: Target node
            k: Edge key
            
        Returns:
            Debug information
        """
        calc = UnifiedTravelTimeCalculator()
        
        try:
            edge_data = G[u][v][k]
            
            # Get parameters
            length = edge_data.get('length', 100)
            speed_kph = edge_data.get('speed_kph', 30)
            congestion = edge_data.get('congestion', 1.0)
            
            # Calculate times
            base_time = calc.calculate_base_travel_time(length, speed_kph)
            congested_time = calc.calculate_edge_travel_time(G, u, v, k, True)
            
            # Get penalty info
            penalty_info = calc.get_congestion_penalty_info(congestion)
            
            return {
                'edge_id': f"{u}_{v}_{k}",
                'length_meters': length,
                'speed_kph': speed_kph,
                'congestion_level': congestion,
                'base_travel_time': base_time,
                'congested_travel_time': congested_time,
                'penalty_info': penalty_info,
                'valid': True
            }
            
        except Exception as e:
            return {
                'edge_id': f"{u}_{v}_{k}",
                'error': str(e),
                'valid': False
            }


# Utility functions for backward compatibility and easy integration

def get_unified_calculator() -> UnifiedTravelTimeCalculator:
    """Get instance of unified travel time calculator."""
    return UnifiedTravelTimeCalculator()

def calculate_realistic_travel_time(G: nx.Graph, path: List[int], 
                                  apply_congestion: bool = True) -> float:
    """
    Convenience function to calculate realistic travel time for a path.
    
    Args:
        G: NetworkX graph
        path: List of node IDs
        apply_congestion: Whether to apply congestion penalties
        
    Returns:
        Total travel time in seconds
    """
    calc = UnifiedTravelTimeCalculator()
    return calc.calculate_path_travel_time(G, path, apply_congestion)

def validate_travel_times(results: Dict[str, Tuple[List[int], float, float]]) -> bool:
    """
    Quick validation of algorithm travel time results.
    
    Args:
        results: Algorithm results dictionary
        
    Returns:
        True if results are valid, False otherwise
    """
    validator = TravelTimeValidator()
    report = validator.validate_algorithm_consistency(results)
    return report['valid']


# Module-level constants
DEFAULT_SPEED_KPH = 30
DEFAULT_LENGTH_METERS = 100
MAX_CONGESTION_MULTIPLIER = 2.5
MAX_ACCEPTABLE_ALGORITHM_RATIO = 3.0

# Version info
__version__ = "1.0.0"
__author__ = "Traffic Simulation System"
