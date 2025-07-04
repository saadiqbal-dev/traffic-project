#!/usr/bin/env python3
"""
Test script for the A* ML System
Quick validation that everything works correctly
"""

import os
import sys

def test_astar_ml_system():
    """Test the complete A* ML system."""
    print("🧪 Testing A* ML System")
    print("=" * 40)
    
    try:
        # Test imports
        print("1. Testing imports...")
        from astar_ml_system import AStarDataCollector, AStarPredictor
        from models import load_london_network, create_evenly_distributed_notable_locations
        print("   ✅ All imports successful")
        
        # Test network loading
        print("\n2. Testing network loading...")
        G = load_london_network()
        notable_locations = create_evenly_distributed_notable_locations(G)
        print(f"   ✅ Network: {len(G.nodes())} nodes, {len(G.edges())} edges")
        print(f"   ✅ Notable locations: {len(notable_locations)}")
        
        # Test data collector
        print("\n3. Testing data collector...")
        collector = AStarDataCollector()
        print("   ✅ AStarDataCollector created")
        
        # Test predictor (without training)
        print("\n4. Testing predictor...")
        predictor = AStarPredictor()
        print("   ✅ AStarPredictor created")
        
        # Test feature extraction
        print("\n5. Testing feature extraction...")
        location_names = list(notable_locations.keys())
        if len(location_names) >= 2:
            source_node = notable_locations[location_names[0]]
            dest_node = notable_locations[location_names[1]]
            
            features = collector._extract_features(
                G, source_node, dest_node, "Normal", 50, 12, "weekday", 2.0
            )
            print(f"   ✅ Features extracted: {len(features)} features")
            print(f"   ✅ Sample features: {list(features.keys())[:5]}...")
        
        print("\n🎉 All A* ML System tests passed!")
        print("\nThe system is ready to use. You can:")
        print("1. Run 'python main.py' and use ASML options")
        print("2. Run 'python astar_ml_system.py' for standalone mode")
        print("3. Use ASML3 from main menu to launch the GUI")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMissing dependencies. Install with:")
        print("pip install scikit-learn pandas numpy matplotlib tkinter")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_astar_ml_system()
    sys.exit(0 if success else 1)