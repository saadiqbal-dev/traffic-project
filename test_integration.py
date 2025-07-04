#!/usr/bin/env python3
"""
Quick test of the advanced ML integration
"""

import sys
import os
sys.path.append(os.getcwd())

def test_integration():
    print("🔧 Testing Advanced ML Integration...")
    
    try:
        # Test imports
        print("✅ Testing imports...")
        from advanced_ml_integration import AdvancedMLManager, handle_advanced_ml_option
        print("   ✓ Advanced ML integration module imported")
        
        from gnn_traffic_prediction import TORCH_GEOMETRIC_AVAILABLE
        print(f"   ✓ PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
        
        # Test manager creation
        print("\n✅ Testing ML Manager creation...")
        ml_manager = AdvancedMLManager()
        print("   ✓ AdvancedMLManager created successfully")
        
        # Test initialization 
        print("\n✅ Testing ML system initialization...")
        init_result = ml_manager.initialize_advanced_ml()
        print(f"   ✓ ML system initialized: {init_result}")
        
        # Load basic network for testing
        print("\n✅ Testing basic network operations...")
        from models import load_london_network, generate_initial_congestion
        G = load_london_network()
        congestion_data = generate_initial_congestion(G)
        print(f"   ✓ Network loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
        
        # Test menu integration
        print("\n✅ Testing menu integration...")
        from advanced_ml_integration import add_advanced_ml_menu_options
        menu_options = add_advanced_ml_menu_options()
        print(f"   ✓ Menu options available: {len(menu_options)} options")
        for key, desc in menu_options.items():
            print(f"      {key}: {desc}")
        
        print("\n🎉 All integration tests passed!")
        print("\nYour advanced ML system is ready to use!")
        print("Run 'python main.py' and use AML1-AML7 options for advanced features.")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Install missing dependencies: pip install torch torch-geometric")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_integration()