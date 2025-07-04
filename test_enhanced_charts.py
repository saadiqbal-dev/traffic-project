#!/usr/bin/env python3
"""
Quick test for the enhanced automated stress test with charts.
This runs a minimal version to verify chart generation works.
"""

import os
import sys
import time
from automated_iterative_stress_test import AutomatedIterativeStressTest

def test_enhanced_charts():
    """Test the enhanced chart generation functionality."""
    print("Testing Enhanced Automated Stress Test with Charts")
    print("=" * 60)
    
    # Create the stress test instance
    stress_test = AutomatedIterativeStressTest()
    
    # Initialize the system
    print("Initializing system...")
    if not stress_test.initialize_system():
        print("Failed to initialize system")
        return False
    
    # Use predefined route (Central Business District to Residential Zone B)
    source = 1236791408  # Central Business District
    destination = 9804685199  # Residential Zone B
    
    print(f"\nUsing predefined route: {source} → {destination}")
    
    # Run a minimal stress test (5 iterations, 5 vehicles per iteration)
    print("Running minimal stress test (5 iterations, 5 vehicles each)...")
    excel_file = stress_test.run_automated_stress_test(
        source=source, 
        destination=destination,
        iterations=5,  # Reduced for testing
        vehicles_per_iteration=5  # Reduced for testing
    )
    
    if excel_file:
        print(f"\n✓ Test completed successfully!")
        print(f"Excel file with charts generated: {excel_file}")
        
        # Check if file exists and has reasonable size
        if os.path.exists(excel_file):
            file_size = os.path.getsize(excel_file)
            print(f"File size: {file_size:,} bytes")
            
            if file_size > 10000:  # Should be at least 10KB with charts
                print("✓ File size indicates charts were likely generated successfully")
                return True
            else:
                print("⚠ File size seems small, charts may not have been generated")
                return False
        else:
            print("✗ Excel file was not created")
            return False
    else:
        print("✗ Test failed")
        return False

if __name__ == "__main__":
    success = test_enhanced_charts()
    if success:
        print("\n🎉 Enhanced chart generation test PASSED!")
    else:
        print("\n❌ Enhanced chart generation test FAILED!")
    
    sys.exit(0 if success else 1)
