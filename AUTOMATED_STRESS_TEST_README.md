# Automated Iterative Stress Test

This script automates the iterative stress test process (Option Z from the main menu) to run completely automatically without manual intervention.

## Features

✅ **Fully Automated**: Runs 30 iterations with 20 vehicles per iteration automatically  
✅ **Point A to Point B Selection**: Uses the same notable locations as the main system  
✅ **Enhanced Excel Export**: Generates Excel file with automatic line charts for travel time trends  
✅ **Progress Tracking**: Shows real-time progress with a progress bar  
✅ **No Code Changes**: Uses existing system functions without modifying anything  
✅ **Automatic Charts**: Creates individual and combined line charts for each algorithm's performance

## Usage

### Quick Start

```bash
python3 automated_iterative_stress_test.py
```

### Step-by-Step Process

1. **Run the Script**

   ```bash
   python3 automated_iterative_stress_test.py
   ```

2. **Select Point A (Source)**

   - Choose from 10 notable locations (same as main system)
   - Enter the number (1-10) for your source location

3. **Select Point B (Destination)**

   - Choose from the remaining notable locations
   - Enter the number (1-10) for your destination location

4. **Confirm and Start**

   - Review the test parameters
   - Type 'y' to start the automated test

5. **Wait for Completion**

   - The script runs 30 iterations automatically
   - Each iteration adds 20 vehicles to the route
   - Progress is shown with a real-time progress bar
   - Takes approximately 5-10 minutes to complete

6. **Excel Report Generated**
   - Automatically saves to `london_simulation/iterative_stress_test/`
   - Option to open the Excel file immediately
   - Same format as manual iterative stress test

## What the Script Does

### Automation Process

1. **Initializes** the London traffic network (same as main system)
2. **Creates** notable locations using `create_evenly_distributed_notable_locations()`
3. **Creates** a test vehicle from Point A to Point B
4. **Runs 30 iterations** automatically:
   - Adds 20 vehicles per iteration along the test route
   - Updates congestion based on vehicle load
   - Recalculates routes for the test vehicle
   - Tracks all algorithm performance data
5. **Generates** comprehensive Excel report with multiple sheets

### Excel Output

The generated Excel file contains enhanced sheets with automatic chart generation:

- **Summary**: Test parameters and final results
- **Chart Data**: Raw data formatted for chart generation
- **A\*\_Chart**: Individual line chart showing A\* algorithm travel time trends
- **Shortest_Path_Chart**: Individual line chart for Shortest Path algorithm
- **Shortest_Path_Congestion_Aware_Chart**: Individual line chart for congestion-aware algorithm
- **Combined_Trends_Chart**: Combined line chart showing all three algorithms on one graph
- **Raw Data**: Detailed data for all iterations (same as manual process)
- **Added Vehicles**: List of all vehicles added during the test

### Chart Features

🎯 **Individual Algorithm Charts**: Each algorithm gets its own dedicated line chart showing:

- X-axis: Iteration number (0 to 30)
- Y-axis: Travel time in seconds
- Clear trend line showing performance degradation over iterations

📊 **Combined Trends Chart**: Single chart comparing all algorithms:

- Three colored lines (Blue: A\*, Orange: Shortest Path, Green: Congestion Aware)
- Legend clearly identifying each algorithm
- Easy visual comparison of algorithm performance

📈 **Professional Formatting**: All charts include:

- Proper titles and axis labels
- Consistent color scheme
- Markers on data points
- Professional styling suitable for presentations

## Example Output

```
================================================================================
AUTOMATED ITERATIVE STRESS TEST
================================================================================
Route: Central Business District → Residential Zone B
Iterations: 30
Vehicles per iteration: 20
Total vehicles to be added: 600

Progress: [==================================================] 100% (Complete!)

Route comparison (Initial vs Final):
  A*: 192.73s → 406.15s (+110.7%)
  Shortest Path: 163.84s → 163.84s (+0.0%)
  Shortest Path Congestion Aware: 192.73s → 406.15s (+110.7%)

✓ Excel report generated: london_simulation/iterative_stress_test/iterative_stress_test_vehicle_1_1751668515.xlsx
```

## Key Benefits

### Time Saving

- **Manual Process**: ~30-60 minutes with constant user input
- **Automated Process**: ~5-10 minutes with zero user input after setup

### Consistency

- Eliminates human error in iteration counting
- Consistent 20 vehicles per iteration
- No missed steps or variations

### Identical Results

- Uses the exact same functions as the manual process
- Same Excel format and data structure
- Perfect compatibility with existing analysis workflows

## Technical Details

### Dependencies

The script uses the same dependencies as the main system:

- osmnx
- networkx
- matplotlib
- pandas
- numpy
- openpyxl

### Integration

- **No modifications** to existing code
- Uses existing functions from:
  - `models.py` - Network loading and vehicle creation
  - `routing.py` - Route calculation algorithms
  - `congestion.py` - Congestion scenarios and updates
  - `vehicle_management.py` - Vehicle addition and metrics
  - `stress_testing.py` - Excel export functionality

### File Structure

```
london_simulation/
├── iterative_stress_test/
│   └── iterative_stress_test_vehicle_1_[timestamp].xlsx
└── excel_data/
    └── congestion_Normal_[timestamp].xlsx
```

## Troubleshooting

### Common Issues

**"Command not found: python"**

- Use `python3` instead of `python`

**"Module not found" errors**

- Ensure all dependencies are installed
- Run from the same directory as the main system files

**"No valid path found"**

- Try different Point A and Point B combinations
- Ensure the London network loaded successfully

### Getting Help

If you encounter issues:

1. Check that all main system dependencies are installed
2. Verify you're running from the correct directory
3. Try running the main system first to ensure it works
4. Check the console output for specific error messages

## Comparison with Manual Process

| Feature             | Manual Process | Automated Process  |
| ------------------- | -------------- | ------------------ |
| User Input Required | Continuous     | Initial setup only |
| Time Required       | 30-60 minutes  | 5-10 minutes       |
| Consistency         | Variable       | Perfect            |
| Excel Output        | Same           | Same               |
| Data Quality        | Same           | Same               |
| Error Prone         | Yes            | No                 |

The automated script provides identical results to the manual process but with significant time savings and improved consistency.
