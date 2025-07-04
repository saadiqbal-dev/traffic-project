# London Traffic Simulation

A comprehensive traffic simulation project that models vehicle routing and congestion in London's road network using real OpenStreetMap data.

## Features

- **Real London Road Network**: Uses OpenStreetMap data via OSMnx
- **Multiple Routing Algorithms**: A*, Greedy Best-First Search, Enhanced Dijkstra
- **Congestion Modeling**: MM1 queuing theory for realistic traffic flow
- **Interactive Scenarios**: Normal, Morning Rush, Evening Rush, Weekend, Special Events
- **Vehicle Management**: Add, track, and analyze vehicles with different routes
- **Stress Testing**: Iterative stress tests to analyze route performance under load
- **Data Analysis**: Excel reports, statistics, and visualizations
- **Real-time Visualization**: Interactive maps showing congestion and routes

## Installation

1. Clone or download this project
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install osmnx networkx numpy matplotlib pandas openpyxl scikit-learn
```

## Usage

Run the simulation:

```bash
python main.py
```

This will start the interactive menu system where you can:

- Apply different congestion scenarios
- Add vehicles with custom routes
- Compare routing algorithm performance
- Run stress tests
- Export analysis to Excel
- Visualize traffic patterns

## Project Structure

- `main.py` - Main entry point with interactive menu
- `models.py` - Core data structures (Vehicle class, graph utilities)
- `routing.py` - Routing algorithms (A*, Greedy, Dijkstra)
- `congestion.py` - Congestion modeling and MM1 queuing
- `visualization.py` - Map plotting and route visualization
- `vehicle_management.py` - Vehicle addition and tracking
- `analysis.py` - Excel reports and statistical analysis
- `stress_testing.py` - Iterative stress testing functionality

## Key Features

### Routing Algorithms

1. **A* Algorithm**: Optimizes for travel time with congestion awareness
2. **Greedy Best-First**: Optimizes for shortest distance
3. **Enhanced Dijkstra**: Congestion-sensitive shortest path

### Congestion Scenarios

- **Normal**: Baseline traffic (2-10 congestion scale)
- **Morning Rush**: High congestion (5-6 average)
- **Evening Rush**: Peak congestion (7-9 average)
- **Weekend**: Moderate congestion (4-8 average)
- **Special Events**: Custom congestion patterns

### Analysis Tools

- Algorithm performance comparison
- Travel time analysis
- Congestion impact reports
- Service rate degradation analysis
- MM1 queuing statistics
- Excel export functionality

## Example Workflow

1. Start the simulation
2. Add vehicles between notable locations
3. Apply different congestion scenarios
4. Compare algorithm performance
5. Run stress tests to analyze route stability
6. Export results to Excel for further analysis

## Output

The simulation creates several output directories:
- `london_simulation/excel_data/` - Congestion data exports
- `london_simulation/scenario_maps/` - Visualization maps
- `london_simulation/analysis_reports/` - Comprehensive analysis
- `london_simulation/impact_reports/` - Traffic impact studies
- `london_simulation/stress_test_reports/` - Stress test results

## Dependencies

- **OSMnx**: Real-world road network data
- **NetworkX**: Graph algorithms and data structures
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting
- **Pandas**: Data analysis and Excel export
- **OpenPyXL**: Excel file creation
- **Scikit-learn**: Additional analysis tools

## Notes

- The simulation uses the City of London area for performance
- All output files include timestamps for version tracking
- The system supports both automated and manual vehicle placement
- Congestion values range from 1-10 (low to high)
- MM1 queuing model provides realistic traffic flow simulation