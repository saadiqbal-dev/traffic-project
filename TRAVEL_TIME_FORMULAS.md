# Travel Time Formulas for Traffic Routing Algorithms

## Overview

This document provides detailed mathematical formulas for travel time calculations used by each routing algorithm in the traffic simulation system. All algorithms use the **Unified Travel Time Calculator** as the foundation to ensure consistency and realistic results.

---

## 1. Base Travel Time Calculation (Foundation for All Algorithms)

### Core Physics Formula

```
Travel Time (seconds) = Distance (meters) / Speed (m/s)
```

### Implementation Formula

```
travel_time = length_meters / (speed_kph / 3.6)
```

**Where:**

- `length_meters`: Edge length in meters (from OSM data)
- `speed_kph`: Speed in kilometers per hour
- `3.6`: Conversion factor from km/h to m/s (1 km/h = 1/3.6 m/s)

### Default Values

- Default speed: 30 km/h
- Default length: 100 meters
- Minimum speed threshold: > 0 km/h (returns infinity if ≤ 0)

---

## 2. Congestion Multiplier Formula

### Research-Based Piecewise Linear Function

```
multiplier = f(congestion_level) where congestion_level ∈ [1, 10]
```

### Detailed Formula by Congestion Range

**Light Traffic (1.0 ≤ congestion ≤ 2.0):**

```
multiplier = 1.0 + (congestion_level - 1.0) × 0.1
Range: 1.0x to 1.1x (0-10% penalty)
```

**Moderate Traffic (2.0 < congestion ≤ 4.0):**

```
multiplier = 1.1 + (congestion_level - 2.0) × 0.1
Range: 1.1x to 1.3x (10-30% penalty)
```

**Heavy Traffic (4.0 < congestion ≤ 6.0):**

```
multiplier = 1.3 + (congestion_level - 4.0) × 0.15
Range: 1.3x to 1.6x (30-60% penalty)
```

**Severe Congestion (6.0 < congestion ≤ 8.0):**

```
multiplier = 1.6 + (congestion_level - 6.0) × 0.2
Range: 1.6x to 2.0x (60-100% penalty)
```

**Gridlock (8.0 < congestion ≤ 10.0):**

```
multiplier = min(2.0 + (congestion_level - 8.0) × 0.25, 2.5)
Range: 2.0x to 2.5x (100-150% penalty, capped at 2.5x)
```

### Final Congested Travel Time

```
congested_time = base_travel_time × congestion_multiplier
```

---

## 3. Algorithm-Specific Formulas

### 3.1 Enhanced A\* Algorithm

**Multi-Criteria Cost Function:**

```
g_score(node) = (congestion_cost, travel_time_cost, distance_cost)
```

**Edge Cost Calculation:**

```
edge_congestion_cost = congestion_level
edge_travel_time = calculate_edge_travel_time(G, u, v, k, apply_congestion=True)
edge_distance = edge_length_meters
```

**Cumulative Cost:**

```
tentative_g_score = (
    current_g[0] + edge_congestion_cost,
    current_g[1] + edge_travel_time,
    current_g[2] + edge_distance
)
```

**Heuristic Function:**

```
h(n1, n2) = (congestion_penalty, travel_time_estimate, distance_estimate)

distance_estimate = √((x1-x2)² + (y1-y2)²)
travel_time_estimate = distance_estimate / (50/3.6)  // 50 km/h average
congestion_penalty = 2.0  // Moderate congestion assumption
```

**F-Score (Total Cost):**

```
f_score = g_score + heuristic
```

**Priority:** Congestion > Travel Time > Distance (lexicographic ordering)

### 3.2 Enhanced Dijkstra's Algorithm

**Distance Calculation:**

```
edge_weight = calculate_edge_travel_time(G, u, v, k, apply_congestion=True)
new_distance = current_distance + edge_weight
```

**Path Selection:**

```
if new_distance < distances[neighbor]:
    distances[neighbor] = new_distance
    previous[neighbor] = current_node
```

**Final Travel Time:**

```
total_time = Σ(edge_travel_times) for all edges in optimal path
```

### 3.3 Shortest Path Algorithm

**Path Finding:**

```
path = nx.shortest_path(G, start, end, weight='length')
```

**Travel Time Calculation (NO Congestion):**

```
total_time = calculate_path_travel_time(G, path, apply_congestion=False)
```

**Edge Time Formula:**

```
edge_time = length_meters / (speed_kph / 3.6)
// No congestion multiplier applied
```

### 3.4 Shortest Path Congestion Aware Algorithm

**Path Finding (Same as Shortest Path):**

```
path = nx.shortest_path(G, start, end, weight='length')
```

**Travel Time Calculation (WITH Congestion):**

```
total_time = calculate_path_travel_time(G, path, apply_congestion=True)
```

**Edge Time Formula:**

```
edge_time = (length_meters / (speed_kph / 3.6)) × congestion_multiplier
```

---

## 4. Path Travel Time Calculation

### Complete Path Formula

```
total_travel_time = Σ(edge_travel_time_i) for i = 1 to n-1
```

**Where:**

- `n` = number of nodes in path
- `edge_travel_time_i` = travel time for edge between node*i and node*{i+1}

### Edge Selection for Multigraph

```
if multiple edges exist between nodes u and v:
    use first available edge key: k = edge_keys[0]
```

### Error Handling

```
if edge not found or calculation fails:
    use default_time = 60.0 seconds (1 minute)
```

---

## 5. London-Specific Speed Calculations

### Highway Type Speed Mapping (km/h)

```
london_highway_speeds = {
    'motorway': 65,        // M25, A40(M) - capped for city sections
    'trunk': 45,           // A4, A40 - major roads with traffic
    'primary': 35,         // A1, A10 - main roads through London
    'secondary': 30,       // A roads, B roads
    'tertiary': 25,        // Local distributor roads
    'unclassified': 20,    // Minor roads
    'residential': 18,     // Residential streets
    'living_street': 15,   // Shared space, very slow
    'service': 15,         // Service roads, car parks
    'track': 12,           // Unpaved roads
    'path': 8              // Footpaths accessible to vehicles
}
```

### Speed Adjustment for London Traffic

```
if posted_speed > 50 km/h:
    actual_speed = min(posted_speed × 0.6, 35)  // Cap at 35 km/h
elif posted_speed > 30 km/h:
    actual_speed = posted_speed × 0.75          // 25% reduction
else:
    actual_speed = max(posted_speed × 0.9, 12)  // Slight reduction, min 12 km/h
```

---

## 6. Service Rate Calculations

### Service Rate Formula

```
service_rate = 1 / computation_time  (routes per second)

if computation_time ≤ 0:
    service_rate = ∞
```

**Where:**

- `computation_time`: Time taken by algorithm to find the route (in seconds)

---

## 7. Validation Formulas

### Travel Time Reasonableness Check

```
min_time_per_edge = 10 seconds
max_time_per_edge = 7200 seconds (2 hours)

edges_count = max(1, path_length - 1)
avg_time_per_edge = total_travel_time / edges_count

valid = min_time_per_edge ≤ avg_time_per_edge ≤ max_time_per_edge
```

### Algorithm Consistency Check

```
max_acceptable_ratio = 3.0
ratio = max_travel_time / min_travel_time

consistent = ratio ≤ max_acceptable_ratio
```

---

## 8. A\* Machine Learning Prediction Formulas

### Feature Extraction

```
straight_distance = √((source_x - dest_x)² + (source_y - dest_y)²)
is_rush_hour = 1 if (7 ≤ hour ≤ 9) or (17 ≤ hour ≤ 19) else 0
is_weekend = 1 if day_type == 'weekend' else 0
```

### ML Model Prediction

```
predicted_time = RandomForestRegressor.predict(scaled_features)
```

**Features used:**

- Source/destination coordinates (x, y)
- Straight-line distance
- Vehicle count
- Hour of day
- Rush hour indicator
- Weekend indicator
- Average network congestion
- Scenario encoding (one-hot)

---

## 9. Summary of Key Differences

| Algorithm                          | Path Selection              | Congestion Applied      | Formula Priority                       |
| ---------------------------------- | --------------------------- | ----------------------- | -------------------------------------- |
| **Enhanced A\***                   | Multi-criteria optimization | ✅ Yes                  | Congestion → Time → Distance           |
| **Enhanced Dijkstra**              | Minimum travel time         | ✅ Yes                  | Travel time only                       |
| **Shortest Path**                  | Minimum distance            | ❌ No                   | Distance only                          |
| **Shortest Path Congestion Aware** | Minimum distance            | ✅ Yes (time calc only) | Distance for path, congestion for time |

---

## 10. Constants and Limits

```python
DEFAULT_SPEED_KPH = 30
DEFAULT_LENGTH_METERS = 100
MAX_CONGESTION_MULTIPLIER = 2.5
MAX_ACCEPTABLE_ALGORITHM_RATIO = 3.0
DEFAULT_EDGE_TIME_SECONDS = 60.0
KMH_TO_MS_CONVERSION = 3.6
```

---

## Implementation Notes

1. **Consistency**: All algorithms use the same `UnifiedTravelTimeCalculator` for base calculations
2. **Realism**: Congestion penalties are research-based and capped at 2.5x
3. **Robustness**: Fallback values prevent infinite or zero travel times
4. **London-Specific**: Speed adjustments reflect actual London driving conditions
5. **Validation**: Built-in checks ensure reasonable results across all algorithms

This unified approach ensures that travel time calculations are consistent, realistic, and comparable across all routing algorithms in the system.
