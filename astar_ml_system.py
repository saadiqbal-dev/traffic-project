"""
Simplified A* Machine Learning Optimization System
==================================================
This module provides a clean, modular ML system focused solely on optimizing A* routing
for the London traffic simulation. It predicts optimal routes based on time and scenario.

Key Features:
- Collects A* training data across all congestion scenarios
- Trains neural network to predict optimal routes considering time and distance
- Simple GUI for testing predictions
- Modular design for future expansion
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# ML imports with fallback
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️  scikit-learn not available. Install with: pip install scikit-learn")

# GUI imports with fallback
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    # Test tkinter initialization on macOS
    test_root = tk.Tk()
    test_root.withdraw()  # Hide the test window
    test_root.destroy()   # Clean up
    GUI_AVAILABLE = True
except Exception as e:
    GUI_AVAILABLE = False
    # Don't print the error during import, only when GUI is actually requested
    tk = None
    ttk = None
    messagebox = None
    FigureCanvasTkAgg = None

# Import local modules
from models import load_london_network, generate_initial_congestion, create_evenly_distributed_notable_locations
from routing import enhanced_a_star_algorithm, create_congestion_graph
from congestion import apply_consistent_congestion_scenario
from vehicle_management import add_bulk_vehicles


class AStarDataCollector:
    """
    Collects A* routing data across different scenarios and times.
    Simple and focused on essential features only.
    """
    
    def __init__(self):
        self.scenarios = ["Normal", "Morning", "Evening", "Weekend", "Special"]
        self.vehicle_counts = [0, 25, 50, 100, 150]
        self.data_dir = os.path.join('london_simulation', 'astar_ml_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_astar_data(self, samples_per_scenario: int = 50) -> str:
        """
        Collect A* routing data across all scenarios.
        
        Args:
            samples_per_scenario: Number of samples to collect per scenario
            
        Returns:
            Path to saved data file
        """
        print("🚀 Starting A* Data Collection...")
        
        # Load network
        G = load_london_network()
        notable_locations = create_evenly_distributed_notable_locations(G)
        location_names = list(notable_locations.keys())
        
        all_data = []
        
        for scenario in self.scenarios:
            print(f"\n📊 Collecting data for {scenario} scenario...")
            
            for vehicle_count in self.vehicle_counts:
                print(f"  🚗 Vehicle count: {vehicle_count}")
                
                # Apply scenario-specific congestion
                original_congestion = generate_initial_congestion(G)
                congestion_data, _ = apply_consistent_congestion_scenario(
                    G, {}, scenario, original_congestion
                )
                
                # Reset network and add vehicles
                vehicles = []
                for u, v, k in G.edges(keys=True):
                    G[u][v][k]['vehicle_count'] = 0
                
                if vehicle_count > 0:
                    add_bulk_vehicles(G, vehicles, vehicle_count, congestion_data, False, notable_locations)
                    from congestion import update_congestion_based_on_vehicles
                    update_congestion_based_on_vehicles(G, congestion_data, original_congestion)
                
                # Create congestion graph for A*
                G_congestion = create_congestion_graph(G, congestion_data)
                
                # Collect samples
                for sample_idx in range(samples_per_scenario):
                    # Random source and destination
                    source_name = np.random.choice(location_names)
                    dest_name = np.random.choice(location_names)
                    
                    if source_name == dest_name:
                        continue
                    
                    source_node = notable_locations[source_name]
                    dest_node = notable_locations[dest_name]
                    
                    # Run A* algorithm
                    try:
                        path, travel_time, comp_time = enhanced_a_star_algorithm(
                            G_congestion, source_node, dest_node
                        )
                        
                        if path and len(path) > 1:
                            # Calculate additional metrics
                            total_distance = self._calculate_path_distance(G, path)
                            avg_congestion = self._calculate_path_congestion(G_congestion, path)
                            
                            # Simulate time of day
                            hour = self._scenario_to_hour(scenario)
                            day_type = self._scenario_to_day_type(scenario)
                            
                            # Extract features
                            features = self._extract_features(
                                G, source_node, dest_node, scenario, vehicle_count,
                                hour, day_type, avg_congestion
                            )
                            
                            # Create data sample
                            sample = {
                                **features,
                                'path_length': len(path),
                                'travel_time': travel_time,
                                'total_distance': total_distance,
                                'avg_congestion': avg_congestion,
                                'computation_time': comp_time
                            }
                            
                            all_data.append(sample)
                            
                    except Exception as e:
                        print(f"    ⚠️  Error in sample {sample_idx}: {e}")
                        continue
                
                print(f"    ✅ Collected {len([d for d in all_data if d.get('scenario') == scenario])} valid samples")
        
        # Save data
        df = pd.DataFrame(all_data)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        data_file = os.path.join(self.data_dir, f"astar_training_data_{timestamp}.csv")
        df.to_csv(data_file, index=False)
        
        print(f"\n✅ Data collection complete! Total samples: {len(df)}")
        print(f"💾 Data saved to: {data_file}")
        
        return data_file
    
    def _extract_features(self, G, source: int, dest: int, scenario: str, 
                         vehicle_count: int, hour: int, day_type: str, 
                         avg_congestion: float) -> Dict:
        """Extract essential features for ML training."""
        
        # Get coordinates
        source_x = G.nodes[source]['x']
        source_y = G.nodes[source]['y']
        dest_x = G.nodes[dest]['x']
        dest_y = G.nodes[dest]['y']
        
        # Calculate straight-line distance
        straight_distance = np.sqrt((source_x - dest_x)**2 + (source_y - dest_y)**2)
        
        # Time-based features
        is_rush_hour = 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0
        is_weekend = 1 if day_type == 'weekend' else 0
        
        # Scenario encoding
        scenario_features = {f'scenario_{s.lower()}': 1 if scenario == s else 0 
                           for s in self.scenarios}
        
        return {
            'source_x': source_x,
            'source_y': source_y,
            'dest_x': dest_x,
            'dest_y': dest_y,
            'straight_distance': straight_distance,
            'vehicle_count': vehicle_count,
            'hour': hour,
            'is_rush_hour': is_rush_hour,
            'is_weekend': is_weekend,
            'avg_network_congestion': avg_congestion,
            'scenario': scenario,
            **scenario_features
        }
    
    def _calculate_path_distance(self, G, path: List[int]) -> float:
        """Calculate total distance of a path."""
        total_distance = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in G and v in G[u]:
                edge_keys = list(G[u][v].keys())
                if edge_keys:
                    total_distance += G[u][v][edge_keys[0]].get('length', 0)
        return total_distance
    
    def _calculate_path_congestion(self, G_congestion, path: List[int]) -> float:
        """Calculate average congestion along a path."""
        congestions = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if u in G_congestion and v in G_congestion[u]:
                edge_keys = list(G_congestion[u][v].keys())
                if edge_keys:
                    edge_data = G_congestion[u][v][edge_keys[0]]
                    # Handle both dict and numeric edge data
                    if isinstance(edge_data, dict):
                        congestion = edge_data.get('congestion', 1.0)
                    else:
                        # If edge_data is numeric (like numpy.float64), use it directly
                        congestion = float(edge_data) if edge_data else 1.0
                    congestions.append(congestion)
        return np.mean(congestions) if congestions else 1.0
    
    def _scenario_to_hour(self, scenario: str) -> int:
        """Convert scenario to representative hour."""
        hour_mapping = {
            "Normal": np.random.randint(10, 16),  # Midday
            "Morning": np.random.randint(7, 10),  # Morning rush
            "Evening": np.random.randint(17, 20), # Evening rush
            "Weekend": np.random.randint(9, 22),  # Flexible weekend
            "Special": np.random.randint(18, 23)  # Evening events
        }
        return hour_mapping.get(scenario, 12)
    
    def _scenario_to_day_type(self, scenario: str) -> str:
        """Convert scenario to day type."""
        return 'weekend' if scenario == 'Weekend' else 'weekday'


class AStarPredictor:
    """
    Neural network predictor for A* optimal routing.
    Uses simple RandomForest for reliability and speed.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.feature_columns = None
        self.is_trained = False
        self.model_dir = os.path.join('london_simulation', 'astar_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def train(self, data_file: str) -> Dict:
        """
        Train the A* prediction model.
        
        Args:
            data_file: Path to training data CSV
            
        Returns:
            Training metrics
        """
        if not ML_AVAILABLE:
            raise ImportError("scikit-learn required for training")
            
        print("🔥 Training A* Prediction Model...")
        
        # Load data
        df = pd.read_csv(data_file)
        print(f"📊 Loaded {len(df)} training samples")
        
        # Check minimum samples
        if len(df) < 5:
            print("⚠️  Warning: Very few samples for training. Consider collecting more data.")
            if len(df) < 2:
                raise ValueError("Need at least 2 samples for training")
        
        # Prepare features
        exclude_cols = ['travel_time', 'total_distance', 'path_length', 
                       'computation_time', 'scenario']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_columns].values
        y = df['travel_time'].values  # Predict travel time as primary metric
        
        # Split data (adjust test_size for small datasets)
        test_size = 0.2 if len(df) >= 10 else 1 / len(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.feature_columns = feature_columns
        self.is_trained = True
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2_score': r2,
            'training_time': training_time,
            'samples_trained': len(X_train)
        }
        
        print(f"✅ Training complete!")
        print(f"   R² Score: {r2:.4f}")
        print(f"   RMSE: {np.sqrt(mse):.2f} seconds")
        print(f"   Training time: {training_time:.1f} seconds")
        
        return metrics
    
    def predict_optimal_route(self, source_name: str, dest_name: str, 
                            hour: int, scenario: str) -> Dict:
        """
        Predict optimal A* route for given conditions.
        
        Args:
            source_name: Source location name
            dest_name: Destination location name
            hour: Hour of day (0-23)
            scenario: Traffic scenario
            
        Returns:
            Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
            
        # Load network and locations
        G = load_london_network()
        notable_locations = create_evenly_distributed_notable_locations(G)
        
        if source_name not in notable_locations or dest_name not in notable_locations:
            raise ValueError("Invalid location names")
            
        source_node = notable_locations[source_name]
        dest_node = notable_locations[dest_name]
        
        # Extract features
        collector = AStarDataCollector()
        features = collector._extract_features(
            G, source_node, dest_node, scenario, 50,  # Assume medium traffic
            hour, 'weekend' if scenario == 'Weekend' else 'weekday', 2.0
        )
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        # Scale and predict
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        predicted_time = self.model.predict(X_scaled)[0]
        
        # Also run actual A* for comparison
        original_congestion = generate_initial_congestion(G)
        congestion_data, _ = apply_consistent_congestion_scenario(
            G, {}, scenario, original_congestion
        )
        G_congestion = create_congestion_graph(G, congestion_data)
        
        actual_path, actual_time, comp_time = enhanced_a_star_algorithm(
            G_congestion, source_node, dest_node
        )
        
        return {
            'predicted_time': predicted_time,
            'actual_time': actual_time,
            'actual_path': actual_path,
            'computation_time': comp_time,
            'accuracy': abs(predicted_time - actual_time) / actual_time * 100 if actual_time > 0 else 0
        }
    
    def save_model(self, filepath: str):
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True
        
        print(f"📂 Model loaded from: {filepath}")


class AStarMLGUI:
    """
    Simple GUI for testing A* ML predictions.
    """
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI not available on this system")
            
        self.predictor = AStarPredictor()
        self.root = tk.Tk()
        self.root.title("A* ML Route Predictor")
        self.root.geometry("800x600")
        
        # Make window more prominent
        self.root.lift()  # Bring to front
        self.root.attributes('-topmost', True)  # Keep on top initially
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))  # Remove topmost after 1 second
        
        # Don't auto-load model to avoid initialization issues
        # User can train model manually if needed
        
        self._create_widgets()
    
    def _try_load_model(self):
        """Try to load the latest trained model."""
        model_dir = self.predictor.model_dir
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
            if model_files:
                latest_model = os.path.join(model_dir, sorted(model_files)[-1])
                try:
                    self.predictor.load_model(latest_model)
                except Exception as e:
                    print(f"Failed to load model: {e}")
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="A* ML Route Predictor", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Training", padding="5")
        train_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(train_frame, text="Load Locations", 
                  command=self._load_locations).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(train_frame, text="Collect Training Data", 
                  command=self._collect_data).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(train_frame, text="Train Model", 
                  command=self._train_model).grid(row=0, column=2, padx=(0, 5))
        
        # Prediction section
        pred_frame = ttk.LabelFrame(main_frame, text="Route Prediction", padding="5")
        pred_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Location selection
        ttk.Label(pred_frame, text="From:").grid(row=0, column=0, sticky=tk.W)
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(pred_frame, textvariable=self.source_var, width=20)
        self.source_combo.grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(pred_frame, text="To:").grid(row=0, column=2, sticky=tk.W)
        self.dest_var = tk.StringVar()
        self.dest_combo = ttk.Combobox(pred_frame, textvariable=self.dest_var, width=20)
        self.dest_combo.grid(row=0, column=3, padx=(5, 0))
        
        # Time and scenario
        ttk.Label(pred_frame, text="Hour (0-23):").grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        self.hour_var = tk.StringVar(value="12")
        ttk.Entry(pred_frame, textvariable=self.hour_var, width=10).grid(row=1, column=1, padx=(5, 10), pady=(5, 0))
        
        ttk.Label(pred_frame, text="Scenario:").grid(row=1, column=2, sticky=tk.W, pady=(5, 0))
        self.scenario_var = tk.StringVar(value="Normal")
        scenario_combo = ttk.Combobox(pred_frame, textvariable=self.scenario_var, 
                                     values=["Normal", "Morning", "Evening", "Weekend", "Special"], width=15)
        scenario_combo.grid(row=1, column=3, padx=(5, 0), pady=(5, 0))
        
        # Predict button
        ttk.Button(pred_frame, text="Predict Route", 
                  command=self._predict_route).grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.results_text = tk.Text(results_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Initialize location data
        self.locations_loaded = False
        self.location_names = []
        
        # Add message to load locations
        self.source_combo['values'] = ["Click 'Load Locations' first"]
        self.dest_combo['values'] = ["Click 'Load Locations' first"]
        
        # Add welcome message after all widgets are created
        self._show_welcome_message()
    
    def _show_welcome_message(self):
        """Show welcome message in the results area."""
        self._log("🚀 Welcome to A* ML Route Predictor!")
        self._log("📋 Getting Started:")
        self._log("   1. Click 'Load Locations' to initialize location data")
        self._log("   2. Click 'Collect Training Data' to gather route data")  
        self._log("   3. Click 'Train Model' to train the ML predictor")
        self._log("   4. Select locations and predict routes!")
        self._log("")
    
    def _load_locations(self):
        """Load available location names."""
        if self.locations_loaded:
            self._log("✅ Locations already loaded")
            return
            
        try:
            self._log("🗺️  Loading London street network...")
            self._log("(This may take 10-30 seconds...)")
            
            # Update GUI to show we're loading
            self.root.update()
            
            G = load_london_network()
            self._log("🏢 Creating notable locations...")
            locations = create_evenly_distributed_notable_locations(G)
            location_names = sorted(locations.keys())
            
            self.source_combo['values'] = location_names
            self.dest_combo['values'] = location_names
            self.location_names = location_names
            self.locations_loaded = True
            
            if location_names:
                self.source_var.set(location_names[0])
                self.dest_var.set(location_names[1] if len(location_names) > 1 else location_names[0])
            
            self._log(f"✅ Loaded {len(location_names)} locations successfully!")
                
        except Exception as e:
            self._log(f"❌ Error loading locations: {e}")
    
    def _collect_data(self):
        """Collect training data."""
        self._log("Starting data collection...")
        try:
            collector = AStarDataCollector()
            data_file = collector.collect_astar_data(samples_per_scenario=30)
            self._log(f"✅ Data collection complete: {data_file}")
        except Exception as e:
            self._log(f"❌ Data collection failed: {e}")
    
    def _train_model(self):
        """Train the ML model."""
        if not ML_AVAILABLE:
            messagebox.showerror("Error", "scikit-learn not available for training")
            return
            
        # Find latest data file
        data_dir = os.path.join('london_simulation', 'astar_ml_data')
        if not os.path.exists(data_dir):
            messagebox.showerror("Error", "No training data found. Collect data first.")
            return
            
        data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        if not data_files:
            messagebox.showerror("Error", "No training data files found. Collect data first.")
            return
            
        latest_data = os.path.join(data_dir, sorted(data_files)[-1])
        
        self._log(f"Training model with: {os.path.basename(latest_data)}")
        try:
            metrics = self.predictor.train(latest_data)
            
            # Save model
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_file = os.path.join(self.predictor.model_dir, f"astar_model_{timestamp}.pkl")
            self.predictor.save_model(model_file)
            
            self._log("✅ Training complete!")
            self._log(f"   R² Score: {metrics['r2_score']:.4f}")
            self._log(f"   RMSE: {metrics['rmse']:.2f} seconds")
            self._log(f"   Training samples: {metrics['samples_trained']}")
            
        except Exception as e:
            self._log(f"❌ Training failed: {e}")
    
    def _predict_route(self):
        """Predict optimal route."""
        if not self.locations_loaded:
            messagebox.showerror("Error", "Please load locations first by clicking 'Load Locations'")
            return
            
        if not self.predictor.is_trained:
            messagebox.showerror("Error", "Model not trained. Train model first.")
            return
            
        try:
            source = self.source_var.get()
            dest = self.dest_var.get()
            hour = int(self.hour_var.get())
            scenario = self.scenario_var.get()
            
            if not source or not dest:
                messagebox.showerror("Error", "Please select source and destination")
                return
                
            if hour < 0 or hour > 23:
                messagebox.showerror("Error", "Hour must be between 0 and 23")
                return
            
            self._log(f"Predicting route: {source} → {dest}")
            self._log(f"Time: {hour}:00, Scenario: {scenario}")
            
            result = self.predictor.predict_optimal_route(source, dest, hour, scenario)
            
            self._log("🔮 Prediction Results:")
            self._log(f"   Predicted travel time: {result['predicted_time']:.2f} seconds")
            self._log(f"   Actual A* travel time: {result['actual_time']:.2f} seconds")
            self._log(f"   Prediction accuracy: {100 - result['accuracy']:.1f}%")
            self._log(f"   A* computation time: {result['computation_time']:.4f} seconds")
            
            if result['actual_path']:
                self._log(f"   Route length: {len(result['actual_path'])} nodes")
                self._log(f"   Path: {result['actual_path'][:5]}...{result['actual_path'][-5:] if len(result['actual_path']) > 10 else result['actual_path']}")
            
        except Exception as e:
            self._log(f"❌ Prediction failed: {e}")
    
    def _log(self, message: str):
        """Log message to results text."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()


# Module Integration Functions

def quick_data_collection() -> str:
    """Quick function to collect A* training data."""
    collector = AStarDataCollector()
    return collector.collect_astar_data(samples_per_scenario=20)

def quick_training(data_file: str) -> AStarPredictor:
    """Quick function to train A* model."""
    predictor = AStarPredictor()
    predictor.train(data_file)
    return predictor

def launch_gui():
    """Launch the A* ML GUI."""
    if not ML_AVAILABLE:
        print("❌ GUI requires scikit-learn. Install with: pip install scikit-learn")
        return
    
    if not GUI_AVAILABLE:
        print("❌ GUI not available on this system.")
        print("This is likely due to macOS tkinter compatibility issues.")
        print("You can still use the command-line interface:")
        print("- ASML1: Collect training data")
        print("- ASML2: Train model")
        print("- ASML4: Quick demo")
        return
    
    try:
        print("🖥️  Launching A* ML GUI window...")
        print("(Look for GUI window - it may open behind other windows)")
        gui = AStarMLGUI()
        print("✅ GUI initialized successfully")
        gui.run()
        print("GUI closed")
    except Exception as e:
        print(f"❌ GUI failed to launch: {e}")
        print("Falling back to command-line interface.")

# Main execution
if __name__ == "__main__":
    print("🚀 A* Machine Learning System")
    print("=" * 40)
    
    choice = input("Choose option:\n1. Collect data\n2. Train model\n3. Launch GUI\n4. Quick demo\nEnter choice (1-4): ")
    
    if choice == "1":
        quick_data_collection()
    elif choice == "2":
        data_files = []
        data_dir = os.path.join('london_simulation', 'astar_ml_data')
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        
        if data_files:
            latest_data = os.path.join(data_dir, sorted(data_files)[-1])
            quick_training(latest_data)
        else:
            print("No training data found. Run option 1 first.")
    elif choice == "3":
        launch_gui()
    elif choice == "4":
        print("Running quick demo...")
        data_file = quick_data_collection()
        predictor = quick_training(data_file)
        print("Demo complete! Use option 3 to launch GUI.")
    else:
        print("Invalid choice")