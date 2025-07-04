#!/usr/bin/env python3
"""
Debug GUI to isolate the exact widget creation issue
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime

class DebugAStarGUI:
    def __init__(self):
        print("1. Creating root window...")
        self.root = tk.Tk()
        self.root.title("Debug A* GUI")
        self.root.geometry("800x600")
        
        print("2. Setting window properties...")
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))
        
        print("3. Creating widgets...")
        self._create_widgets()
        print("4. GUI initialization complete!")
        
    def _create_widgets(self):
        print("  3.1. Creating main frame...")
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        print("  3.2. Creating title...")
        title_label = ttk.Label(main_frame, text="Debug A* ML Route Predictor", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        print("  3.3. Creating training frame...")
        train_frame = ttk.LabelFrame(main_frame, text="Training", padding="5")
        train_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        print("  3.4. Creating buttons...")
        ttk.Button(train_frame, text="Load Locations", 
                  command=self._test_button).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(train_frame, text="Collect Data", 
                  command=self._test_button).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(train_frame, text="Train Model", 
                  command=self._test_button).grid(row=0, column=2, padx=(0, 5))
        
        print("  3.5. Creating prediction frame...")
        pred_frame = ttk.LabelFrame(main_frame, text="Route Prediction", padding="5")
        pred_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        print("  3.6. Creating form controls...")
        ttk.Label(pred_frame, text="From:").grid(row=0, column=0, sticky=tk.W)
        self.source_combo = ttk.Combobox(pred_frame, width=20, values=["Test Location 1", "Test Location 2"])
        self.source_combo.grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(pred_frame, text="To:").grid(row=0, column=2, sticky=tk.W)
        self.dest_combo = ttk.Combobox(pred_frame, width=20, values=["Test Location 1", "Test Location 2"])
        self.dest_combo.grid(row=0, column=3, padx=(5, 0))
        
        print("  3.7. Creating results frame...")
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        print("  3.8. Creating text widget...")
        self.results_text = tk.Text(results_frame, height=15, width=80)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        print("  3.9. Configuring grid weights...")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        print("  3.10. Adding initial message...")
        self._log("Debug GUI initialized successfully!")
        
    def _test_button(self):
        self._log("Button clicked!")
        
    def _log(self, message: str):
        print(f"    Logging: {message}")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
        
    def run(self):
        print("5. Starting mainloop...")
        self.root.mainloop()
        print("6. GUI closed")

def test_debug_gui():
    try:
        print("Starting debug GUI test...")
        gui = DebugAStarGUI()
        gui.run()
        return True
    except Exception as e:
        print(f"Error in debug GUI: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_debug_gui()