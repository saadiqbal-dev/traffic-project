#!/usr/bin/env python3
"""
Minimal GUI test to isolate the issue
"""

import tkinter as tk
from tkinter import ttk
from datetime import datetime

class TestGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Test GUI")
        self.root.geometry("600x400")
        
        # Make window prominent
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))
        
        self._create_widgets()
        
    def _create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Test GUI", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Button
        test_button = ttk.Button(main_frame, text="Test Button", command=self._test_action)
        test_button.grid(row=1, column=0, pady=(0, 10))
        
        # Text area
        self.results_text = tk.Text(main_frame, height=15, width=60)
        self.results_text.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Add test message
        self._log("GUI initialized successfully!")
        
    def _test_action(self):
        self._log("Button clicked!")
        
    def _log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)
        self.root.update()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    print("Testing minimal GUI...")
    gui = TestGUI()
    gui.run()
    print("GUI closed")