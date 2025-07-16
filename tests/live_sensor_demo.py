#!/usr/bin/env python3
"""
Live Sensor Demo for NormalizedDynamics
=======================================

A standalone demonstration of real-time manifold learning with NormalizedDynamics.
This script simulates IoT sensor data and shows live 2D embedding updates.

Features:
- Real-time sensor data simulation (temperature, humidity, pressure, etc.)
- Live 2D embedding visualization with NormalizedDynamics
- Anomaly injection capabilities
- Performance metrics display

Usage:
    python tests/live_sensor_demo.py

Author: NormalizedDynamics Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
import argparse
from datetime import datetime

# Add parent directory to path to access src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import our algorithm
from src.normalized_dynamics_optimized import NormalizedDynamicsOptimized
from src.streaming_simulator import StreamingSensorSimulator


class LiveSensorDemo:
    """
    Real-time sensor data embedding demonstration.
    """
    
    def __init__(self, n_sensors=6, window_size=200, update_interval=100):
        """
        Initialize the live demo.
        
        Args:
            n_sensors: Number of sensors to simulate
            window_size: Number of points to keep in visualization
            update_interval: Animation update interval in milliseconds
        """
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.update_interval = update_interval
        
        # Initialize components
        self.simulator = StreamingSensorSimulator(n_sensors=n_sensors, update_interval=0.1)
        self.algorithm = NormalizedDynamicsOptimized(dim=2, max_iter=10, device='cpu')
        
        # Data storage
        self.sensor_history = deque(maxlen=window_size)
        self.embedding_history = deque(maxlen=window_size)
        self.time_history = deque(maxlen=window_size)
        self.anomaly_history = deque(maxlen=window_size)
        
        # Performance tracking
        self.update_times = deque(maxlen=50)
        self.total_points = 0
        self.start_time = time.time()
        
        # Setup visualization
        self.setup_plot()
        
    def setup_plot(self):
        """Setup the matplotlib figure and axes."""
        self.fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Live IoT Sensor Data Analysis with NormalizedDynamics', 
                         fontsize=16, fontweight='bold')
        
        # 2D Embedding plot
        self.ax_embedding = axes[0, 0]
        self.ax_embedding.set_title('Real-Time 2D Embedding')
        self.ax_embedding.set_xlabel('Dimension 1')
        self.ax_embedding.set_ylabel('Dimension 2')
        self.ax_embedding.grid(True, alpha=0.3)
        self.scatter = self.ax_embedding.scatter([], [], c=[], s=50, alpha=0.7, cmap='viridis')
        
        # Sensor readings plot
        self.ax_sensors = axes[0, 1]
        self.ax_sensors.set_title('Current Sensor Readings')
        self.ax_sensors.set_xlabel('Sensor')
        self.ax_sensors.set_ylabel('Value')
        self.sensor_bars = None
        
        # Performance metrics
        self.ax_performance = axes[1, 0]
        self.ax_performance.set_title('Performance Metrics')
        self.ax_performance.text(0.1, 0.8, '', transform=self.ax_performance.transAxes, 
                               fontsize=12, verticalalignment='top')
        self.ax_performance.set_xticks([])
        self.ax_performance.set_yticks([])
        
        # Sensor timeline
        self.ax_timeline = axes[1, 1]
        self.ax_timeline.set_title('Sensor Timeline')
        self.ax_timeline.set_xlabel('Time (s)')
        self.ax_timeline.set_ylabel('Temperature (°C)')
        
        plt.tight_layout()
        
    def update_visualization(self, frame):
        """Update the visualization with new data."""
        try:
            # Get new sensor reading
            reading = self.simulator.generate_sensor_reading()
            
            # Time the embedding update
            start_time = time.time()
            
            # Update algorithm with new point
            sensor_values = np.array(reading['values_array'])
            embedding = self.algorithm.update_embedding(sensor_values)
            
            update_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.update_times.append(update_time)
            
            # Store data
            self.sensor_history.append(reading)
            if len(embedding) > 0:
                self.embedding_history.append(embedding[-1])  # Latest point
            self.time_history.append(reading['timestamp'])
            self.anomaly_history.append(reading.get('has_anomaly', False))
            self.total_points += 1
            
            # Update embedding plot
            if len(self.embedding_history) > 1:
                embeddings = np.array(list(self.embedding_history))
                colors = np.array(list(self.anomaly_history), dtype=float)
                
                self.ax_embedding.clear()
                self.ax_embedding.set_title('Real-Time 2D Embedding')
                self.ax_embedding.set_xlabel('Dimension 1')
                self.ax_embedding.set_ylabel('Dimension 2')
                self.ax_embedding.grid(True, alpha=0.3)
                
                # Normal points in blue, anomalies in red
                normal_mask = colors == 0
                anomaly_mask = colors == 1
                
                if np.any(normal_mask):
                    self.ax_embedding.scatter(embeddings[normal_mask, 0], embeddings[normal_mask, 1], 
                                            c='blue', s=30, alpha=0.6, label='Normal')
                if np.any(anomaly_mask):
                    self.ax_embedding.scatter(embeddings[anomaly_mask, 0], embeddings[anomaly_mask, 1], 
                                            c='red', s=50, alpha=0.8, label='Anomaly')
                
                if np.any(anomaly_mask):
                    self.ax_embedding.legend()
            
            # Update sensor readings bar chart
            self.ax_sensors.clear()
            self.ax_sensors.set_title('Current Sensor Readings')
            sensor_names = ['Temp', 'Humid', 'Press', 'CO2', 'Light', 'Vibr'][:self.n_sensors]
            values = list(reading['values_array'])[:self.n_sensors]
            
            colors = ['red' if reading.get('has_anomaly', False) else 'steelblue' 
                     for _ in range(len(values))]
            self.ax_sensors.bar(sensor_names, values, color=colors)
            self.ax_sensors.set_ylabel('Value')
            
            # Update performance metrics
            self.ax_performance.clear()
            self.ax_performance.set_title('Performance Metrics')
            
            avg_update_time = np.mean(self.update_times) if self.update_times else 0
            throughput = self.total_points / (time.time() - self.start_time) if self.total_points > 0 else 0
            
            metrics_text = f"""
Points Processed: {self.total_points}
Avg Update Time: {avg_update_time:.1f} ms
Throughput: {throughput:.1f} points/sec
Memory Usage: {len(self.embedding_history)} points
Anomalies Detected: {sum(self.anomaly_history)}
Current Status: {'🚨 ANOMALY' if reading.get('has_anomaly', False) else '✅ Normal'}
            """.strip()
            
            self.ax_performance.text(0.05, 0.95, metrics_text, transform=self.ax_performance.transAxes, 
                                   fontsize=11, verticalalignment='top', fontfamily='monospace')
            self.ax_performance.set_xticks([])
            self.ax_performance.set_yticks([])
            
            # Update timeline
            if len(self.sensor_history) > 1:
                times = list(self.time_history)
                temps = [r.get('temperature', 0) for r in self.sensor_history]
                
                self.ax_timeline.clear()
                self.ax_timeline.set_title('Temperature Timeline')
                self.ax_timeline.plot(times, temps, 'b-', alpha=0.7)
                self.ax_timeline.set_xlabel('Time (s)')
                self.ax_timeline.set_ylabel('Temperature (°C)')
                self.ax_timeline.grid(True, alpha=0.3)
                
                # Highlight anomalies
                anomaly_times = [t for i, t in enumerate(times) if self.anomaly_history[i]]
                anomaly_temps = [temps[i] for i, is_anom in enumerate(self.anomaly_history) if is_anom]
                if anomaly_times:
                    self.ax_timeline.scatter(anomaly_times, anomaly_temps, c='red', s=50, zorder=5)
            
        except Exception as e:
            print(f"Error in visualization update: {e}")
            
    def inject_anomaly(self, anomaly_type='spike', duration=50):
        """Inject an anomaly for demonstration."""
        self.simulator.inject_anomaly(anomaly_type, duration=duration)
        print(f"🚨 Injected {anomaly_type} anomaly for {duration} time steps")
        
    def run(self):
        """Start the live demo."""
        print("🚀 Starting Live Sensor Demo...")
        print("📊 Real-time NormalizedDynamics embedding of IoT sensor data")
        print("🔄 Press Ctrl+C to stop\n")
        
        # Create animation
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_visualization, 
            interval=self.update_interval,
            blit=False,
            save_count=1000
        )
        
        # Show plot
        plt.show()
        

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Live Sensor Demo for NormalizedDynamics')
    parser.add_argument('--sensors', type=int, default=6, help='Number of sensors to simulate')
    parser.add_argument('--window', type=int, default=200, help='Number of points to display')
    parser.add_argument('--interval', type=int, default=100, help='Update interval in milliseconds')
    parser.add_argument('--anomaly', type=str, choices=['spike', 'drift', 'failure'], 
                       help='Inject anomaly after 5 seconds')
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = LiveSensorDemo(
        n_sensors=args.sensors,
        window_size=args.window,
        update_interval=args.interval
    )
    
    # Schedule anomaly injection if requested
    if args.anomaly:
        import threading
        def delayed_anomaly():
            time.sleep(5)  # Wait 5 seconds
            demo.inject_anomaly(args.anomaly)
        
        thread = threading.Thread(target=delayed_anomaly)
        thread.daemon = True
        thread.start()
    
    try:
        demo.run()
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")


if __name__ == "__main__":
    main() 