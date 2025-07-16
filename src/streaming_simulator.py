import numpy as np
import time
from datetime import datetime
import json

class StreamingSensorSimulator:
    """
    Lightweight sensor data simulator for real-time streaming demo.
    Generates realistic IoT sensor patterns with temporal correlations.
    """
    
    def __init__(self, n_sensors=6, update_interval=0.1):
        self.n_sensors = n_sensors
        self.update_interval = update_interval
        self.time_step = 0
        self.start_time = time.time()
        
        # Sensor configuration with realistic ranges and patterns
        self.sensor_config = {
            'temperature': {
                'base': 22.0,
                'amplitude': 5.0,
                'frequency': 0.02,  # Slow daily cycle
                'noise': 0.3,
                'unit': '°C'
            },
            'humidity': {
                'base': 50.0,
                'amplitude': 15.0,
                'frequency': 0.025,  # Slightly offset from temperature
                'noise': 2.0,
                'unit': '%'
            },
            'pressure': {
                'base': 1013.25,
                'amplitude': 8.0,
                'frequency': 0.01,  # Weather patterns
                'noise': 0.5,
                'unit': 'hPa'
            },
            'co2': {
                'base': 400.0,
                'amplitude': 200.0,
                'frequency': 0.05,  # Activity cycles
                'noise': 10.0,
                'unit': 'ppm'
            },
            'light': {
                'base': 300.0,
                'amplitude': 400.0,
                'frequency': 0.08,  # Day/night cycles
                'noise': 15.0,
                'unit': 'lux'
            },
            'vibration': {
                'base': 2.0,
                'amplitude': 1.5,
                'frequency': 0.3,  # Equipment cycles
                'noise': 0.2,
                'unit': 'm/s²'
            }
        }
        
        # Track sensor readings for correlations
        self.last_temperature = self.sensor_config['temperature']['base']
        
        # Anomaly injection system
        self.active_anomalies = {}
        self.anomaly_history = []
        self.anomaly_counter = 0
        
    def generate_sensor_reading(self):
        """Generate a single timestamped sensor reading with realistic patterns"""
        current_time = time.time() - self.start_time
        
        reading = {}
        sensor_values = []
        
        for i, (sensor_name, config) in enumerate(self.sensor_config.items()):
            if i >= self.n_sensors:
                break
                
            # Base pattern: sinusoidal with noise
            base_value = config['base']
            pattern = config['amplitude'] * np.sin(current_time * config['frequency'] + i)
            noise = np.random.normal(0, config['noise'])
            
            # Add correlations for realism
            if sensor_name == 'humidity':
                # Humidity inversely correlated with temperature
                temp_effect = -0.5 * (self.last_temperature - self.sensor_config['temperature']['base'])
                pattern += temp_effect
            elif sensor_name == 'co2':
                # CO2 has daily activity patterns
                activity_boost = 100 * np.sin(current_time * 0.1) ** 2
                pattern += activity_boost
            elif sensor_name == 'light':
                # Light follows day cycle (more pronounced)
                day_cycle = np.maximum(0, np.sin(current_time * 0.04))
                pattern = config['amplitude'] * day_cycle
                
            value = base_value + pattern + noise
            
            # Apply active anomalies
            anomaly_applied = False
            for anomaly_id, anomaly in self.active_anomalies.items():
                if not anomaly['active']:
                    continue
                    
                # Check if anomaly should still be active
                if self.time_step - anomaly['start_time'] > anomaly['duration']:
                    anomaly['active'] = False
                    continue
                    
                # Apply anomaly if this sensor is affected
                if sensor_name in anomaly['sensors']:
                    anomaly_applied = True
                    intensity = anomaly['intensity']
                    
                    if anomaly['type'] == 'spike':
                        # Sudden spike in readings
                        spike_factor = 2.0 + intensity
                        value = base_value + (pattern * spike_factor) + (noise * 3)
                        
                    elif anomaly['type'] == 'drift':
                        # Gradual drift over time
                        drift_progress = (self.time_step - anomaly['start_time']) / anomaly['duration']
                        drift_amount = config['amplitude'] * intensity * drift_progress
                        value = base_value + pattern + drift_amount + noise
                        
                    elif anomaly['type'] == 'failure':
                        # Sensor failure - stuck readings
                        value = config['base']  # Stuck at baseline
                        
                    elif anomaly['type'] == 'fire_alarm':
                        # Simulate fire conditions
                        if sensor_name == 'temperature':
                            value = 45 + 20 * intensity + noise  # High temperature
                        elif sensor_name == 'co2':
                            value = 800 + 400 * intensity + noise  # High CO2
                        elif sensor_name == 'light':
                            value = 50 + noise  # Low light (smoke)
                            
                    elif anomaly['type'] == 'equipment_failure':
                        # Equipment malfunction
                        if sensor_name == 'vibration':
                            value = 5 + 3 * intensity + noise * 2  # High vibration
                        elif sensor_name == 'temperature':
                            value = 35 + 10 * intensity + noise  # Overheating
                            
                    elif anomaly['type'] == 'network_interference':
                        # Network interference - noisy/erratic readings
                        interference_noise = np.random.normal(0, config['noise'] * 5 * intensity)
                        value = base_value + pattern + interference_noise
            
            # Ensure realistic bounds
            if sensor_name == 'humidity':
                value = np.clip(value, 0, 100)
            elif sensor_name == 'pressure':
                value = np.clip(value, 980, 1040)
            elif sensor_name == 'co2':
                value = np.clip(value, 300, 1500)
            elif sensor_name == 'light':
                value = np.clip(value, 0, 1000)
            elif sensor_name == 'vibration':
                value = np.clip(value, 0, 10)
                
            reading[sensor_name] = round(value, 2)
            sensor_values.append(value)
            
            if sensor_name == 'temperature':
                self.last_temperature = value
        
        # Add metadata
        reading['timestamp'] = current_time
        reading['values_array'] = sensor_values[:self.n_sensors]
        reading['anomaly_status'] = self.get_anomaly_status()
        reading['has_anomaly'] = len([a for a in self.active_anomalies.values() if a['active']]) > 0
        
        self.time_step += 1
        return reading
    
    def stream(self):
        """Generator that yields sensor readings at specified interval"""
        while True:
            reading = self.generate_sensor_reading()
            yield reading
            time.sleep(self.update_interval)
    
    def inject_anomaly(self, anomaly_type, sensor_name=None, duration=50, intensity=1.0):
        """
        Inject various types of anomalies for testing anomaly detection
        
        Args:
            anomaly_type: 'spike', 'drift', 'failure', 'fire_alarm', 'equipment_failure', 'network_interference'
            sensor_name: Specific sensor or None for auto-selection
            duration: How long the anomaly lasts (in time steps)
            intensity: Multiplier for anomaly strength (1.0 = normal, 2.0 = 2x stronger)
        """
        self.anomaly_counter += 1
        anomaly_id = f"anomaly_{self.anomaly_counter}"
        
        anomaly = {
            'id': anomaly_id,
            'type': anomaly_type,
            'sensor': sensor_name,
            'start_time': self.time_step,
            'duration': duration,
            'intensity': intensity,
            'active': True
        }
        
        # Auto-select sensors based on anomaly type
        if sensor_name is None:
            if anomaly_type == 'fire_alarm':
                anomaly['sensors'] = ['temperature', 'co2', 'light']
            elif anomaly_type == 'equipment_failure':
                anomaly['sensors'] = ['vibration', 'temperature']
            elif anomaly_type == 'network_interference':
                anomaly['sensors'] = list(self.sensor_config.keys())[:3]  # Affect first 3 sensors
            else:
                anomaly['sensors'] = [list(self.sensor_config.keys())[0]]  # Default to first sensor
        else:
            anomaly['sensors'] = [sensor_name]
        
        self.active_anomalies[anomaly_id] = anomaly
        self.anomaly_history.append(anomaly.copy())
        
        return anomaly_id
    
    def clear_anomalies(self):
        """Clear all active anomalies"""
        self.active_anomalies.clear()
    
    def get_anomaly_status(self):
        """Get current anomaly status"""
        active_count = len([a for a in self.active_anomalies.values() if a['active']])
        return {
            'active_anomalies': active_count,
            'total_injected': len(self.anomaly_history),
            'current_anomalies': list(self.active_anomalies.keys())
        } 