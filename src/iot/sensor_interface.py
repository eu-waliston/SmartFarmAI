# src/iot/sensor_interface.py
import random
import time
from datetime import datetime
import json
import torch

class VirtualSensor:
    def __init__(self, sensor_type):
        self.sensor_type = sensor_type
        self.base_values = {
            'temperature': 25.0,
            'humidity': 60.0,
            'soil_ph': 6.5,
            'light_intensity': 50000
        }

    def read_data(self):
        base = self.base_values[self.sensor_type]

        if self.sensor_type == 'temperature':
            value = base + random.uniform(-5, 5)
        elif self.sensor_type == 'humidity':
            value = base + random.uniform(-20, 20)
        elif self.sensor_type == 'soil_ph':
            value = base + random.uniform(-0.5, 0.5)
        else:  # light_intensity
            value = base + random.uniform(-10000, 10000)

        return {
            'sensor_type': self.sensor_type,
            'value': max(0, value),  # Evitar valores negativos
            'timestamp': datetime.now().isoformat(),
            'unit': self._get_unit()
        }

    def _get_unit(self):
        units = {
            'temperature': '°C',
            'humidity': '%',
            'soil_ph': 'pH',
            'light_intensity': 'lux'
        }
        return units[self.sensor_type]

class SensorNetwork:
    def __init__(self):
        self.sensors = [
            VirtualSensor('temperature'),
            VirtualSensor('humidity'),
            VirtualSensor('soil_ph'),
            VirtualSensor('light_intensity')
        ]

    def collect_all_data(self):
        data = {}
        for sensor in self.sensors:
            sensor_data = sensor.read_data()
            data[sensor.sensor_type] = sensor_data
        return data

    def get_sensor_tensor(self):
        data = self.collect_all_data()
        values = [data['temperature']['value'],
                 data['humidity']['value'],
                 data['soil_ph']['value'],
                 data['light_intensity']['value']]
        return torch.tensor(values, dtype=torch.float32).unsqueeze(0)