from typing import Dict
import carla
from actor_utils import cleanup_sensors
import time

class SensorManager:
    def __init__(self, world):
        self.world = world
        self.sensors = []
        self.blueprint_library = self.world.get_blueprint_library()

    def init_sensor(self, sensor_type, sensor_config, attach_to):
        if sensor_type == 'RGBCamera':
            self._validate_camera_configuration(sensor_config)

        elif sensor_type == 'LiDAR':
            self._validate_lidar_configuration(sensor_config)
        
        sensor = self._spawn_sensor(sensor_type, sensor_config, attach_to)
        return sensor

    def _spawn_sensor(self, sensor_type: str, sensor_config: Dict, attach_to):
        """
        Spawn sensor with enhanced configuration and validation.
        
        Sensor is critical component - needs robust setup with proper configuration
        and validation to ensure quality and performance.
        """
        try:
            if sensor_type == 'RGBCamera':
                sensor_bp = self.blueprint_library.find('sensor.camera.rgb')
            elif sensor_type == 'LiDAR':
                sensor_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
            
            for key in sensor_config:
                if key != 'transform':
                    sensor_bp.set_attribute(key, str(sensor_config[key]))
            
            # Attach camera to vehicle with proper transform
            sensor_transform = carla.Transform(
                carla.Location(**sensor_config['transform']['location']),
                carla.Rotation(**sensor_config['transform']['rotation']))
            
            sensor = self.world.spawn_actor(
                sensor_bp, 
                sensor_transform, 
                attach_to=attach_to)
            self.sensors.append(sensor)
            
            print(f"Sensor {sensor_type} configured and attached successfully")

            return sensor
            
        except Exception as e:
            raise RuntimeError(f"{sensor_type} setup failed: {e}")
        
    def restart_sensor(self, sensor, sensor_type, sensor_config, attach_to):
        """
        Attempt to restart sensor on repeated failures.
        
        Sometimes sensors get into bad state and need restart.
        This is a recovery mechanism for persistent issues.
        """
        try:
            print(f"Attempting {sensor_type} restart...")
            
            if sensor and sensor.is_alive:
                sensor.stop()
                time.sleep(0.5)
                sensor.destroy()
            
            # Remove from actors list to prevent double cleanup
            if sensor in self.sensors:
                self.sensors.remove(sensor)
            
            # Spawn new camera
            self.init_sensor(sensor_type, sensor_config, attach_to)
            
            print(f"{sensor_type} restart successful")
            
        except Exception as e:
            print(f"{sensor} restart failed: {e}")
    
    def _validate_camera_configuration(self, sensor_config: Dict):
        # Validate camera configuration
        camera_keys = ['image_size_x', 'image_size_y', 'fov', 'transform']
        for key in camera_keys:
            if key not in sensor_config:
                raise ValueError(f"Missing camera config key: {key}")
            if key != 'transform' and sensor_config[key] <= 0:
                raise ValueError(f"Camera config key: {key} must be positive!")

    def _validate_lidar_configuration(self, sensor_config: Dict):
        # Validate camera configuration
        lidar_keys = ['range', 'dropoff_general_rate', 'dropoff_intensity_limit', 'dropoff_zero_intensity', 'transform']
        for key in lidar_keys:
            if key not in sensor_config:
                raise ValueError(f"Missing LiDAR config key: {key}")
            if key != 'transform' and sensor_config[key] < 0:
                raise ValueError(f"LiDAR config key: {key} must be positive or 0!")

    def get_sensors(self):
        return self.sensors

    def cleanup(self):
        """Cleanup all sensors"""
        if self.sensors:
            cleanup_sensors(self.sensors)
