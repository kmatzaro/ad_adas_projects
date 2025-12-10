from typing import Dict
import carla
from actors.actor_utils import cleanup_sensors
import time
import numpy as np
import cv2
from performance.performance_metrics import PerformanceMonitor


class SensorManager:
    def __init__(self, world, display_manager, perception_module, config: Dict):
        self.world = world
        self.sensors = []
        self.blueprint_library = self.world.get_blueprint_library()
        self.display_manager = display_manager
        self.perception_module = perception_module
        self.sensor_config = config["sensors"]

        # Pre-allocate image buffer for performance
        self._image_buffer = None

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(config)

        self.validation_mode = config['carla']['validation_mode']
        if self.validation_mode:
            self.frame_id = 0
            self.capture_times = 0.0
            self.logs = []
            self.sim_time = None

    def init_sensor(self, sensor_type, attach_to, sensor_id: str):
        if sensor_type == 'RGBCamera':
            self._validate_camera_configuration(self.sensor_config[sensor_id])

        elif sensor_type == 'LiDAR':
            self._validate_lidar_configuration(self.sensor_config[sensor_id])
        
        sensor = self._spawn_sensor(sensor_type, attach_to, sensor_id)
        return sensor

    def _spawn_sensor(self, sensor_type: str, attach_to, sensor_id: str):
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
            
            for key in self.sensor_config[sensor_id]:
                if key != 'transform':
                    if key == 'sensor_tick':
                        sensor_bp.set_attribute(key, str(1.0/self.sensor_config[sensor_id][key]))
                    else:
                        sensor_bp.set_attribute(key, str(self.sensor_config[sensor_id][key]))
            
            # Attach camera to vehicle with proper transform
            sensor_transform = carla.Transform(
                carla.Location(**self.sensor_config[sensor_id]['transform']['location']),
                carla.Rotation(**self.sensor_config[sensor_id]['transform']['rotation']))
            
            sensor = self.world.spawn_actor(
                sensor_bp, 
                sensor_transform, 
                attach_to=attach_to)
            self.sensors.append(sensor)
            
            print(f"Sensor {sensor_type} configured and attached successfully")

            return sensor
            
        except Exception as e:
            raise RuntimeError(f"{sensor_type} setup failed: {e}")
        
    def restart_sensor(self, sensor, sensor_type, attach_to):
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
            self.init_sensor(sensor_type, self.sensor_config, attach_to)
            
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

    def camera_callback(self, image, camera_id: str):
        """
        Enhanced camera callback with performance monitoring and error handling.
        
        Camera callback is called at high frequency and any error here crashes
        the entire pipeline. Need comprehensive error handling and performance monitoring.
        """
        try:
            timestamp = self.world.get_snapshot().timestamp
            
            # Convert CARLA image efficiently with pre-allocated buffer and resize
            frame = self._convert_carla_image(image)
            frame = self.perception_module.lane_detector._preprocess_image(frame)
            
            # Lane detection with timing for performance monitoring
            if camera_id == 'front_camera':
                processed_frame = self.perception_module.combined_camera_process(frame, camera_id, timestamp)

                # Update debug images for the overlays
                self.display_manager.update_debug_images(processed_frame.debug_images)

            elif camera_id in ["left_camera", "right_camera", "rear_camera"]:
                processed_frame = self.perception_module.process_object_detection_camera(frame, camera_id, timestamp)
            
            # Validation only when needed to minimize performance impact
            # if self.validation_mode and hasattr(self, 'lane_validator'):
            #     self.frame_id, self.capture_times, self.logs = self.lane_validator.run_validation(
            #         timestamp, result, self.frame_id, self.capture_times, 
            #         self.logs, left_coords, right_coords
            #     )
            
            # Update display manager's camera image
            self.display_manager.update_sensor_image(camera_id, processed_frame.processed_image)
            
            # Performance monitoring for system health
            self.performance_monitor.add_frame_data(camera_id, processed_frame.timing_metrics)
            
        except Exception as e:
            print(f"Camera callback error: {e}")
        
    def lidar_callback(self, image):
        """Process LiDAR point cloud"""
        try:
            disp_size = self.display_manager.get_cell_display_size()
            lidar_range = 2.0*self.sensor_config['lidar']['range']

            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(disp_size) / lidar_range
            lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (disp_size[1], disp_size[0], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

            lidar_img[tuple(lidar_data.T[::-1])] = (255, 255, 255)

            self.display_manager.update_sensor_image('lidar', lidar_img)
        
        except Exception as e:
            print(f"Lidar callback error: {e}")

    def _convert_carla_image(self, image):
        """
        Optimized CARLA image conversion with pre-allocated buffer.
        
        Image conversion happens every frame at high frequency.
        Pre-allocating buffer avoids memory allocation overhead.
        """
        # Pre-allocate buffer for performance (avoid allocation every frame)
        if (self._image_buffer is None or 
            self._image_buffer.shape != (image.height, image.width, 4)):
            self._image_buffer = np.empty((image.height, image.width, 4), dtype=np.uint8)
        
        # Direct reshape is faster than multiple copies
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape(self._image_buffer.shape)
        
        # Convert BGR to RGB and remove alpha channel
        frame = cv2.cvtColor(array[:, :, :3], cv2.COLOR_BGR2RGB)
        
        return frame
    
    def cleanup(self):
        """Cleanup all sensors"""
        if self.sensors:
            cleanup_sensors(self.sensors)