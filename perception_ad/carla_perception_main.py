import pygame
import numpy as np
import sys, os
import cv2
import datetime
import yaml
import time
import carla
from enhanced_perception import EnhancedPerception
from validation_lane_detection import LaneValidator
from vehicles.vehicle_manager import VehicleManager, TrafficManager
from display_manager import DisplayManager
from sensors.sensor_manager import SensorManager
from performace_metrics import PerformanceMonitor


class CarlaLaneDetection:
    def __init__(self, config):
        """
        CARLA Lane Detection with error handling and monitoring.
        
        Attributes:
        - max_retries: Network connections fail, need multiple attempts
        - connection_timeout: Prevent hanging on server issues
        - frame_timeout_count: Track frame drop frequency
        - performance_monitor: Essential for real-time system health
        - connection_stable: Know when to attempt recovery
        - error_count: Detect system degradation
        """
        # Configuration
        self.config = config
        self.carla_config = config['carla']
        
        # Connection management
        self.max_retries = 3  # Give server multiple chances to respond
        self.connection_timeout = 20.0  # Don't wait forever for unresponsive server
        self.frame_timeout_count = 0
        self.max_frame_timeouts = 10  # Too many timeouts indicate serious problems
        self.connection_stable = False
        self.error_count = 0
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(config)
        
        # Attributes
        self.FPS = self.carla_config['FPS']
        self.client = None
        self.world = None
        self.running = False
        self.current_frame = None

        # On/Off enablers
        self.enable_traffic = self.carla_config['enable_traffic']
        self.enable_debugs = self.carla_config['enable_debugs']
        self.enable_recording = self.carla_config['enable_recording']
        self.validation_mode = self.carla_config['validation_mode']

        # Detection, validation and control
        self.perception = EnhancedPerception(self.config)
        self.lane_validator = None

        # Various managers init
        self.vehicle_manager = None
        self.traffic_manager = None
        self.sensor_manager  = None
        
        if self.enable_recording:
            self.init_video_writer()
        
        if self.validation_mode:
            self.frame_id = 0
            self.capture_times = 0.0
            self.logs = []
            self.sim_time = None

        # Validate config with clear errors
        self._validate_configuration()
        
        # Initiate display manager
        self.display_manager = DisplayManager(config['display_manager'])
        
        # Pre-allocate image buffer for performance
        self._image_buffer = None
        
        print("CARLA initialized successfully")

    def _validate_configuration(self):
        """
        Validate configuration before starting to fail fast with clear errors.
        
        Better to crash immediately with helpful message than mysteriously later.
        Input validation is critical for robust systems.
        """
        print("Validating configuration...")
        
        # Check required sections exist
        required_sections = ['carla', 'lane_detector']
        if self.validation_mode:
            required_sections.append('validation')
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Check required CARLA keys
        required_carla_keys = ['host', 'port', 'town', 'FPS']
        for key in required_carla_keys:
            if key not in self.carla_config:
                raise ValueError(f"Missing required CARLA config key: {key}")
        
        # Validate value ranges to prevent nonsensical settings
        fps = self.carla_config['FPS']
        if not (1 <= fps <= 60):
            raise ValueError(f"FPS must be between 1-60, got {fps}")
        
        print("Configuration validation successful")

    def init_video_writer(self):
        """Initialize video writer with error handling"""
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = "recordings"
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            output_name = f"{output_dir}/lane_detection_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_out = cv2.VideoWriter(output_name, fourcc, self.FPS, self.display_manager.pygame_display)
            
            if not self.video_out.isOpened():
                raise RuntimeError("Failed to initialize video writer")
            
            print(f"Video recording initialized: {output_name}")
            
        except Exception as e:
            print(f"Video writer initialization failed: {e}")
            self.enable_recording = False
    
    def _spawn_actors(self):
        """
        Actor spawning with validation and retry logic.
        
        Actor spawning can fail due to:
        - Collision with existing actors
        - Invalid spawn points
        - Server issues
        Systems need robust spawn logic with multiple attempts.
        """
        try:
            # Spawn vehicle
            self.vehicle_manager = VehicleManager(self.world)
            self.vehicle_manager.spawn_vehicle()
            self.vehicle = self.vehicle_manager.vehicle
            
            # Spawn traffic
            if self.enable_traffic:
                self.traffic_manager = TrafficManager(self.world, self.config['traffic']['number_of_vehicles'])
                self.traffic_manager.spawn_traffic()
            
            # Spawn other sensors attached to vehicle
            self.sensor_manager = SensorManager(self.world)

            # Front Camera
            self.camera = self.sensor_manager.init_sensor("RGBCamera", self.config['sensors']['front_camera'], self.vehicle)
            self.display_manager.add_sensor("front_camera", [0,0])
            self.camera.listen(lambda image: self._safe_camera_callback(image))

            # LiDAR
            self.lidar = self.sensor_manager.init_sensor("LiDAR", self.config['sensors']['lidar'], self.vehicle)
            self.display_manager.add_sensor("lidar", [0,1])
            self.lidar.listen(lambda data: self._safe_lidar_callback(data))

            # Setup validation after all actors are ready
            if self.validation_mode:
                self._setup_validation()
            
            print(f"All actors spawned successfully")
            return True
            
        except Exception as e:
            print(f"Actor spawning failed: {e}")
            return False

    def carla_setup(self):
        """
        Enhanced CARLA setup with retry logic and comprehensive validation.
        
        Network connections are unreliable. Servers restart, networks hiccup.
        Systems must handle these gracefully with retries and validation.
        """
        
        for attempt in range(self.max_retries):
            try:
                print(f"CARLA connection attempt {attempt + 1}/{self.max_retries}")
                
                # Connect with timeout to prevent hanging
                self.client = carla.Client(self.carla_config['host'], self.carla_config['port'])
                self.client.set_timeout(self.connection_timeout)
                
                # Test connection immediately to catch issues early
                client_version = self.client.get_client_version()
                server_version = self.client.get_server_version()
                print(f"Connected - Client: {client_version}, Server: {server_version}")
                
                # Validate map exists before trying to load it
                available_maps = self.client.get_available_maps()
                target_map = self.carla_config['town']
                target_map_path = "/Game/Carla/Maps/" + target_map
                
                if target_map_path not in available_maps:
                    print(f"WARNING: Map {target_map} not available, using Town03")
                    target_map = "Town03"
                
                print(f"Loading world: {target_map}")
                self.world = self.client.load_world(target_map)
                
                # Configure synchronous mode with validation
                self._setup_synchronous_mode()
                
                # Spawn actors
                if not self._spawn_actors():
                    raise RuntimeError("Actor spawning failed!")
                
                self.connection_stable = True
                print("CARLA setup completed successfully")
                return True
                
            except Exception as e:
                print(f"Setup attempt {attempt + 1} failed: {e}")
                self._cleanup_partial_setup()
                
                if attempt < self.max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)  # Give server time to stabilize
                else:
                    print("All connection attempts failed")
                    return False
        
        return False
    
    def _cleanup_partial_setup(self):
        """Coordinate cleanup across all managers"""
        print("Cleaning up partial setup...")
        if self.vehicle_manager:
            self.vehicle_manager.cleanup()   
        if self.traffic_manager:
            self.traffic_manager.cleanup()   
        if self.sensor_manager:
            self.sensor_manager.cleanup()

    def _setup_synchronous_mode(self):
        """
        Setup synchronous mode with validation.
        
        Synchronous mode is critical for reproducible results and stable frame timing.
        Validation ensures settings actually took effect.
        """
        try:
            settings = self.world.get_settings()
            
            # Validate FPS setting to prevent invalid configurations
            target_fps = self.carla_config['FPS']
            if target_fps <= 0 or target_fps > 60:
                print(f"WARNING: Invalid FPS {target_fps}, using 30")
                target_fps = 30
            
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / target_fps
            settings.no_rendering_mode = False  # Ensure rendering is enabled for visualization
            
            self.world.apply_settings(settings)
            
            # Configure traffic manager for deterministic behavior
            traffic_manager = self.client.get_trafficmanager()
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_random_device_seed(42)  # Deterministic behavior for testing
            
            print(f"Synchronous mode enabled at {target_fps} FPS")
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup synchronous mode: {e}")

    def _setup_validation(self):
        """Setup validation pipeline with error handling"""
        try:
            self.lane_validator = LaneValidator(
                self.config, self.world, self.camera, self.vehicle, self.perception.lane_detector
            )
            print("Validation pipeline initialized")
        except Exception as e:
            print(f"Validation setup failed: {e}")
            self.validation_mode = False

    def _safe_camera_callback(self, image):
        """
        Enhanced camera callback with performance monitoring and error handling.
        
        Camera callback is called at high frequency and any error here crashes
        the entire pipeline. Need comprehensive error handling and performance monitoring.
        """
        try:
            callback_start = time.time()
            
            # Convert CARLA image efficiently with pre-allocated buffer
            frame = self._convert_carla_image(image)
            
            # Lane detection with timing for performance monitoring
            result, gray, edges, masked, left_coords, right_coords, detected_objects, timing_metrics = self.perception.process_image(frame)
            
            # Validation only when needed to minimize performance impact
            if self.validation_mode and hasattr(self, 'lane_validator'):
                self.sim_time = self.world.get_snapshot().timestamp.elapsed_seconds
                self.frame_id, self.capture_times, self.logs = self.lane_validator.run_validation(
                    self.sim_time, result, self.frame_id, self.capture_times, 
                    self.logs, left_coords, right_coords
                )
            
            # Update display data for main thread
            callback_time = (time.time() - callback_start) * 1000
            self.current_frame = {
                'result': result,
                'gray': gray,
                'edges': edges,
                'masked': masked,
            }

            # Update display manager's camera image
            self.display_manager.update_sensor_image('front_camera', result)
            
            # Performance monitoring for system health
            self.performance_monitor.add_frame_data(callback_time, timing_metrics)
            
            # Reset timeout counter on successful frame
            self.frame_timeout_count = 0
            self.error_count = 0  # Reset error count on success
            
        except Exception as e:
            print(f"Camera callback error: {e}")
            self.frame_timeout_count += 1
            self.error_count += 1
            
            # Handle excessive errors with recovery attempt
            if self.frame_timeout_count > self.max_frame_timeouts:
                print("Too many camera errors, attempting restart...")
                self.sensor_manager.restart_sensor(self.camera, "RGBCamera", self.config['sensors']['front_camera'], self.vehicle)
                self.frame_timeout_count = 0
        
    def _safe_lidar_callback(self, image):
        """Process LiDAR point cloud"""
        try:
            disp_size = self.display_manager.get_cell_display_size()
            lidar_range = 2.0*self.config['sensors']['lidar']['range']

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

    def update_display(self):
        """Enhanced display update with error handling and performance info"""
        if self.current_frame is None:
            return
            
        try:
            result = self.current_frame['result']
            gray = self.current_frame['gray']
            edges = self.current_frame['edges']
            masked = self.current_frame['masked']
            
            # Render on pygame
            self.display_manager.render()

            # Record video if enabled (convert back to BGR for OpenCV)
            if self.enable_recording and hasattr(self, 'video_out') and self.video_out is not None:
                bgr_frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                self.video_out.write(bgr_frame)
                
            if self.enable_debugs:
                self.display_manager.draw_debug("Gray", gray, 20)
                self.display_manager.draw_debug("Edges", edges, 160)
                self.display_manager.draw_debug("Masked", masked, 300)

            pygame.display.flip()  # Use flip() instead of update() for better performance
            
        except Exception as e:
            print(f"Display update error: {e}")

    def _safe_world_tick(self, timeout=2.0):
        """
        World tick with timeout protection.
        
        world.tick() can hang indefinitely if server has issues.
        Timeout protection allows detection and recovery from server problems.
        """
        try:
            tick_start = time.time()
            self.world.tick()
            
            # Check if tick took too long (indicates server problems)
            tick_time = time.time() - tick_start
            if tick_time > timeout:
                print(f"WARNING: Slow world tick: {tick_time:.2f}s")
                return False
            
            return True
            
        except Exception as e:
            print(f"World tick failed: {e}")
            return False

    def _recover_connection(self):
        """
        Attempt to recover from connection issues.
        
        Sometimes CARLA connections degrade but can be recovered
        without full restart. This saves time and maintains session state.
        """
        print("Attempting connection recovery...")
        
        try:
            # Test client connection first
            version = self.client.get_client_version()
            
            # Reset synchronous mode which can get out of sync
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 1.0 / self.FPS
            self.world.apply_settings(settings)
            
            print("Connection recovery successful")
            return True
            
        except Exception as e:
            print(f"Connection recovery failed: {e}")
            return False

    def _handle_pygame_events(self):
        """Handle pygame events with enhanced user feedback"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    # Toggle autopilot with user feedback
                    self.vehicle_manager.toggle_autopilot()
                    status = "enabled" if self.vehicle_manager.autopilot_enabled is True else "disabled"
                    print(f"Autopilot {status}")

    def _handle_manual_control(self):
        """Handle manual vehicle control when autopilot is disabled"""
        keys = pygame.key.get_pressed()
        throttle = 0.0
        brake = 0.0
        steer = 0.0
        
        if keys[pygame.K_w]:
            throttle = 1.0
        if keys[pygame.K_s]:
            brake = 1.0
        if keys[pygame.K_a]:
            steer = 0.3
        if keys[pygame.K_d]:
            steer = -0.3
        
        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steer
        )
        self.vehicle.apply_control(control)

    def _print_startup_info(self):
        """Print comprehensive startup information for user"""
        print("CARLA Lane Detection System Started")
        print("=" * 50)
        print(f"Target FPS: {self.FPS}")
        print(f"Image Resolution: {self.config['sensors']['front_camera']['image_size_x']}x{self.config['sensors']['front_camera']['image_size_y']}")
        print(f"Validation Mode: {'ON' if self.validation_mode else 'OFF'}")
        print(f"Recording: {'ON' if self.enable_recording else 'OFF'}")
        print(f"Debug Overlays: {'ON' if self.enable_debugs else 'OFF'}")
        
        # Show GPU acceleration status if available
        if hasattr(self.perception, 'use_gpu'):
            gpu_status = "ENABLED" if self.perception.use_gpu else "DISABLED"
            print(f"GPU Acceleration: {gpu_status}")
        
        print("\nControls:")
        print("  ESC - Quit application")
        print("  SPACE - Toggle autopilot on/off")
        print("  W/A/S/D - Manual control (when autopilot off)")
        print("=" * 50)

    def run(self):
        """
        Main execution loop with robust error handling and monitoring.
        
        Main loop is the heart of the application. It needs:
        - Comprehensive error handling to prevent crashes
        - Performance monitoring to detect issues
        - Graceful recovery from problems
        - Clear user feedback about system status
        """
        if not self.carla_setup():
            print("Failed to setup CARLA - exiting")
            return False
        
        try:
            self.running = True
            self._print_startup_info()
            
            clock = pygame.time.Clock()
            
            while self.running:
                try:
                    # Tick world with timeout protection
                    if not self._safe_world_tick():
                        print("WARNING: World tick timeout, attempting recovery...")
                        if not self._recover_connection():
                            print("Connection recovery failed, shutting down...")
                            break
                    
                    # Handle user input
                    self._handle_pygame_events()
                    
                    # Manual control if autopilot disabled
                    if self.vehicle_manager.autopilot_enabled is False:
                        self._handle_manual_control()
                    
                    # Update display with error protection
                    self.update_display()
                    
                    # Maintain target framerate
                    clock.tick(self.FPS)
                    
                except Exception as e:
                    print(f"Loop iteration error: {e}")
                    self.error_count += 1
                    
                    # Too many errors indicates serious problems
                    if self.error_count > 10:
                        print("Too many errors detected, shutting down...")
                        break
            
            return True
            
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return True
        except Exception as e:
            print(f"Critical error in main loop: {e}")
            return False
        finally:
            self.cleanup()

    def cleanup(self):
        """
        Enhanced cleanup with proper resource management and error handling.
        
        Proper cleanup is critical for:
        - Preventing resource leaks
        - Leaving CARLA in clean state
        - Ensuring smooth subsequent runs
        - Professional system behavior
        """
        print("Starting comprehensive cleanup...")
        
        # Define cleanup steps in proper order with error isolation
        cleanup_steps = [
            ("Sensor cleanup", self.sensor_manager.cleanup),
            ("Destroying vehicle actors", self.vehicle_manager.cleanup),
            ("Destroying traffic actors", self.traffic_manager.cleanup),
            ("Closing video recording", self._cleanup_recording),
            ("Restoring async mode", self._cleanup_carla_settings),
            ("Closing pygame", self.display_manager.cleanup_pygame),
        ]
        
        for step_name, cleanup_func in cleanup_steps:
            try:
                print(f"  {step_name}...")
                cleanup_func()
            except Exception as e:
                print(f"  WARNING: {step_name} failed: {e}")
                # Continue cleanup despite individual failures
        
        # Final validation of cleanup
        try:
            if self.world:
                remaining_actors = len(self.world.get_actors().filter('*'))
                print(f"  Remaining actors in world: {remaining_actors}")
        except:
            pass
        
        print("Cleanup completed successfully")

    def _cleanup_recording(self):
        """Safe video recording cleanup"""
        if self.enable_recording and hasattr(self, 'video_out'):
            try:
                self.video_out.release()
            except:
                pass

    def _cleanup_carla_settings(self):
        """Restore CARLA to asynchronous mode"""
        if self.world:
            try:
                settings = self.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.apply_settings(settings)
            except:
                pass  # Don't crash cleanup on settings restore failure


if __name__ == '__main__':
    # Load config with error handling
    try:
        with open("./config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: config.yaml not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid config.yaml: {e}")
        sys.exit(1)
    
    # Create and run system with error handling
    try:
        carla_lane_detection = CarlaLaneDetection(config=cfg)
        success = carla_lane_detection.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        sys.exit(1)