import pygame
import numpy as np
import sys, os
import cv2
# from simple_lane_detection import SimpleLaneDetector
from enhanced_perception import EnhancedPerception
# from gpu_simple_lane_detector import GPUEnhancedLaneDetector as SimpleLaneDetector
from validation_lane_detection import LaneValidator
import random
import datetime
import yaml
import time
from collections import deque
import carla
from dataclasses import dataclass
from typing import Optional, Dict
from vehicles.vehicle_manager import VehicleManager, TrafficManager


@dataclass
class TimingMetrics:
    average_detection_time: float
    min_detection_time: float
    max_detection_time: float
    current_fps_est: int
    real_time_performance: Optional[bool]


class PerformanceMonitor:
    """
    Performance monitoring for CARLA pipeline
    
    In real-time systems, you must monitor performance to detect:
    - Processing bottlenecks
    - Frame drops
    - Memory leaks
    - System degradation over time
    """
    
    def __init__(self, config, window_size=1000):
        # Rolling window prevents memory growth and gives recent performance
        self.window_size = window_size
        self.callback_times = deque(maxlen=window_size)
        self.lane_detection_times = deque(maxlen=window_size)
        self.object_detection_times = deque(maxlen=window_size)
        self.total_perception_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.FPS = config['carla']['FPS']
        self.start_time = time.time()
        
    def add_frame_data(self, callback_time, perception_time: Dict):
        """Add performance data for current frame"""
        self.callback_times.append(callback_time)
        self.lane_detection_times.append(perception_time['lane_detection_time_ms'])
        self.object_detection_times.append(perception_time['object_detection_time_ms'])
        self.total_perception_times.append(perception_time['total_end_time'])
        self.frame_count += 1
        
        # Log every 60 frames = 1 second at 60fps, provides regular feedback
        if self.frame_count % 60 == 0:
            self._log_performance_summary()
       
    def get_performance_stats(self, detection_times) -> TimingMetrics:
        """Statistics for performance tracking"""

        if not detection_times:
            return TimingMetrics(0, 0, 0, 0, False)
        
        detection_times_list = list(detection_times)

        # Average detection time
        avg_detection_time_ms = np.average(detection_times_list)

        # Min/max detection time
        min_detection_time = np.min(detection_times_list)
        max_detection_time = np.max(detection_times_list)

        # FPS estimate
        fps_estimate = 1 / avg_detection_time_ms * 1000

        # Real time performance
        if avg_detection_time_ms <= 1/self.FPS * 1000:
            real_time_performance = True
        else:
            real_time_performance = False

        return TimingMetrics(
            average_detection_time = avg_detection_time_ms,
            min_detection_time = min_detection_time,
            max_detection_time = max_detection_time,
            current_fps_est = int(fps_estimate),
            real_time_performance = real_time_performance
        )

    def _log_performance_summary(self):
        """Log function to print performance stats"""

        timing_components = {
        "Lane Detection": self.lane_detection_times,
        "Object Detection": self.object_detection_times, 
        "Total Perception": self.total_perception_times,
        "Callback Overhead": self.callback_times
        }

        print("=" * 50)
        print(f"Enhanced Perception Performance ({len(self.total_perception_times)} frames):")

        for name, timing_data in timing_components.items():
            if timing_data:
                performance_stats = self.get_performance_stats(timing_data)
                print(f"  {name:18s}: avg={performance_stats.average_detection_time:.1f}ms  ({performance_stats.min_detection_time:.1f}-{performance_stats.max_detection_time:.1f}ms)")
        
        # Real-time status
        if self.total_perception_times:
            real_time_budget = 1000 / self.FPS
            avg_perception = np.mean(self.total_perception_times)
            status = "GOOD" if avg_perception < real_time_budget else " SLOW"
            print(f"  Real-time Status: {status} (target: <{real_time_budget:.1f}ms)")


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
        self.last_frame_time = time.time()
        
        # Attributes
        self.FPS = self.carla_config['FPS']
        self.enable_debugs = self.carla_config['enable_debugs']
        self.client = None
        self.world = None
        self.camera = None
        self.camera_config = self.carla_config['camera']
        self.vehicle = None
        self.vehicle_manager = None
        self.enable_traffic = self.carla_config['traffic']['enable_traffic']
        self.running = False
        self.actors = []
        self.current_frame = None
        self.pygame_display = (
            self.carla_config['pygame_display']['display_width'], 
            self.carla_config['pygame_display']['display_height']
        )

        # Detection, validation and control
        # self.lane_detector = SimpleLaneDetector(self.config)
        self.perception = EnhancedPerception(self.config)
        self.validation_mode = self.carla_config['validation_mode']
        self.lane_validator = None
        
        self.enable_recording = self.carla_config['enable_recording']
        if self.enable_recording:
            self.init_video_writer()
        
        if self.validation_mode:
            self.frame_id = 0
            self.capture_times = 0.0
            self.logs = []
            self.sim_time = None

        # Validate config with clear errors
        self._validate_configuration()
        
        # Pygame and display settings
        pygame.init()
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 16)
        self.display = pygame.display.set_mode(self.pygame_display)
        pygame.display.set_caption("CARLA Synchronous Client")
        
        # Pre-allocate image buffer for performance
        self._image_buffer = None
        
        print("CARLA Lane Detection initialized successfully")

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
        required_carla_keys = ['host', 'port', 'town', 'FPS', 'camera']
        for key in required_carla_keys:
            if key not in self.carla_config:
                raise ValueError(f"Missing required CARLA config key: {key}")
        
        # Validate value ranges to prevent nonsensical settings
        fps = self.carla_config['FPS']
        if not (1 <= fps <= 60):
            raise ValueError(f"FPS must be between 1-60, got {fps}")
        
        # Validate camera configuration
        camera_keys = ['image_width', 'image_height', 'fov', 'transform']
        for key in camera_keys:
            if key not in self.camera_config:
                raise ValueError(f"Missing camera config key: {key}")
        
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
            self.video_out = cv2.VideoWriter(output_name, fourcc, self.FPS, self.pygame_display)
            
            if not self.video_out.isOpened():
                raise RuntimeError("Failed to initialize video writer")
            
            print(f"Video recording initialized: {output_name}")
            
        except Exception as e:
            print(f"Video writer initialization failed: {e}")
            self.enable_recording = False

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
                
                # Spawn actors with validation and retry logic
                self.vehicle_manager = VehicleManager(self.world)
                vehicles_spawned, self.vehicle = self.vehicle_manager.spawn_vehicle()
                if not vehicles_spawned:
                    raise RuntimeError("Failed to spawn required actors")
                
                # Spawn actors with validation and retry logic
                if self.enable_traffic:
                    self.traffic_manager = TrafficManager(self.world)
                    if not self.traffic_manager.spawn_traffic():
                        raise RuntimeError("Failed to spawn traffic actors")
                
                # Setup validation after all actors are ready
                if self.validation_mode:
                    self._setup_validation()
                
                self.connection_stable = True
                print("CARLA setup completed successfully")
                return True
                
            except Exception as e:
                print(f"Setup attempt {attempt + 1} failed: {e}")
                self.vehicle_manager._cleanup_partial_setup()
                self.traffic_manager._cleanup_partial_setup()
                
                if attempt < self.max_retries - 1:
                    print("Retrying in 5 seconds...")
                    time.sleep(5)  # Give server time to stabilize
                else:
                    print("All connection attempts failed")
                    return False
        
        return False
    
    def cleanup_partial_setup(self):
        """Coordinate cleanup across all managers"""
        print("Cleaning up partial setup...")
        self.vehicle_manager.cleanup()   # Each manager cleans itself
        self.traffic_manager.cleanup()   
        self.camera_manager.cleanup()

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

    def _spawn_camera(self):
        """
        Spawn camera with enhanced configuration and validation.
        
        Camera is critical component - needs robust setup with proper configuration
        and validation to ensure image quality and performance.
        """
        try:
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Enhanced camera configuration for better image quality
            camera_config = self.camera_config
            camera_bp.set_attribute('image_size_x', str(camera_config['image_width']))
            camera_bp.set_attribute('image_size_y', str(camera_config['image_height']))
            camera_bp.set_attribute('fov', str(camera_config['fov']))
            camera_bp.set_attribute('sensor_tick', str(1.0 / self.FPS))  # Match FPS
            
            # Enable post-processing for better image quality
            camera_bp.set_attribute('enable_postprocess_effects', 'true')
            camera_bp.set_attribute('gamma', '2.2')
            
            # Attach camera to vehicle with proper transform
            camera_transform = carla.Transform(
                carla.Location(**camera_config['transform']['location']),
                carla.Rotation(**camera_config['transform']['rotation'])
            )
            
            self.camera = self.world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            self.actors.append(self.camera)
            
            # Setup camera callback with error handling wrapper
            self.camera.listen(lambda image: self._safe_camera_callback(image))
            
            print("Camera configured and attached successfully")
            
        except Exception as e:
            raise RuntimeError(f"Camera setup failed: {e}")

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
                self._restart_camera()

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

    def _restart_camera(self):
        """
        Attempt to restart camera on repeated failures.
        
        Sometimes cameras get into bad state and need restart.
        This is a recovery mechanism for persistent issues.
        """
        try:
            print("Attempting camera restart...")
            
            if self.camera and self.camera.is_alive:
                self.camera.stop()
                time.sleep(0.5)
                self.camera.destroy()
            
            # Remove from actors list to prevent double cleanup
            if self.camera in self.actors:
                self.actors.remove(self.camera)
            
            # Spawn new camera
            self._spawn_camera()
            
            self.frame_timeout_count = 0
            print("Camera restart successful")
            
        except Exception as e:
            print(f"Camera restart failed: {e}")

    def update_display(self):
        """Enhanced display update with error handling and performance info"""
        if self.current_frame is None:
            return
            
        try:
            result = self.current_frame['result']
            gray = self.current_frame['gray']
            edges = self.current_frame['edges']
            masked = self.current_frame['masked']
            
            # Result is already in RGB format from lane detector
            surface = pygame.surfarray.make_surface(np.rot90(np.fliplr(result)))
            self.display.blit(surface, (0, 0))

            # Record video if enabled (convert back to BGR for OpenCV)
            if self.enable_recording and hasattr(self, 'video_out') and self.video_out is not None:
                bgr_frame = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
                self.video_out.write(bgr_frame)

            # Create debug miniatures for development and debugging
            def draw_debug(title, img, y_offset):
                debug_img = cv2.resize(img, (160, 120))
                if len(debug_img.shape) == 2:  # Convert grayscale to RGB for display
                    debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2RGB)
                debug_surface = pygame.surfarray.make_surface(np.rot90(np.fliplr(debug_img)))
                self.display.blit(debug_surface, (self.config['lane_detector']['image_resize']['image_width']-200, y_offset))
                
                # Add text label for clarity
                text = self.font.render(title, True, (255, 255, 255))
                self.display.blit(text, (self.config['lane_detector']['image_resize']['image_width']-200, y_offset))
                
            if self.enable_debugs:
                draw_debug("Gray", gray, 20)
                draw_debug("Edges", edges, 160)
                draw_debug("Masked", masked, 300)

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
                    self.autopilot_enabled = not self.autopilot_enabled
                    self.vehicle.set_autopilot(self.autopilot_enabled)
                    status = "enabled" if self.autopilot_enabled else "disabled"
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
        print(f"Image Resolution: {self.camera_config['image_width']}x{self.camera_config['image_height']}")
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
                    if not self.autopilot_enabled:
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
            ("Disabling autopilot", self._cleanup_autopilot),
            ("Stopping camera", self._cleanup_camera),
            ("Destroying vehicle actors", self.vehicle_manager._cleanup_actors),
            ("Destroying traffic actors", self.traffic_manager._cleanup_actors),
            ("Closing video recording", self._cleanup_recording),
            ("Restoring async mode", self._cleanup_carla_settings),
            ("Closing pygame", self._cleanup_pygame),
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

    def _cleanup_autopilot(self):
        """Disable autopilot safely"""
        if self.vehicle and self.vehicle.is_alive:
            self.vehicle.set_autopilot(False)

    def _cleanup_camera(self):
        """Safe camera cleanup with proper stop sequence"""
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            time.sleep(0.1)  # Let camera stop properly before destroy

    # def _cleanup_actors(self):
    #     """Enhanced actor cleanup with proper sequencing"""
    #     if not hasattr(self, 'actors'):
    #         return
        
    #     # Destroy all actors
    #     for actor in self.actors:
    #         try:
    #             if actor.is_alive:
    #                 actor.destroy()
    #         except:
    #             pass  # Don't let individual destroy failure stop cleanup
        
    #     self.actors.clear()

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

    def _cleanup_pygame(self):
        """Safe pygame cleanup"""
        try:
            pygame.quit()
        except:
            pass


if __name__ == '__main__':
    # Load config with error handling
    try:
        with open("config.yaml") as f:
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