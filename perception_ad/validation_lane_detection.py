import numpy as np
import cv2
import csv
import os
import datetime
import logging
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class ValidationMetrics:
    """Data class for validation metrics"""
    frame_id: int
    mean_error_px: Optional[float]
    rmse_px: Optional[float]
    max_error_px: Optional[float]
    min_error_px: Optional[float]
    pct_within_threshold: Optional[float]
    num_gt_points: int
    num_detected_points: int
    detection_success: bool
    timestamp: float

class LaneValidator:
    """
    Lane detection validation pipeline for autonomous driving systems.
    
    This validator compares detected lane centerlines against CARLA's high-definition
    map ground truth using quantitative metrics and visual overlays.
    
    Features:
    - Ground truth extraction from CARLA HD maps
    - Proper camera intrinsic/extrinsic calibration 
    - Multiple validation metrics (pixel error, RMSE, accuracy percentage)
    - Automated visual overlay generation
    - Comprehensive logging and error handling
    - Statistical analysis and reporting
    """
    
    def __init__(self, config: dict, world, camera_actor, vehicle, lane_detector):
        """
        Initialize lane validation pipeline.
        
        Args:
            config: Configuration dictionary with validation parameters
            world: CARLA world object  
            camera_actor: CARLA camera sensor actor
            vehicle: CARLA vehicle actor (ego vehicle)
            lane_detector: Lane detection system instance
        """
        self.logger = self._setup_logging()
        
        # Store CARLA objects and configuration
        self.config = config['validation']
        self.world = world
        self.camera = camera_actor
        self.vehicle = vehicle
        self.lane_detector = lane_detector
        self.map = world.get_map()
        
        # Setup output directory with timestamp
        self._setup_output_directory()
        
        # Compute camera intrinsics and validate setup
        self._compute_camera_intrinsics()
        self._validate_configuration()
        
        # Initialize logging structures
        self.validation_logs: List[ValidationMetrics] = []
        self.statistics = {
            'total_frames': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_error': 0.0,
            'best_error': float('inf'),
            'worst_error': 0.0
        }
        
        self.logger.info(f"Lane validator initialized - Output: {self.output_dir}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation pipeline"""
        logger = logging.getLogger(f"{__name__}.LaneValidator")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_output_directory(self) -> None:
        """Create timestamped output directory for validation results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = self.config.get('output_dir', 'validation_results')
        self.output_dir = f"{base_dir}_{timestamp}"
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Create subdirectories for organization
            os.makedirs(os.path.join(self.output_dir, 'overlays'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, 'logs'), exist_ok=True)
            
        except OSError as e:
            raise RuntimeError(f"Failed to create output directory {self.output_dir}: {e}")
    
    def _compute_camera_intrinsics(self) -> None:
        """
        Compute camera intrinsic matrix from CARLA camera parameters.
        
        The intrinsic matrix K transforms 3D camera coordinates to 2D image coordinates:
        [u, v, 1]^T = K * [X_cam, Y_cam, Z_cam]^T
        """
        try:
            # Get image dimensions from lane detector
            self.img_w = self.lane_detector.params.img_width 
            self.img_h = self.lane_detector.params.img_height
            
            # Extract field of view from camera attributes
            fov_degrees = float(self.camera.attributes['fov'])
            
            # Convert FOV to focal length in pixels
            # focal_length = image_width / (2 * tan(fov/2))
            focal_length = self.img_w / (2.0 * np.tan(np.radians(fov_degrees) / 2.0))
            
            # Construct intrinsic matrix
            self.K = np.array([
                [focal_length, 0,            self.img_w / 2],
                [0,            focal_length, self.img_h / 2], 
                [0,            0,            1             ]
            ], dtype=np.float64)
            
            self.logger.debug(f"Camera intrinsics computed: focal_length={focal_length:.2f}, FOV={fov_degrees}°")
            
        except (KeyError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Failed to compute camera intrinsics: {e}")
    
    def _validate_configuration(self) -> None:
        """Validate configuration parameters"""
        required_keys = ['threshold_px', 'y_min_pct', 'num_captures', 'interval_seconds']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required validation config key: {key}")
        
        # Validate parameter ranges
        if not (0.0 <= self.config['y_min_pct'] <= 1.0):
            raise ValueError(f"y_min_pct must be in [0,1], got {self.config['y_min_pct']}")
        
        if self.config['threshold_px'] <= 0:
            raise ValueError(f"threshold_px must be positive, got {self.config['threshold_px']}")
    
    def sample_ground_truth_waypoints(self, vehicle, sample_distance: float = 0.1, 
                                    max_points: int = 100, lookahead_distance: float = 50.0) -> List:
        """
        Sample ground truth waypoints along the ego vehicle's lane.
        
        Args:
            vehicle: CARLA vehicle actor
            sample_distance: Distance between waypoint samples in meters
            max_points: Maximum number of waypoints to sample
            lookahead_distance: Maximum distance to sample ahead
            
        Returns:
            List of CARLA Location objects representing lane centerline
        """
        try:
            ego_location = vehicle.get_transform().location
            ego_waypoint = self.map.get_waypoint(ego_location)
            
            if ego_waypoint is None:
                self.logger.warning("Vehicle not on a valid lane - using vehicle position")
                return [ego_location]
            
            waypoints = []
            current_waypoint = ego_waypoint
            total_distance = 0.0
            
            for _ in range(max_points):
                waypoints.append(current_waypoint.transform.location)
                
                # Get next waypoint
                next_waypoints = current_waypoint.next(sample_distance)
                if not next_waypoints:
                    self.logger.debug(f"Reached end of lane after {len(waypoints)} waypoints")
                    break
                
                current_waypoint = next_waypoints[0]
                total_distance += sample_distance
                
                # Stop if we've gone too far ahead
                if total_distance > lookahead_distance:
                    break
            
            self.logger.debug(f"Sampled {len(waypoints)} ground truth waypoints over {total_distance:.1f}m")
            return waypoints
            
        except Exception as e:
            self.logger.error(f"Failed to sample ground truth waypoints: {e}")
            return []
    
    def project_3d_to_image(self, world_location) -> Optional[Tuple[int, int]]:
        """
        Project 3D world coordinates to 2D image pixels using camera transform.
        
        Args:
            world_location: CARLA Location in world coordinates
            
        Returns:
            Tuple (u, v) of pixel coordinates, or None if point is behind camera
        """
        try:
            # Convert to homogeneous coordinates
            world_point = np.array([
                world_location.x, 
                world_location.y, 
                world_location.z, 
                1.0
            ])
            
            # Transform from world to camera coordinates
            camera_transform_matrix = np.array(self.camera.get_transform().get_inverse_matrix())
            camera_point = camera_transform_matrix @ world_point
            
            # CARLA coordinate system conversion: (x,y,z) → (y,-z,x)
            camera_coords = np.array([camera_point[1], -camera_point[2], camera_point[0]])
            
            # Check if point is behind camera (negative Z)
            if camera_coords[2] <= 0:
                return None
            
            # Project to image coordinates using intrinsic matrix
            image_coords_homogeneous = self.K @ camera_coords
            
            # Convert to pixel coordinates
            u = int(image_coords_homogeneous[0] / image_coords_homogeneous[2])
            v = int(image_coords_homogeneous[1] / image_coords_homogeneous[2])
            
            # Check if point is within image bounds
            if 0 <= u < self.img_w and 0 <= v < self.img_h:
                return (u, v)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to project 3D point to image: {e}")
            return None
    
    def extract_detected_centerline(self, left_coords: Optional[np.ndarray], 
                                  right_coords: Optional[np.ndarray],
                                  num_points: int = 30) -> List[Tuple[int, int]]:
        """
        Extract detected lane centerline points from left and right lane coordinates.
        
        Args:
            left_coords: Left lane coordinates [x1, y1, x2, y2]
            right_coords: Right lane coordinates [x1, y1, x2, y2] 
            num_points: Number of interpolated centerline points
            
        Returns:
            List of (u, v) centerline pixel coordinates
        """
        if left_coords is None or right_coords is None:
            return []
        
        try:
            # Extract line endpoints
            x1_left, y1_left, x2_left, y2_left = left_coords
            x1_right, y1_right, x2_right, y2_right = right_coords
            
            # Use consistent y-range (assuming lines have similar y-coordinates)
            y_start = max(y1_left, y1_right)  # Bottom of image
            y_end = min(y2_left, y2_right)    # Top of detection region
            
            if y_start <= y_end:
                self.logger.warning("Invalid lane line y-coordinates")
                return []
            
            # Generate interpolated y-coordinates
            y_coords = np.linspace(y_start, y_end, num_points, dtype=int)
            
            centerline_points = []
            for y in y_coords:
                # Linear interpolation for each lane
                t = (y - y_start) / (y_end - y_start) if y_end != y_start else 0
                
                # Interpolate x-coordinates for left and right lanes
                x_left = int(x1_left + t * (x2_left - x1_left))
                x_right = int(x1_right + t * (x2_right - x1_right))
                
                # Compute centerline
                x_center = (x_left + x_right) // 2
                centerline_points.append((x_center, y))
            
            return centerline_points
            
        except Exception as e:
            self.logger.error(f"Failed to extract detected centerline: {e}")
            return []
    
    def compute_validation_metrics(self, detected_points: List[Tuple[int, int]], 
                                 ground_truth_points: List[Tuple[int, int]]) -> ValidationMetrics:
        """
        Compute comprehensive validation metrics comparing detected vs ground truth.
        
        Args:
            detected_points: List of detected centerline points (u, v)
            ground_truth_points: List of ground truth points (u, v)
            
        Returns:
            ValidationMetrics object with all computed metrics
        """
        if not detected_points or not ground_truth_points:
            return ValidationMetrics(
                frame_id=0,
                mean_error_px=None,
                rmse_px=None,
                max_error_px=None,
                min_error_px=None,
                pct_within_threshold=None,
                num_gt_points=len(ground_truth_points),
                num_detected_points=len(detected_points),
                detection_success=False,
                timestamp=0.0
            )
        
        try:
            # Find closest detected point for each ground truth point
            errors = []
            for u_gt, v_gt in ground_truth_points:
                # Find detected point with closest y-coordinate
                y_distances = [abs(v_gt - v_det) for _, v_det in detected_points]
                closest_idx = np.argmin(y_distances)
                u_det, v_det = detected_points[closest_idx]
                
                # Compute horizontal (lateral) error
                lateral_error = abs(u_gt - u_det)
                errors.append(lateral_error)
            
            errors = np.array(errors)
            
            # Compute metrics
            mean_error = float(np.mean(errors))
            rmse = float(np.sqrt(np.mean(errors**2)))
            max_error = float(np.max(errors))
            min_error = float(np.min(errors))
            
            # Percentage within threshold
            threshold = self.config['threshold_px']
            pct_within = float(np.mean(errors < threshold)) * 100
            
            return ValidationMetrics(
                frame_id=0,  # Will be set by caller
                mean_error_px=mean_error,
                rmse_px=rmse,
                max_error_px=max_error,
                min_error_px=min_error,
                pct_within_threshold=pct_within,
                num_gt_points=len(ground_truth_points),
                num_detected_points=len(detected_points),
                detection_success=True,
                timestamp=0.0  # Will be set by caller
            )
            
        except Exception as e:
            self.logger.error(f"Failed to compute validation metrics: {e}")
            return ValidationMetrics(
                frame_id=0,
                mean_error_px=None,
                rmse_px=None, 
                max_error_px=None,
                min_error_px=None,
                pct_within_threshold=None,
                num_gt_points=len(ground_truth_points),
                num_detected_points=len(detected_points),
                detection_success=False,
                timestamp=0.0
            )
    
    def create_validation_overlay(self, image: np.ndarray, ground_truth_points: List[Tuple[int, int]],
                                detected_points: List[Tuple[int, int]], metrics: ValidationMetrics) -> np.ndarray:
        """
        Create visual overlay showing ground truth vs detected points with metrics.
        
        Args:
            image: Base image for overlay
            ground_truth_points: Ground truth centerline points
            detected_points: Detected centerline points  
            metrics: Validation metrics for annotation
            
        Returns:
            Annotated image with overlays
        """
        overlay_image = image.copy()
        
        try:
            # Draw ground truth points (yellow circles)
            for u, v in ground_truth_points:
                cv2.circle(overlay_image, (u, v), 4, (0, 255, 255), -1)  # Yellow
            
            # Draw detected points (red circles)
            for u, v in detected_points:
                cv2.circle(overlay_image, (u, v), 4, (0, 0, 255), -1)  # Red
            
            # Add metrics text overlay
            if metrics.detection_success and metrics.mean_error_px is not None:
                text_lines = [
                    f"Mean Error: {metrics.mean_error_px:.1f}px",
                    f"RMSE: {metrics.rmse_px:.1f}px", 
                    f"Within {self.config['threshold_px']}px: {metrics.pct_within_threshold:.1f}%",
                    f"GT Points: {metrics.num_gt_points}, Det Points: {metrics.num_detected_points}"
                ]
                
                # Draw text background
                text_height = 25
                background_height = len(text_lines) * text_height + 10
                cv2.rectangle(overlay_image, (10, 10), (400, background_height), (0, 0, 0), -1)
                
                # Draw text
                for i, line in enumerate(text_lines):
                    y_position = 30 + i * text_height
                    cv2.putText(overlay_image, line, (15, y_position), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(overlay_image, "DETECTION FAILED", (15, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Add legend
            cv2.putText(overlay_image, "Yellow: Ground Truth", (15, self.img_h - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(overlay_image, "Red: Detected", (15, self.img_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        except Exception as e:
            self.logger.error(f"Failed to create validation overlay: {e}")
        
        return overlay_image
    
    def validate_single_frame(self, image: np.ndarray, left_coords: Optional[np.ndarray],
                            right_coords: Optional[np.ndarray], frame_id: int, 
                            timestamp: float) -> ValidationMetrics:
        """
        Perform validation on a single frame.
        
        Args:
            image: Camera image
            left_coords: Detected left lane coordinates
            right_coords: Detected right lane coordinates
            frame_id: Frame identifier
            timestamp: Simulation timestamp
            
        Returns:
            ValidationMetrics for this frame
        """
        try:
            # 1. Sample ground truth waypoints
            gt_3d_points = self.sample_ground_truth_waypoints(self.vehicle)
            
            # 2. Project to 2D image coordinates
            gt_2d_points = []
            for location in gt_3d_points:
                pixel_coords = self.project_3d_to_image(location)
                if pixel_coords is not None:
                    gt_2d_points.append(pixel_coords)
            
            # 3. Filter ground truth points by y-coordinate (match detection region)
            y_min_threshold = int(self.config['y_min_pct'] * self.img_h)
            filtered_gt_points = [(u, v) for u, v in gt_2d_points if v >= y_min_threshold]
            
            # 4. Extract detected centerline points
            detected_centerline = self.extract_detected_centerline(left_coords, right_coords)
            
            # 5. Compute validation metrics
            metrics = self.compute_validation_metrics(detected_centerline, filtered_gt_points)
            metrics.frame_id = frame_id
            metrics.timestamp = timestamp
            
            # 6. Update statistics
            self._update_statistics(metrics)
            
            # 7. Create and save visualization if configured
            if self.config.get('draw_det_vs_gt', True):
                overlay = self.create_validation_overlay(image, filtered_gt_points, 
                                                       detected_centerline, metrics)
                
                # Save overlay image
                overlay_path = os.path.join(self.output_dir, 'overlays', f"frame_{frame_id:04d}.png")
                cv2.imwrite(overlay_path, overlay)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Validation failed for frame {frame_id}: {e}")
            # Return empty metrics on failure
            return ValidationMetrics(
                frame_id=frame_id,
                mean_error_px=None,
                rmse_px=None,
                max_error_px=None,
                min_error_px=None,
                pct_within_threshold=None,
                num_gt_points=0,
                num_detected_points=0,
                detection_success=False,
                timestamp=timestamp
            )
    
    def _update_statistics(self, metrics: ValidationMetrics) -> None:
        """Update running statistics with new frame metrics"""
        self.statistics['total_frames'] += 1
        
        if metrics.detection_success and metrics.mean_error_px is not None:
            self.statistics['successful_detections'] += 1
            
            # Update error statistics
            error = metrics.mean_error_px
            self.statistics['average_error'] = (
                (self.statistics['average_error'] * (self.statistics['successful_detections'] - 1) + error) /
                self.statistics['successful_detections']
            )
            self.statistics['best_error'] = min(self.statistics['best_error'], error)
            self.statistics['worst_error'] = max(self.statistics['worst_error'], error)
        else:
            self.statistics['failed_detections'] += 1
    
    def save_validation_results(self) -> None:
        """Save all validation results to CSV and generate summary report"""
        try:
            # Save detailed metrics to CSV
            csv_path = os.path.join(self.output_dir, 'logs', self.config.get('log_csv', 'validation_metrics.csv'))
            
            if self.validation_logs:
                # Convert ValidationMetrics objects to dictionaries
                fieldnames = [
                    'frame_id', 'timestamp', 'mean_error_px', 'rmse_px', 'max_error_px', 
                    'min_error_px', 'pct_within_threshold', 'num_gt_points', 
                    'num_detected_points', 'detection_success'
                ]
                
                with open(csv_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for metrics in self.validation_logs:
                        writer.writerow({
                            'frame_id': metrics.frame_id,
                            'timestamp': metrics.timestamp,
                            'mean_error_px': metrics.mean_error_px,
                            'rmse_px': metrics.rmse_px,
                            'max_error_px': metrics.max_error_px,
                            'min_error_px': metrics.min_error_px,
                            'pct_within_threshold': metrics.pct_within_threshold,
                            'num_gt_points': metrics.num_gt_points,
                            'num_detected_points': metrics.num_detected_points,
                            'detection_success': metrics.detection_success
                        })
            
            # Generate summary report
            self._generate_summary_report()
            
            self.logger.info(f"Validation results saved to {self.output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")
    
    def _generate_summary_report(self) -> None:
        """Generate summary report with statistics"""
        report_path = os.path.join(self.output_dir, 'logs', 'validation_summary.txt')
        
        try:
            with open(report_path, 'w') as f:
                f.write("Lane Detection Validation Summary\n")
                f.write("=" * 40 + "\n\n")
                
                f.write(f"Total Frames Processed: {self.statistics['total_frames']}\n")
                f.write(f"Successful Detections: {self.statistics['successful_detections']}\n")
                f.write(f"Failed Detections: {self.statistics['failed_detections']}\n")
                
                if self.statistics['successful_detections'] > 0:
                    success_rate = (self.statistics['successful_detections'] / 
                                  self.statistics['total_frames']) * 100
                    f.write(f"Success Rate: {success_rate:.1f}%\n\n")
                    
                    f.write("Error Statistics:\n")
                    f.write(f"  Average Error: {self.statistics['average_error']:.2f} pixels\n")
                    f.write(f"  Best Error: {self.statistics['best_error']:.2f} pixels\n")
                    f.write(f"  Worst Error: {self.statistics['worst_error']:.2f} pixels\n")
                
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")
    
    def run_validation(self, sim_time: float, image: np.ndarray, frame_id: int, 
                      capture_times: float, logs: list, left_coords: Optional[np.ndarray], 
                      right_coords: Optional[np.ndarray]) -> Tuple[int, float, list]:
        """
        Validation method.
        """
        # Only process frames we're going to save
        if sim_time >= capture_times and frame_id < self.config['num_captures']:
            
            # NOW we do the validation work (only for frames we're saving)
            metrics = self.validate_single_frame(image, left_coords, right_coords, frame_id, sim_time)
            
            # Convert metrics to dictionary for backward compatibility
            metrics_dict = {
                'frame': metrics.frame_id,
                'mean_error': metrics.mean_error_px,
                'rmse': metrics.rmse_px,
                f"pct_within_{self.config['threshold_px']}px": metrics.pct_within_threshold,
                'len_det_mid': metrics.num_detected_points
            }
            
            logs.append(metrics_dict)
            self.validation_logs.append(metrics)
            
            frame_id += 1
            capture_times += self.config['interval_seconds']
            
            # Save results
            self.save_validation_results()
            
            if frame_id >= self.config['num_captures']:
                if not hasattr(self, '_validation_complete_logged'):
                    self.logger.info(f"Validation complete: {self.statistics['total_frames']} frames processed")
                    print(f"Validation complete: results in {self.output_dir}")
                    self._validation_complete_logged = True
        
        return frame_id, capture_times, logs