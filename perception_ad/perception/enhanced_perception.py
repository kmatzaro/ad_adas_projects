import cv2
import numpy as np
from perception.simple_lane_detection import SimpleLaneDetector
from perception.object_detection import ObjectDetector, DetectedObject
from perception.object_tracker import ObjectTracker
from perception.lidar_depth_fusion import LiDARDepthFusion
from typing import List, Dict, Optional, Tuple, Literal
import logging
import time
from dataclasses import dataclass

@dataclass
class PerceptionResult:
    camera_id: str
    processed_image: np.ndarray
    detected_objects: List[DetectedObject] = None  # Only for object detection
    lane_coords: Tuple = None                      # Only for lane detection
    debug_images: Dict[str, np.ndarray] = None     # Debug overlays
    timing_metrics: Dict = None
    timestamp: float = None

class EnhancedPerception():
    """
    Enhanced Perception class that fuses information from the LaneDetector class
    and the ObjectDetector class
    """

    def __init__(self, config: Dict):
        """
        Initialize the class instance
        """

        self.logger = self._setup_logging()
        self.logger.info("Initializing the Enhanced Perception System")

        # Store configuration
        self.config = config

        # Initialize lane + object detectors
        self.lane_detector = SimpleLaneDetector(config)
        self.object_detector = ObjectDetector(config)

        # Frame-to-frame object tracker
        self.object_tracker = ObjectTracker(config)

        # LiDAR-camera depth fusion
        self.lidar_depth_fusion = LiDARDepthFusion(config)

        # Get image dimensions for processing
        self.image_width = config['lane_detector']['image_resize']['image_width']
        self.image_height = config['lane_detector']['image_resize']['image_height']

        # Log system status
        self._log_system_status()

        self.color_map = {
            'person': (255, 0, 0),          # RED (BGR format)
            'bicycle': (0, 255, 255),       # Yellow
            'car': (0, 0, 255),             # Blue
            'motorcycle': (255, 255, 0),    # Cyan
            'bus': (0, 100, 255),           # Orange
            'truck': (0, 0, 200),           # Dark red
            'traffic_light': (0, 255, 255), # Yellow
            'stop_sign': (255, 0, 255),     # Magenta
        }

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _log_system_status(self):
        """Log the status of all system components"""

        lane_status = "Ready"  # Lane detection always works
        object_status = "Ready" if self.object_detector.enabled else "Disabled (YOLO not available)"

        self.logger.info("Enhanced Perception System Status:")
        self.logger.info(f"  Lane Detection: {lane_status}")
        self.logger.info(f"  Object Detection: {object_status}")
        self.logger.info(f"  Target Resolution: {self.image_width}x{self.image_height}")

        if not self.object_detector.enabled:
            self.logger.warning("Object detection disabled - system will work with lanes only")

    def process_lane_detection_camera(self, image, camera_id: str, timestamp) -> PerceptionResult:
        """
        Process a single frame with integrated lane detection

        Args:
            image: Input image

        Returns:
            PerceptionResult dataclass
        """
        try:
            # Lane detection system
            lane_start_time = time.time()
            lane_result, gray, edges, masked, left_coords, right_coords = self.lane_detector.process_image(image)
            lane_end_time = (time.time() - lane_start_time) * 1000 # For ms

            return PerceptionResult(
                camera_id = camera_id,
                processed_image = lane_result,
                lane_coords = (left_coords, right_coords),
                debug_images = {"gray": gray, "edges": edges, "masked": masked},
                timing_metrics = {f'{camera_id}_lane_detection_time_ms': lane_end_time},
                timestamp = timestamp
            )

        except Exception as e:
            self.logger.error("Failed to process image in the lane detection system")
            return PerceptionResult(
                camera_id = camera_id,
                processed_image = image,  # Return original image on failure
                lane_coords = None,
                debug_images = None,
                timing_metrics = {f'{camera_id}_lane_detection_time_ms': 0, 'error': True},
                timestamp = timestamp
            )

    def process_object_detection_camera(self, image, camera_id: str, timestamp) -> PerceptionResult:
        """
        Process a single frame with integrated object detection

        Args:
            image: Input image

        Returns:
            PerceptionResult dataclass
        """
        try:
            # Object detection time
            object_start_time = time.time()
            detected_objects = self.object_detector.detect_objects(image)
            detected_objects = self.object_tracker.update(detected_objects)
            object_end_time = (time.time() - object_start_time) * 1000 # For ms

            # Draw objects on image
            perception_image = self._draw_objects(detected_objects, image)

            return PerceptionResult(
                camera_id = camera_id,
                processed_image = perception_image,
                detected_objects = detected_objects,
                timing_metrics= {f"{camera_id}_object_detection_time_ms": object_end_time},
                timestamp = timestamp
            )

        except Exception as e:
            self.logger.error("Failed to process image in the perception system")
            return PerceptionResult(
                camera_id = camera_id,
                processed_image = image,
                timing_metrics= {f"{camera_id}_object_detection_time_ms": 0, 'error': True},
                timestamp = timestamp
            )

    def combined_camera_process(
        self,
        image,
        camera_id: str,
        timestamp,
        lidar_points: Optional[np.ndarray] = None,
    ) -> PerceptionResult:
        """
        Process a single frame with integrated lane detection and object detection.
        If lidar_points is provided, depth estimates are fused into each detection.

        Args:
            image:        Input image (preprocessed).
            camera_id:    Sensor identifier string.
            timestamp:    CARLA snapshot timestamp.
            lidar_points: (N, 4) float32 LiDAR point cloud in sensor frame, or None.

        Returns:
            PerceptionResult dataclass
        """
        try:
            # Lane detection
            lane_start_time = time.time()
            lane_result, gray, edges, masked, left_coords, right_coords = self.lane_detector.process_image(image)
            lane_end_time = (time.time() - lane_start_time) * 1000

            # Object detection + tracking + depth fusion
            object_start_time = time.time()
            detected_objects = self.object_detector.detect_objects(image)
            detected_objects = self.object_tracker.update(detected_objects)
            detected_objects = self.lidar_depth_fusion.assign_depths(detected_objects, lidar_points)
            object_end_time = (time.time() - object_start_time) * 1000

            # Draw objects on image
            perception_image = self._draw_objects(detected_objects, lane_result)

            return PerceptionResult(
                camera_id = camera_id,
                processed_image = perception_image,
                detected_objects = detected_objects,
                lane_coords = (left_coords, right_coords),
                debug_images = {"gray": gray, "edges": edges, "masked": masked},
                timing_metrics = {
                    f'{camera_id}_lane_detection_time_ms': lane_end_time,
                    f"{camera_id}_object_detection_time_ms": object_end_time,
                    f"total_{camera_id}_time_ms": lane_end_time + object_end_time,
                },
                timestamp = timestamp
            )

        except Exception as e:
            self.logger.error("Failed to process image in the lane detection system")
            return PerceptionResult(
                camera_id = camera_id,
                processed_image = image,  # Return original image on failure
                detected_objects = None,
                lane_coords = None,
                debug_images = None,
                timing_metrics = {f"total_{camera_id}_time_ms": 0, 'error': True},
                timestamp = timestamp
            )

    def _draw_objects(self, detected_objects, image):
        """Draw the bounding boxes of detected objects"""

        if not detected_objects or not self.object_detector.enabled:
            return image

        object_image = image.copy()

        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            color = self._get_color_for_object(obj.class_name)

            cv2.rectangle(object_image, (x1, y1), (x2, y2), color, 2)

            track_str = f" #{obj.track_id}" if obj.track_id is not None else ""
            depth_str = f" {obj.distance_estimate:.1f}m" if obj.distance_estimate is not None else ""
            label = f"{obj.class_name}{track_str}: {obj.confidence:.2f}{depth_str} ({obj.relative_position})"

            label_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
            cv2.putText(object_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        return object_image

    def _get_color_for_object(self, obj_class: str) -> Tuple:
        return self.color_map.get(obj_class, (255, 255, 255))  # White default