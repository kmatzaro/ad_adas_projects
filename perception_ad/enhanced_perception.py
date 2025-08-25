import cv2
import numpy as np
from simple_lane_detection import SimpleLaneDetector
from object_detection import ObjectDetector, DetectedObject
from typing import List, Dict, Tuple, Optional
import logging
import time

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

        # Initilize lane + object detectors
        self.lane_detector = SimpleLaneDetector(config)
        self.object_detector = ObjectDetector(config)

        # Get image dimensions for processing
        self.image_width = config['lane_detector']['image_resize']['image_width'] 
        self.image_height = config['lane_detector']['image_resize']['image_height']

        # Log system status
        self._log_system_status()

        self.color_map ={   
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

    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                        Optional[np.ndarray], Optional[np.ndarray], List[DetectedObject], Dict]:
        """Process a single frame with integrated perception
        
        Args:
            image: Input image
        
        Returns:
            To be determined
        """
        try:
            # Resize image once
            resized_image = self.lane_detector._preprocess_image(image)

            # Lane detection system
            lane_start_time = time.time()
            lane_result, gray, edges, masked, left_coords, right_coords = self.lane_detector.process_image(resized_image)
            lane_end_time = (time.time() - lane_start_time) * 1000 # For ms

            # Object detection time
            object_start_time = time.time()
            detected_objects = self.object_detector.detect_objects(resized_image)
            object_end_time = (time.time() - object_start_time) * 1000 # For ms

            # Draw objects on image
            perception_image = self._draw_objects(detected_objects, lane_result)

            timing_metrics = {
                'lane_detection_time_ms': lane_end_time,
                'object_detection_time_ms': object_end_time,
                'total_end_time': lane_end_time + object_end_time
            }

            return perception_image, gray, edges, masked, left_coords, right_coords, detected_objects, timing_metrics

        except Exception as e:
            self.logger.error("Failed to process image with the enhanced perception system")
            return self.lane_detector._create_error_output(image)
        
    def _draw_objects(self, detected_objects, image):
        """Draw the bounding boxes of detected objects"""

        if not detected_objects or not self.object_detector.enabled:
            return image
        
        object_image = image.copy()

        for obj in detected_objects:
            x1, y1, x2, y2 = obj.bbox
            color = self._get_color_for_object(obj.class_name)

            cv2.rectangle(object_image, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name}: {obj.confidence:.2f} ({obj.relative_position})"
            label_y = y1 - 10 if y1 - 10 > 20 else y2 + 20
            cv2.putText(object_image, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return object_image
    
    def _get_color_for_object(self, obj_class: str) -> Tuple:
        return self.color_map.get(obj_class, (255, 255, 255))  # White default  