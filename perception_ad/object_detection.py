import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Literal
from dataclasses import dataclass

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class DetectedObject:
    """Data class representing a detected object with driving-relevant information"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center_x: int
    center_y: int
    area: int
    distance_estimate: Optional[float] = None
    relative_position: Literal['left', 'right', 'unknown'] = 'unknown'

class ObjectDetector:
    """
    High-performance YOLO-based object detection optimized for autonomous driving.
    
    Features:
    - Real-time detection
    - Driving-relevant object filtering
    - Confidence-based filtering
    - Performance monitoring
    - Graceful degradation
    """
    
    def __init__(self, config: Dict):
        """Initialize object detector with configuration"""
        self.logger = self._setup_logging()
        self.config = config['object_detection']
        self.image_center_x = config['lane_detector']['image_resize']['image_width']/2
        self.FPS = config['carla']['FPS']
        
        # Check is YOLO model is available
        if not YOLO_AVAILABLE:
            self.enabled = False
            self.logger.error("YOLO detection model is not available, check if ultralytics library is installed correctly!")
            return

        # Set configuration values for YOLO model
        self.model_size = self.config['model_size']
        self.confidence_threshold = self.config['confidence_threshold']
        self.nms_thershold = self.config['nms_threshold']

        # Detection classes from COCO dataset
        self.detection_classes = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck',
            9: 'traffic light',
            11: 'stop sign',
        }

        self.model = None
        self._initialize_yolo()

        self.logger.info(f"ObjectDetector initialized: model size {self.model_size}")

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
    
    def _initialize_yolo(self):
        """Load and configure yolo model for inference"""

        try:
            model_name = f"yolo{self.model_size}.pt"
            self.logger.info(f"Loading YOLO model: {model_name}")
            self.model = YOLO(model_name)

            # Configure for inference (disable training features)
            self.model.overrides.update({
                'verbose': False,    # Reduce console output
                'save': False,       # Don't save results
                'save_txt': False,   # Don't save annotations
                'save_conf': False,  # Don't save confidence scores
                'save_crop': False,  # Don't save cropped detections
                'show': False,       # Don't display results
                'plots': False       # Don't create plots
            })

            # Warm up the model for consistent performance
            self._warmup_model()

            self.enabled = True
            self.logger.info("YOLO model loaded and ready")
        
        except Exception as e:
            self.logger.error(f"Failed to initialize YOLO model: {e}")
            self.model = None
            self.enabled = False

    
    def _warmup_model(self):
        """
        Warm up YOLO model with dummy inference.
        
        This ensures consistent performance by loading the model into GPU memory
        and running the initial compilation steps.
        """
        try:
            self.logger.debug("Warming up YOLO model...")
            
            # Create dummy image (640x640 is YOLO's default input size)
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run a few dummy inferences
            for _ in range(3):
                _ = self.model(dummy_image, verbose=False)
            
            self.logger.debug("YOLO warmup completed")
            
        except Exception as e:
            self.logger.warning(f"YOLO warmup failed: {e}")

    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """
        Detect objects using the YOLO model

        Args: 
            image: Input image in RGB format (numpy array)
        Returns:
            List of detected objects as dictionary
        """
        if not self.enabled or self.model is None:
            return []
        
        try:
            results = self.model(
                image,
                classes = list(self.detection_classes.keys()),
                conf = self.confidence_threshold,
                iou = self.nms_thershold,
                verbose = False)
            
            detected_objects = self._extract_detections(results[0])

            return detected_objects
        
        except Exception as e:
            self.logger.error(f"Object detection failed: {type(e).__name__}: {str(e)}")
            return []
    
    def _extract_detections(self, results) -> List[DetectedObject]:
        """Extract detected objects' information and convert to DetectedObject dataclass"""

        if results.boxes is None or len(results.boxes) == 0:
            return []
        
        try:

            bbox_coords = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()
            class_ids = results.boxes.cls.cpu().numpy()

            detected_objects = list()

            for bbox, confidence, ids in zip(bbox_coords, conf, class_ids):
                
                # Skip if detection class is not in the desired detection class
                if int(ids) not in self.detection_classes:
                    continue

                # Calulate center coordinates for bbox and area
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                bbox_area = abs(bbox[3] - bbox[1])*abs(bbox[2] - bbox[0])

                # Calculate relative position
                if center_x < self.image_center_x:
                    rel_pos = 'left'
                else:
                    rel_pos = 'right'

                detected_objects.append(
                    DetectedObject(
                    class_id=int(ids),
                    class_name=self.detection_classes[int(ids)],
                    confidence=float(confidence),
                    bbox=tuple(bbox.astype(int)),
                    center_x=int(center_x),
                    center_y=int(center_y),
                    area=int(bbox_area),
                    relative_position=rel_pos)
                    )
            
            return detected_objects

        except Exception as e:
            self.logger.error("Failed to extract information from detection results")
            return []