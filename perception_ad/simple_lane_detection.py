import numpy as np
import cv2
import logging
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from bev_transformer import BEVTransformer

@dataclass
class LaneCoordinates:
    """Data class to represent lane coordinates"""
    x1: int
    y1: int
    x2: int
    y2: int
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])
    
    @classmethod
    def from_array(cls, coords: np.ndarray) -> 'LaneCoordinates':
        return cls(int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))

class LaneDetectionParams:
    """Configuration parameters for lane detection"""
    def __init__(self, config: dict):
        lane_config = config['lane_detector']

        # Original image parameters
        self.img_width_original = lane_config['image_resize']['image_width']
        self.img_height_original = lane_config['image_resize']['image_height']
        
        # Image processing parameters
        self.img_width = lane_config['image_resize']['image_width']
        self.img_height = lane_config['image_resize']['image_height']
        
        # Gaussian blur parameters
        gauss_config = lane_config['gaussian_blur']
        self.gauss_kernel_x = gauss_config['kernel_size_x']
        self.gauss_kernel_y = gauss_config['kernel_size_y']
        self.gauss_sigma_x = gauss_config['sigma_x']
        
        # Canny edge detection parameters
        canny_config = lane_config['canny']
        self.canny_low = canny_config['low_thresh']
        self.canny_high = canny_config['high_thresh']
        
        # Hough line detection parameters
        hough_config = lane_config['hough']
        self.hough_rho = hough_config['rho']
        self.hough_threshold = hough_config['threshold']
        self.hough_min_line_len = hough_config['min_line_len']
        self.hough_max_line_gap = hough_config['max_line_gap']
        
        # Temporal filtering parameters
        self.smoothing_factor = lane_config['smoothing_factor']
        self.max_missing_frames = lane_config['max_missing']
        
        # Display options
        self.display_lane_overlay = lane_config['display_lane_overlay']
        self.display_lane_lines = lane_config['display_lane_lines']
        self.display_center_lane_line = lane_config['display_center_lane_line']
        
        # BEV parameters
        bev_config = lane_config['bev_lane_detector']
        self.bev_enabled = bev_config['bev_enabled']
        
        # Line filtering parameters
        self.min_slope_threshold = 0.3
        self.max_slope_threshold = 3.0

class ROIGenerator:
    """Handles region of interest generation"""
    
    @staticmethod
    def create_dual_lane_roi(width: int, height: int, bev_mode: bool = False) -> np.ndarray:
        """
        Create two separate ROI polygons for left and right lanes.
        This approach provides better lane separation and reduces cross-contamination.
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if bev_mode:
            # BEV mode - different polygon shapes optimized for bird's eye view
            left_polygon = np.array([[
                (0, height),                             # Bottom left corner
                (0, int(height * 0.8)),                  # Left edge, 90% up
                (int(width * 0.3), int(height * 0.0)),   # Approaching center, near top
                (int(width * 0.5), int(height * 0.0)),   # Center, near top
                (int(width * 0.1), height)               # Bottom, 10% from left
            ]], np.int32)

            right_polygon = np.array([[
                (width, height),                         # Bottom right corner
                (width, int(height * 0.8)),              # Right edge, 90% up
                (int(width * 0.5), int(height * 0.0)),   # Approaching center, near top
                (int(width * 0.7), int(height * 0.0)),   # Center, near top
                (int(width * 0.9), height)               # Bottom, 90% from left
            ]], np.int32)
        else:
            # Camera mode - traditional trapezoid shapes
            left_polygon = np.array([[
                (0, height),                            # Bottom left corner
                (0, int(height * 0.9)),                 # Left edge, 90% up
                (int(width * 0.45), int(height * 0.5)), # Approaching center, middle
                (int(width * 0.5), int(height * 0.5)),  # Center, middle
                (int(width * 0.1), height)              # Bottom, 10% from left
            ]], np.int32)

            right_polygon = np.array([[
                (width, height),                        # Bottom right corner
                (width, int(height * 0.9)),             # Right edge, 90% up
                (int(width * 0.55), int(height * 0.5)), # Approaching center, middle
                (int(width * 0.5), int(height * 0.5)),  # Center, middle
                (int(width * 0.9), height)              # Bottom, 90% from left
            ]], np.int32)
        
        # Fill both polygons - this creates two separate lane detection zones
        cv2.fillPoly(mask, left_polygon, 255)
        cv2.fillPoly(mask, right_polygon, 255)
        
        return mask

class LineFilter:
    """Handles filtering and classification of detected lines"""
    
    def __init__(self, params: LaneDetectionParams):
        self.params = params
        self.logger = logging.getLogger(__name__)
    
    def filter_and_classify_lines(self, lines: np.ndarray, frame_width: int) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Filter detected lines and classify them as left or right lanes.
        
        Returns:
            Tuple of (left_lane_fits, right_lane_fits) where each fit is (slope, intercept)
        """
        if lines is None or len(lines) == 0:
            return [], []
        
        left_fits = []
        right_fits = []
        center_x = frame_width // 2
        
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            
            # Skip if line is too short or vertical
            if not self._is_valid_line_geometry(x1, y1, x2, y2):
                continue
            
            # Calculate line parameters
            slope, intercept = self._calculate_line_parameters(x1, y1, x2, y2)
            
            # Apply slope filtering
            if not self._is_valid_slope(slope):
                continue
            
            # Classify as left or right lane
            line_center_x = (x1 + x2) // 2
            
            if self._is_left_lane(slope, line_center_x, center_x):
                left_fits.append((slope, intercept))
            elif self._is_right_lane(slope, line_center_x, center_x):
                right_fits.append((slope, intercept))
        
        return left_fits, right_fits
    
    def _is_valid_line_geometry(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if line has valid geometry"""
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return line_length > 10 and x1 != x2  # Avoid vertical lines and very short lines
    
    def _calculate_line_parameters(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[float, float]:
        """Calculate slope and intercept for a line"""
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept
    
    def _is_valid_slope(self, slope: float) -> bool:
        """Check if slope is within acceptable range"""
        return (self.params.min_slope_threshold < abs(slope) < self.params.max_slope_threshold)
    
    def _is_left_lane(self, slope: float, line_center_x: int, center_x: int) -> bool:
        """Determine if line belongs to left lane"""
        return slope < 0 and line_center_x < center_x
    
    def _is_right_lane(self, slope: float, line_center_x: int, center_x: int) -> bool:
        """Determine if line belongs to right lane"""
        return slope > 0 and line_center_x > center_x

class TemporalTracker:
    """Handles temporal tracking and smoothing of lane detections"""
    
    def __init__(self, params: LaneDetectionParams):
        self.params = params
        self.prev_left_coords = None
        self.prev_right_coords = None
        self.missing_left_count = 0
        self.missing_right_count = 0
        self.logger = logging.getLogger(__name__)
    
    def update_and_smooth(self, left_coords: Optional[LaneCoordinates], 
                         right_coords: Optional[LaneCoordinates]) -> Tuple[Optional[LaneCoordinates], Optional[LaneCoordinates]]:
        """Update temporal tracking and apply smoothing"""
        
        # Apply temporal smoothing
        left_coords = self._apply_smoothing(left_coords, self.prev_left_coords)
        right_coords = self._apply_smoothing(right_coords, self.prev_right_coords)
        
        # Handle missing detections
        left_coords = self._handle_missing_left(left_coords)
        right_coords = self._handle_missing_right(right_coords)
        
        return left_coords, right_coords
    
    def _apply_smoothing(self, current: Optional[LaneCoordinates], 
                        previous: Optional[LaneCoordinates]) -> Optional[LaneCoordinates]:
        """Apply exponential moving average smoothing"""
        if previous is None or current is None:
            return current
        
        # Apply weighted average
        smoothed_coords = (self.params.smoothing_factor * previous.to_array() + 
                          (1 - self.params.smoothing_factor) * current.to_array()).astype(int)
        
        return LaneCoordinates.from_array(smoothed_coords)
    
    def _handle_missing_left(self, left_coords: Optional[LaneCoordinates]) -> Optional[LaneCoordinates]:
        """Handle missing left lane detection"""
        if left_coords is None:
            self.missing_left_count += 1
            if self.missing_left_count < self.params.max_missing_frames:
                # Use previous coordinates
                left_coords = self.prev_left_coords
            else:
                # Too many misses, clear previous
                self.prev_left_coords = None
                left_coords = None
        else:
            # Good detection, reset counter and update previous
            self.missing_left_count = 0
            self.prev_left_coords = left_coords
        
        return left_coords
    
    def _handle_missing_right(self, right_coords: Optional[LaneCoordinates]) -> Optional[LaneCoordinates]:
        """Handle missing right lane detection"""
        if right_coords is None:
            self.missing_right_count += 1
            if self.missing_right_count < self.params.max_missing_frames:
                right_coords = self.prev_right_coords
            else:
                self.prev_right_coords = None
                right_coords = None
        else:
            self.missing_right_count = 0
            self.prev_right_coords = right_coords
        
        return right_coords

class LaneVisualizer:
    """Handles visualization of lane detection results"""
    
    def __init__(self, params: LaneDetectionParams):
        self.params = params
    
    def draw_results(self, image: np.ndarray, left_coords: Optional[LaneCoordinates], 
                    right_coords: Optional[LaneCoordinates]) -> np.ndarray:
        """Draw all enabled visualization elements"""
        result_image = image.copy()
        
        if self.params.display_lane_lines:
            self._draw_lane_lines(result_image, left_coords, right_coords)
        
        if left_coords and right_coords:
            if self.params.display_lane_overlay:
                self._draw_lane_area(result_image, left_coords, right_coords)
            if self.params.display_center_lane_line:
                self._draw_center_lane(result_image, left_coords, right_coords)
        
        return result_image
    
    def _draw_lane_lines(self, image: np.ndarray, left_coords: Optional[LaneCoordinates], 
                        right_coords: Optional[LaneCoordinates]) -> None:
        """Draw detected lane lines"""
        if left_coords:
            cv2.line(image, (left_coords.x1, left_coords.y1), (left_coords.x2, left_coords.y2), 
                    (0, 255, 0), 4)  # Green
        
        if right_coords:
            cv2.line(image, (right_coords.x1, right_coords.y1), (right_coords.x2, right_coords.y2), 
                    (0, 255, 0), 4)  # Green
    
    def _draw_lane_area(self, image: np.ndarray, left_coords: LaneCoordinates, 
                       right_coords: LaneCoordinates) -> None:
        """Draw semi-transparent lane area"""
        try:
            lane_area = np.array([[
                [left_coords.x1, left_coords.y1],   # Left bottom
                [left_coords.x2, left_coords.y2],   # Left top
                [right_coords.x2, right_coords.y2], # Right top
                [right_coords.x1, right_coords.y1]  # Right bottom
            ]], np.int32)
            
            overlay = image.copy()
            cv2.fillPoly(overlay, lane_area, (0, 255, 255))  # Yellow fill
            cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error drawing lane area: {e}")
    
    def _draw_center_lane(self, image: np.ndarray, left_coords: LaneCoordinates, 
                         right_coords: LaneCoordinates, color=(255, 255, 0), thickness=5) -> None:
        """Draw center lane line"""
        try:
            mid_bottom = ((left_coords.x1 + right_coords.x1) // 2, left_coords.y1)
            mid_top = ((left_coords.x2 + right_coords.x2) // 2, right_coords.y2)
            
            overlay = image.copy()
            cv2.line(overlay, mid_bottom, mid_top, color, thickness)
            cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error drawing center line: {e}")

class SimpleLaneDetector:
    """
    Lane detection system using classical computer vision techniques.
    
    Supports both camera and Bird's Eye View (BEV) processing modes with robust
    temporal tracking and configurable visualization options.
    """
    
    def __init__(self, config: dict):
        """Initialize the lane detection system"""
        self.logger = self._setup_logging()
        
        # Initialize components
        self.params = LaneDetectionParams(config)
        self.line_filter = LineFilter(self.params)
        self.temporal_tracker = TemporalTracker(self.params)
        self.visualizer = LaneVisualizer(self.params)
        
        # BEV transformer (if enabled)
        self.bev_transformer = None
        if self.params.bev_enabled:
            self.bev_transformer = BEVTransformer(config)
            self.logger.info("BEV transformation enabled")
        
        # Current detection results (in camera coordinates for display)
        self.left_coords = None
        self.right_coords = None
        
        self.logger.info("Lane detector initialized successfully")
    
    def process_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
                                                      Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Process input image and detect lane lines.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Tuple of (result_image, gray, edges, masked, left_coords, right_coords)
        """
        try:
            if self.params.img_height == self.params.img_height_original and self.params.img_width == self.params.img_width_original:
                # 1. Preprocess image
                frame = self._preprocess_image(image)
            else:
                frame = image
            
            # 2. Choose processing frame (BEV or camera)
            processing_frame = self._get_processing_frame(frame)
            
            # 3. Image processing pipeline
            gray = self._convert_to_grayscale(processing_frame)
            blur = self._apply_gaussian_blur(gray)
            edges = self._detect_edges(blur)
            masked = self._apply_roi(edges)
            
            # 4. Detect and process lane lines
            left_coords_proc, right_coords_proc = self._detect_lanes(masked, processing_frame)
            
            # 5. Apply temporal tracking and smoothing
            left_coords_proc, right_coords_proc = self.temporal_tracker.update_and_smooth(
                left_coords_proc, right_coords_proc)
            
            # 6. Transform coordinates for display
            left_coords_display, right_coords_display = self._prepare_display_coordinates(
                left_coords_proc, right_coords_proc)
            
            # 7. Generate result image
            result_image = self.visualizer.draw_results(frame, left_coords_display, right_coords_display)
            
            # 8. Update instance variables for external access
            self.left_coords = left_coords_display.to_array() if left_coords_display else None
            self.right_coords = right_coords_display.to_array() if right_coords_display else None
            
            return result_image, gray, edges, masked, self.left_coords, self.right_coords
            
        except Exception as e:
            self.logger.error(f"Error in process_image: {e}")
            return self._create_error_output(image)
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Resize input image to target dimensions"""
        return cv2.resize(image, (self.params.img_width, self.params.img_height))
    
    def _get_processing_frame(self, frame: np.ndarray) -> np.ndarray:
        """Get the frame to use for processing (BEV or camera)"""
        if self.params.bev_enabled and self.bev_transformer:
            return self.bev_transformer.warp(frame)
        return frame
    
    def _convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image.copy()
    
    def _apply_gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to reduce noise"""
        return cv2.GaussianBlur(gray, 
                               (self.params.gauss_kernel_x, self.params.gauss_kernel_y),
                               self.params.gauss_sigma_x)
    
    def _detect_edges(self, blur: np.ndarray) -> np.ndarray:
        """Detect edges using Canny edge detector"""
        return cv2.Canny(blur, self.params.canny_low, self.params.canny_high)
    
    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        """Apply region of interest using dual polygon approach"""
        height, width = edges.shape
        roi_mask = ROIGenerator.create_dual_lane_roi(width, height, self.params.bev_enabled)
        return cv2.bitwise_and(edges, roi_mask)
    
    def _detect_lanes(self, masked: np.ndarray, frame: np.ndarray) -> Tuple[Optional[LaneCoordinates], Optional[LaneCoordinates]]:
        """Detect lane lines using Hough line detection"""
        # Detect lines
        lines = cv2.HoughLinesP(
            masked,
            rho=self.params.hough_rho,
            theta=np.pi/180,
            threshold=self.params.hough_threshold,
            minLineLength=self.params.hough_min_line_len,
            maxLineGap=self.params.hough_max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return None, None
        
        # Filter and classify lines
        left_fits, right_fits = self.line_filter.filter_and_classify_lines(lines, frame.shape[1])
        
        # Convert to coordinates
        left_coords = self._fits_to_coordinates(frame, left_fits)
        right_coords = self._fits_to_coordinates(frame, right_fits)
        
        return left_coords, right_coords
    
    def _fits_to_coordinates(self, frame: np.ndarray, fits: List[Tuple[float, float]]) -> Optional[LaneCoordinates]:
        """Convert line fits to coordinate format"""
        if not fits:
            return None
        
        # Average multiple fits
        avg_slope = np.mean([fit[0] for fit in fits])
        avg_intercept = np.mean([fit[1] for fit in fits])
        
        # Convert to coordinates
        return self._line_params_to_coordinates(frame, avg_slope, avg_intercept)
    
    def _line_params_to_coordinates(self, frame: np.ndarray, slope: float, intercept: float) -> Optional[LaneCoordinates]:
        """Convert line parameters to coordinate format"""
        try:
            height, width = frame.shape[:2]
            
            # Define y coordinates
            y1 = height
            y2 = int(height * 0.6) if not self.params.bev_enabled else int(height * 0.2)
            
            # Calculate x coordinates
            if abs(slope) < 1e-6:  # Avoid division by zero
                return None
            
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            
            # Create coordinates object and validate
            coords = LaneCoordinates(
                max(0, min(width-1, x1)),y1,
                max(0, min(width-1, x2)),y2
            )
            
            return coords if coords else None
            
        except (ZeroDivisionError, ValueError, TypeError):
            return None
    
    def _prepare_display_coordinates(self, left_coords: Optional[LaneCoordinates], 
                                   right_coords: Optional[LaneCoordinates]) -> Tuple[Optional[LaneCoordinates], Optional[LaneCoordinates]]:
        """Transform coordinates to camera view for display"""
        if not self.params.bev_enabled or not self.bev_transformer:
            return left_coords, right_coords
        
        # Transform BEV coordinates back to camera coordinates
        left_display = self._transform_to_camera(left_coords)
        right_display = self._transform_to_camera(right_coords)
        
        return left_display, right_display
    
    def _transform_to_camera(self, coords: Optional[LaneCoordinates]) -> Optional[LaneCoordinates]:
        """Transform BEV coordinates to camera coordinates"""
        if coords is None or not self.bev_transformer:
            return coords
        
        try:
            # Pack coordinates for transformation
            pts = np.array([
                [coords.x1, coords.y1],
                [coords.x2, coords.y2]
            ], dtype=np.float32).reshape(-1, 1, 2)
            
            # Transform
            cam_pts = cv2.perspectiveTransform(pts, self.bev_transformer.inverse_transform_matrix)
            
            # Unpack and create new coordinates
            (x1c, y1c), (x2c, y2c) = cam_pts.reshape(2, 2)
            return LaneCoordinates(int(x1c), int(y1c), int(x2c), int(y2c))
            
        except Exception as e:
            self.logger.error(f"Error transforming coordinates: {e}")
            return None
    
    def _create_error_output(self, image: np.ndarray) -> Tuple:
        """Create consistent error output"""
        frame = cv2.resize(image, (self.params.img_width, self.params.img_height))
        empty_debug = np.zeros((self.params.img_height, self.params.img_width), dtype=np.uint8)
        return frame, empty_debug, empty_debug, empty_debug, None, None
    
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
    
    # Public API methods for external access
    def get_lane_coordinates(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current lane coordinates in camera view"""
        return self.left_coords, self.right_coords
    
    def get_detection_status(self) -> dict:
        """Get current detection status information"""
        return {
            'left_detected': self.left_coords is not None,
            'right_detected': self.right_coords is not None,
            'missing_left_count': self.temporal_tracker.missing_left_count,
            'missing_right_count': self.temporal_tracker.missing_right_count,
            'bev_enabled': self.params.bev_enabled
        }
    
    def debug_pipeline(self, image: np.ndarray) -> dict:
        """Debug each step of the pipeline"""
        debug_info = {}
        
        try:
            # Step 1: Preprocessing
            frame = self._preprocess_image(image)
            debug_info['frame_shape'] = frame.shape
            
            # Step 2: Processing frame
            processing_frame = self._get_processing_frame(frame)
            debug_info['processing_frame_shape'] = processing_frame.shape
            debug_info['bev_enabled'] = self.params.bev_enabled
            
            # Step 3: Grayscale
            gray = self._convert_to_grayscale(processing_frame)
            debug_info['gray_shape'] = gray.shape
            debug_info['gray_mean'] = np.mean(gray)
            
            # Step 4: Blur
            blur = self._apply_gaussian_blur(gray)
            debug_info['blur_mean'] = np.mean(blur)
            
            # Step 5: Edges
            edges = self._detect_edges(blur)
            debug_info['edges_pixels'] = np.sum(edges > 0)
            debug_info['edges_percentage'] = (np.sum(edges > 0) / edges.size) * 100
            
            # Step 6: ROI
            masked = self._apply_roi(edges)
            debug_info['masked_pixels'] = np.sum(masked > 0)
            debug_info['roi_percentage'] = (np.sum(masked > 0) / np.sum(edges > 0)) * 100 if np.sum(edges > 0) > 0 else 0
            
            # Step 7: Hough lines
            lines = cv2.HoughLinesP(
                masked,
                rho=self.params.hough_rho,
                theta=np.pi/180,
                threshold=self.params.hough_threshold,
                minLineLength=self.params.hough_min_line_len,
                maxLineGap=self.params.hough_max_line_gap
            )
            
            debug_info['hough_lines_detected'] = len(lines) if lines is not None else 0
            
            if lines is not None and len(lines) > 0:
                # Step 8: Line filtering
                left_fits, right_fits = self.line_filter.filter_and_classify_lines(lines, processing_frame.shape[1])
                debug_info['left_fits'] = len(left_fits)
                debug_info['right_fits'] = len(right_fits)
            else:
                debug_info['left_fits'] = 0
                debug_info['right_fits'] = 0
                
            return debug_info
            
        except Exception as e:
            debug_info['error'] = str(e)
            return debug_info