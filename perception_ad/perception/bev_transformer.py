import cv2
import numpy as np
import logging
from typing import Union
from dataclasses import dataclass

@dataclass
class TransformationQuality:
    """Data class to assess the quality of perspective transformation"""
    determinant: float
    condition_number: float
    is_numerically_stable: bool
    quality_score: str  # 'excellent', 'good', 'poor', 'invalid'

class BEVTransformer:
    """
    Bird's Eye View Perspective Transformer for Autonomous Driving Applications
    
    This class implements robust perspective transformation for converting camera images
    to bird's eye view, which is essential for accurate lane detection and path planning
    in autonomous driving systems.
    
    Key Features:
    - Robust parameter validation and error handling
    - Numerical stability assessment of transformation matrices
    - Comprehensive logging for debugging and analysis
    - Point coordinate transformation capabilities
    - Visualization tools for calibration verification
    
    Mathematical Foundation:
    Uses homogeneous coordinates and perspective transformation matrices to map
    points from camera coordinates (u,v) to BEV coordinates (x,y) using:
    [x', y', w'] = M * [u, v, 1]
    where (x,y) = (x'/w', y'/w')
    """
    
    def __init__(self, config: dict):
        """
        Initialize BEV transformer with configuration validation.
        
        Args:
            config: Configuration dictionary containing:
                - lane_detector.image_resize: Input image dimensions
                - lane_detector.bev_lane_detector.bev_size: Output BEV dimensions  
                - lane_detector.bev_lane_detector.src_pts_frac: Source points as fractions [0,1]
                
        Raises:
            ValueError: If configuration parameters are invalid
            RuntimeError: If transformation matrices cannot be computed
        """
        self.logger = self._setup_logging()
        
        try:
            # Extract and validate configuration parameters
            self._parse_configuration(config)
            self._validate_parameters()
            
            # Compute perspective transformation matrices
            self._compute_transformation_matrices()
            
            # Assess transformation quality for debugging
            self.quality_metrics = self._assess_transformation_quality()
            
            # Cache output dimensions for performance
            self.output_size = (self.bev_width, self.bev_height)
            
            self.logger.info(
                f"BEV Transformer initialized successfully: "
                f"{self.image_width}×{self.image_height} → {self.bev_width}×{self.bev_height} "
                f"(Quality: {self.quality_metrics.quality_score})"
            )
            
            # Warn about potential quality issues
            if self.quality_metrics.quality_score in ['poor', 'invalid']:
                self.logger.warning(f"Transformation quality assessment: {self.quality_metrics.quality_score}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize BEV transformer: {e}")
            raise
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for the transformer with appropriate format."""
        logger = logging.getLogger(f"{__name__}.BEVTransformer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _parse_configuration(self, config: dict) -> None:
        """
        Extract configuration parameters with proper error handling.
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If required configuration keys are missing
        """
        try:
            bev_config = config['lane_detector']['bev_lane_detector']
            resize_config = config['lane_detector']['image_resize']
            
            # Input image dimensions
            self.image_width = resize_config['image_width']
            self.image_height = resize_config['image_height']
            
            # BEV output dimensions
            self.bev_width = bev_config['bev_size']['bev_width']
            self.bev_height = bev_config['bev_size']['bev_height']
            
            # Source points in fractional coordinates [0,1]
            self.src_points_fractional = bev_config['src_pts_frac']
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration parameter: {e}")
    
    def _validate_parameters(self) -> None:
        """
        Validate all configuration parameters for correctness.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        # Validate image dimensions
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError(
                f"Invalid input dimensions: {self.image_width}×{self.image_height}. "
                "Dimensions must be positive integers."
            )
        
        if self.bev_width <= 0 or self.bev_height <= 0:
            raise ValueError(
                f"Invalid BEV dimensions: {self.bev_width}×{self.bev_height}. "
                "Dimensions must be positive integers."
            )
        
        # Validate source points
        if len(self.src_points_fractional) != 4:
            raise ValueError(
                f"Expected exactly 4 source points, got {len(self.src_points_fractional)}. "
                "Source points should define a quadrilateral."
            )
        
        # Validate fractional coordinates
        for i, (u, v) in enumerate(self.src_points_fractional):
            if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                raise ValueError(
                    f"Source point {i} coordinates ({u:.3f}, {v:.3f}) outside valid range [0,1]. "
                    "Fractional coordinates must be between 0 and 1."
                )
        
        # Check for degenerate cases
        self._validate_quadrilateral_geometry()
    
    def _validate_quadrilateral_geometry(self) -> None:
        """
        Ensure source points form a valid, non-degenerate quadrilateral.
        
        Raises:
            ValueError: If quadrilateral is degenerate or invalid
        """
        # Convert to pixel coordinates for geometric validation
        src_pixels = np.array([
            [u * self.image_width, v * self.image_height] 
            for u, v in self.src_points_fractional
        ], dtype=np.float32)
        
        # Check for duplicate or nearly coincident points
        min_distance_threshold = 10.0  # pixels
        for i in range(len(src_pixels)):
            for j in range(i + 1, len(src_pixels)):
                distance = np.linalg.norm(src_pixels[i] - src_pixels[j])
                if distance < min_distance_threshold:
                    raise ValueError(
                        f"Source points {i} and {j} are too close ({distance:.1f} pixels). "
                        f"Minimum separation required: {min_distance_threshold} pixels."
                    )
        
        # Compute quadrilateral area to check for degeneracy
        area = self._compute_quadrilateral_area(src_pixels)
        min_area = 0.01 * self.image_width * self.image_height  # 1% of image area
        
        if area < min_area:
            raise ValueError(
                f"Source quadrilateral area ({area:.0f} px²) too small. "
                f"Minimum area required: {min_area:.0f} px² (1% of image)."
            )
    
    def _compute_quadrilateral_area(self, points: np.ndarray) -> float:
        """
        Compute area of quadrilateral using shoelace formula.
        
        Args:
            points: Array of 4 points defining quadrilateral
            
        Returns:
            Area of quadrilateral in square pixels
        """
        # Shoelace formula for polygon area
        x = points[:, 0]
        y = points[:, 1]
        return 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
    
    def _compute_transformation_matrices(self) -> None:
        """
        Compute forward and inverse perspective transformation matrices.
        
        Raises:
            RuntimeError: If OpenCV fails to compute transformation matrices
        """
        try:
            # Define destination points (BEV coordinate system)
            # Convention: origin at top-left, y-axis pointing down
            dst_points = np.array([
                [0, self.bev_height],           # Bottom-left
                [self.bev_width, self.bev_height],  # Bottom-right  
                [self.bev_width, 0],            # Top-right
                [0, 0]                          # Top-left
            ], dtype=np.float32)
            
            # Convert fractional source points to pixel coordinates
            src_points = np.array([
                [u * self.image_width, v * self.image_height] 
                for u, v in self.src_points_fractional
            ], dtype=np.float32)
            
            # Compute perspective transformation matrices
            self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
            
            # Store point correspondences for debugging
            self._src_points_pixels = src_points
            self._dst_points_pixels = dst_points
            
            self.logger.debug("Perspective transformation matrices computed successfully")
            
        except cv2.error as e:
            raise RuntimeError(f"OpenCV failed to compute perspective transformation: {e}")
    
    def _assess_transformation_quality(self) -> TransformationQuality:
        """
        Assess the numerical quality and stability of the computed transformation.
        
        Returns:
            TransformationQuality object with assessment metrics
        """
        try:
            # Compute matrix determinant (measure of area scaling)
            det = np.linalg.det(self.transform_matrix[:2, :2])
            
            # Compute condition number (measure of numerical stability)
            # Lower values indicate better numerical stability
            cond = np.linalg.cond(self.transform_matrix)
            
            # Assess numerical stability
            is_stable = abs(det) > 1e-8 and cond < 1e8
            
            # Determine quality score based on metrics
            if abs(det) < 1e-8:
                quality = 'invalid'
            elif cond > 1e6:
                quality = 'poor'
            elif cond > 1e4:
                quality = 'good'
            else:
                quality = 'excellent'
            
            return TransformationQuality(
                determinant=det,
                condition_number=cond,
                is_numerically_stable=is_stable,
                quality_score=quality
            )
            
        except Exception as e:
            self.logger.error(f"Failed to assess transformation quality: {e}")
            return TransformationQuality(
                determinant=0.0,
                condition_number=float('inf'),
                is_numerically_stable=False,
                quality_score='invalid'
            )
    
    def warp(self, image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Transform camera image to bird's eye view.
        
        Args:
            image: Input camera image (height x width x channels)
            interpolation: Interpolation method for resampling
                - cv2.INTER_LINEAR: Good quality, moderate speed (default)
                - cv2.INTER_CUBIC: Best quality, slower
                - cv2.INTER_NEAREST: Fastest, lower quality
                
        Returns:
            BEV image with dimensions (bev_height x bev_width x channels)
            
        Raises:
            ValueError: If input image has incorrect dimensions
            RuntimeError: If transformation fails
            
        Example:
            bev_image = transformer.warp(camera_image)
        """
        self._validate_input_image(image)
        
        try:
            return cv2.warpPerspective(
                image, 
                self.transform_matrix, 
                self.output_size,
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except cv2.error as e:
            raise RuntimeError(f"Image transformation to BEV failed: {e}")
    
    def unwarp(self, bev_image: np.ndarray, interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Transform bird's eye view image back to camera perspective.
        
        Args:
            bev_image: BEV image to transform back
            interpolation: Interpolation method for resampling
            
        Returns:
            Image in camera perspective
            
        Raises:
            ValueError: If BEV image has incorrect dimensions
            RuntimeError: If inverse transformation fails
            
        Example:
            camera_image = transformer.unwarp(bev_image)
        """
        self._validate_bev_image(bev_image)
        
        try:
            return cv2.warpPerspective(
                bev_image,
                self.inverse_transform_matrix,
                (self.image_width, self.image_height),
                flags=interpolation,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except cv2.error as e:
            raise RuntimeError(f"Image transformation from BEV failed: {e}")
    
    def transform_points_to_bev(self, camera_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transform point coordinates from camera to BEV coordinate system.
        
        This is essential for converting detected lane line coordinates from camera
        space to the more stable BEV space for processing.
        
        Args:
            camera_points: Points in camera coordinates
                - Single point: [x, y] or (x, y)
                - Multiple points: [[x1, y1], [x2, y2], ...] or Nx2 array
                
        Returns:
            Points in BEV coordinates (same format as input)
            
        Raises:
            ValueError: If input points have invalid format
            RuntimeError: If transformation fails
        """
        points_array = self._normalize_points_input(camera_points)
        
        try:
            # Reshape for OpenCV perspective transformation (requires Nx1x2 format)
            points_reshaped = points_array.reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(points_reshaped, self.transform_matrix)
            return transformed.reshape(-1, 2)
            
        except cv2.error as e:
            raise RuntimeError(f"Point transformation to BEV failed: {e}")
    
    def transform_points_to_camera(self, bev_points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Transform point coordinates from BEV to camera coordinate system.
        
        Useful for visualizing BEV-space results back in the original camera image.
        
        Args:
            bev_points: Points in BEV coordinates
            
        Returns:
            Points in camera coordinates
            
        Raises:
            ValueError: If input points have invalid format
            RuntimeError: If inverse transformation fails
        """
        points_array = self._normalize_points_input(bev_points)
        
        try:
            points_reshaped = points_array.reshape(-1, 1, 2).astype(np.float32)
            transformed = cv2.perspectiveTransform(points_reshaped, self.inverse_transform_matrix)
            return transformed.reshape(-1, 2)
            
        except cv2.error as e:
            raise RuntimeError(f"Point transformation to camera failed: {e}")
    
    def visualize_transformation_region(self, image: np.ndarray) -> np.ndarray:
        """
        Create visualization showing the source region used for BEV transformation.
        
        This is invaluable for debugging and verifying that the transformation
        region covers the appropriate road area.
        
        Args:
            image: Camera image to annotate
            
        Returns:
            Annotated image showing transformation region
            
        Example:
            annotated_img = transformer.visualize_transformation_region(camera_image)
            cv2.imshow("BEV Source Region", annotated_img)
        """
        self._validate_input_image(image)
        
        visualization = image.copy()
        
        # Draw transformation quadrilateral
        points = self._src_points_pixels.astype(np.int32)
        cv2.polylines(visualization, [points], isClosed=True, color=(0, 255, 0), thickness=3)
        
        # Mark corner points with different colors
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        corner_labels = ['BL', 'BR', 'TR', 'TL']  # Bottom-Left, Bottom-Right, etc.
        
        for i, (point, color, label) in enumerate(zip(points, corner_colors, corner_labels)):
            # Draw corner marker
            cv2.circle(visualization, tuple(point), radius=8, color=color, thickness=-1)
            
            # Add text label
            text_position = (point[0] - 15, point[1] - 15)
            cv2.putText(
                visualization, label, text_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        return visualization
    
    def get_transformation_info(self) -> dict:
        """
        Get comprehensive information about the transformation for debugging.
        
        Returns:
            Dictionary containing transformation parameters and quality metrics
        """
        return {
            'input_dimensions': (self.image_width, self.image_height),
            'output_dimensions': (self.bev_width, self.bev_height),
            'source_points_fractional': self.src_points_fractional,
            'source_points_pixels': self._src_points_pixels.tolist(),
            'quality_assessment': {
                'determinant': self.quality_metrics.determinant,
                'condition_number': self.quality_metrics.condition_number,
                'is_numerically_stable': self.quality_metrics.is_numerically_stable,
                'quality_score': self.quality_metrics.quality_score
            }
        }
    
    def _validate_input_image(self, image: np.ndarray) -> None:
        """Validate that input image has correct dimensions."""
        if len(image.shape) < 2:
            raise ValueError("Input image must be at least 2-dimensional")
        
        height, width = image.shape[:2]
        if height != self.image_height or width != self.image_width:
            raise ValueError(
                f"Input image size ({width}×{height}) doesn't match "
                f"expected size ({self.image_width}×{self.image_height})"
            )
    
    def _validate_bev_image(self, image: np.ndarray) -> None:
        """Validate that BEV image has correct dimensions."""
        if len(image.shape) < 2:
            raise ValueError("BEV image must be at least 2-dimensional")
        
        height, width = image.shape[:2]
        if height != self.bev_height or width != self.bev_width:
            raise ValueError(
                f"BEV image size ({width}×{height}) doesn't match "
                f"expected size ({self.bev_width}×{self.bev_height})"
            )
    
    def _normalize_points_input(self, points: Union[np.ndarray, list]) -> np.ndarray:
        """
        Normalize various point input formats to consistent numpy array format.
        
        Args:
            points: Points in various formats
            
        Returns:
            Normalized points as Nx2 numpy array
            
        Raises:
            ValueError: If points format is invalid
        """
        points_array = np.asarray(points, dtype=np.float32)
        
        if points_array.size == 0:
            raise ValueError("Points array cannot be empty")
        
        # Handle different input formats
        if points_array.ndim == 1:
            if len(points_array) != 2:
                raise ValueError("Single point must have exactly 2 coordinates [x, y]")
            points_array = points_array.reshape(1, 2)
        elif points_array.ndim == 2:
            if points_array.shape[1] != 2:
                raise ValueError("Points array must have shape (N, 2) for N points")
        else:
            raise ValueError("Points must be 1D (single point) or 2D (multiple points)")
        
        return points_array