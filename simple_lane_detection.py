import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('SVG')

class SimpleLaneDetector:
    """SimpleLaneDetector class that uses classic CV techniques to obtain lane lines."""
    def __init__(self, image_size):
        self.img_size = image_size
        self.prev_left_coords = None
        self.prev_right_coords = None
        self.missing_left = 0
        self.missing_right = 0
        self.smoothing_factor = 0.7  # For temporal smoothing

    def average_slope_intercept(self, frame, lines):
        left_fit = []
        right_fit = []

        if lines is None:
            return [None, None]

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            line_parameters = np.polyfit([x1, x2], [y1, y2], 1)
            slope, intercept = line_parameters
            
            # Filter out lines with slopes that are too small (nearly horizontal)
            if abs(slope) < 0.3:
                continue
                
            # Filter out lines with extreme slopes
            if abs(slope) > 3.0:
                continue
                
            # Classify left and right lanes based on slope and position
            center_x = frame.shape[1] // 2
            line_center_x = (x1 + x2) // 2
            
            if slope < 0 and line_center_x < center_x:  # Left lane (negative slope, left side)
                left_fit.append((slope, intercept))
            elif slope > 0 and line_center_x > center_x:  # Right lane (positive slope, right side)
                right_fit.append((slope, intercept))

        # Calculate averages only if we have lines
        left_coords = None
        right_coords = None
        
        if len(left_fit) > 0:
            left_avg = np.average(left_fit, axis=0)
            left_coords = self.make_coordinates(frame, left_avg)
            
        if len(right_fit) > 0:
            right_avg = np.average(right_fit, axis=0)
            right_coords = self.make_coordinates(frame, right_avg)
            
        return [left_coords, right_coords]

    def make_coordinates(self, frame, line_params):
        try:
            slope, intercept = line_params
            height = frame.shape[0]
            width = frame.shape[1]
            
            # Define y coordinates
            y1 = height              # Bottom of image
            y2 = int(height * 0.6)   # 60% up from bottom
            
            # Calculate x coordinates
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            
            # Clamp coordinates to image boundaries
            x1 = max(0, min(width-1, x1))
            x2 = max(0, min(width-1, x2))
            
            # Ensure coordinates are valid
            if x1 == x2:  # Vertical line
                return None
                
            return np.array([x1, y1, x2, y2])
            
        except (ZeroDivisionError, ValueError, TypeError):
            return None

    def smooth_lines(self, current_coords, prev_coords):
        """Apply temporal smoothing to reduce jitter"""
        if prev_coords is None or current_coords is None:
            return current_coords
            
        # Weighted average between current and previous coordinates
        smoothed = (self.smoothing_factor * prev_coords + 
                   (1 - self.smoothing_factor) * current_coords).astype(int)
        return smoothed

    def process_image(self, image):
        """Main processing pipeline"""
        try:
            # Resize image to specified size
            frame = cv2.resize(image, self.img_size)
            
            # Convert to grayscale (handle both RGB and BGR)
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame.copy()
                
            # Apply Gaussian blur to reduce noise
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            
            # Edge detection with adaptive thresholds
            edges = cv2.Canny(blur, 50, 150, apertureSize=3)

            # Define region of interest (trapezoid)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            
            # Improved polygon for better lane detection
            left_polygon = np.array([[
                (0, height),                            # Bottom left of the image
                (0, int(height * 0.8)),                 # Bottom left and a bit up
                (int(width * 0.4), int(height * 0.5)),  # Almost middle of the screen
                (int(width * 0.5), int(height * 0.5)),  
                (int(width * 0.2), height)              # Bottom left and a bit to the right
            ]], np.int32)

            right_polygon = np.array([[
                (width, height),                        # Bottom right
                (width, int(height * 0.8)),             # Bottom right and a bit up
                (int(width * 0.6), int(height * 0.5)),  # Almost middle of screen
                (int(width * 0.5), int(height * 0.5)),
                (int(width * 0.8), height)              # Bottom right and a bit to the left
            ]], np.int32)

            cv2.fillPoly(mask, left_polygon, 255)
            cv2.fillPoly(mask, right_polygon, 255)
            masked = cv2.bitwise_and(edges, mask)

            # Uncomment to freeze carla and see the polygon mask
            plt.imshow(mask)
            plt.show()

            # Hough line detection with optimized parameters
            lines = cv2.HoughLinesP(
                masked, 
                rho=1,                    # Distance resolution in pixels
                theta=np.pi / 180,        # Angle resolution in radians
                threshold=30,             # Minimum votes
                minLineLength=40,         # Minimum line length
                maxLineGap=100            # Maximum gap between line segments
            )

            # Process and draw detected lanes
            lane_image = frame.copy()
            
            if lines is not None and len(lines) > 0:
                left_coords, right_coords = self.average_slope_intercept(frame, lines)
                
                # Apply temporal smoothing
                left_coords = self.smooth_lines(left_coords, self.prev_left_coords)
                right_coords = self.smooth_lines(right_coords, self.prev_right_coords)

                if left_coords is None:
                    self.missing_left += 1
                    if self.missing_left < 50: # This number repersents the number of frames the line will be kept
                        left_coords = self.prev_left_coords
                else:
                    # Update previous coordinates
                    self.missing_left = 0
                    self.prev_left_coords = left_coords
                
                if right_coords is None:
                    self.missing_right += 1
                    if self.missing_right < 50: # This number repersents the number of frames the line will be kept
                        left_coords = self.prev_left_coords
                else:
                    # Update previous coordinates
                    self.missing_right = 0
                    self.prev_right_coords = right_coords
                
                # Draw lanes with different colors
                if left_coords is not None:
                    x1, y1, x2, y2 = left_coords
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 200, 0), 4)  # Darker green outline
                    
                if right_coords is not None:
                    x1, y1, x2, y2 = right_coords
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 5)  # Green
                    cv2.line(lane_image, (x1, y1), (x2, y2), (0, 200, 0), 4)  # Darker green outline
                
                # Draw lane area if both lanes detected
                if left_coords is not None and right_coords is not None:
                    self.draw_lane_area(lane_image, left_coords, right_coords)

            return lane_image, gray, edges, masked
            
        except Exception as e:
            print(f"Error in process_image: {e}")
            # Return original frame with empty debug images
            return frame, np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0])

    def draw_lane_area(self, image, left_coords, right_coords):
        """Draw the lane area between detected lanes"""
        try:
            # Create points for the lane area
            left_x1, left_y1, left_x2, left_y2 = left_coords
            right_x1, right_y1, right_x2, right_y2 = right_coords
            
            # Define the lane area polygon
            lane_area = np.array([[
                [left_x1, left_y1],   # Left bottom
                [left_x2, left_y2],   # Left top
                [right_x2, right_y2], # Right top
                [right_x1, right_y1]  # Right bottom
            ]], np.int32)
            
            # Create overlay for semi-transparent effect
            overlay = image.copy()
            cv2.fillPoly(overlay, lane_area, (0, 255, 255))  # Yellow fill
            cv2.addWeighted(image, 0.8, overlay, 0.2, 0, image)
            
        except Exception as e:
            print(f"Error drawing lane area: {e}")