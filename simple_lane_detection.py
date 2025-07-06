import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

class SimpleLaneDetector:
    def __init__(self, image_size, display_lane_overlay = True, display_lane_lines = True, display_center_lane_line = True):
        """SimpleLaneDetector class that uses classic CV techniques to obtain lane lines.
    
        Paramters:
            img_size                        : Size of the image to process 
            display_lane_overlay (bool)     : Whether to display the lane overlay 
            display_lane_lines (bool)       : Whether to display the lane lines
            display_center_lane_line (bool) : Whether to display the projected center line
            prev_left_coords                : Coordinates of the left detected line in the previous frame
            prev_right_coords               : Coordinates of the right detected line in the previous frame
            missing_left                    : Number of misses for detecting a left lane line
            missing_right                   : Number of misses for detecting a right lane line
            smoothing_factor                : Smoothing factor for the EMA (Exponential Moving Average)
        """
        
        self.img_size = image_size
        self.display_lane_overlay = display_lane_overlay
        self.display_lane_lines = display_lane_lines
        self.display_center_lane_line = display_center_lane_line
        self.prev_left_coords = None
        self.prev_right_coords = None
        self.left_coords = None
        self.right_coords = None
        self.missing_left = 0
        self.missing_right = 0
        self.smoothing_factor = 0.9  # For temporal smoothing

    def average_slope_intercept(self, frame, lines):
        left_fit = []
        right_fit = []

        self.left_coords, self.right_coords = None, None

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

        
        if len(left_fit) > 0:
            left_avg = np.average(left_fit, axis=0)
            self.left_coords = self.make_coordinates(frame, left_avg)
            
        if len(right_fit) > 0:
            right_avg = np.average(right_fit, axis=0)
            self.right_coords = self.make_coordinates(frame, right_avg)
            
        return [self.left_coords, self.right_coords]

    def make_coordinates(self, frame, line_params):
        """
        Given the frame and the fitted line parameters (slope, intercept) determine 
        two coordinate points (x1, y1) and (x2, y2) that lie along the fitted line
        """
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
            blur = cv2.GaussianBlur(gray, (3, 3), 5)
            
            # Edge detection with adaptive thresholds
            edges = cv2.Canny(blur, 50, 150, apertureSize=3)

            # Define region of interest (trapezoid)
            height, width = edges.shape
            mask = np.zeros_like(edges)
            
            # Improved polygon for better lane detection
            left_polygon = np.array([[
                (0, height),                            # Bottom left of the image
                (0, int(height * 0.9)),                 # Bottom left and a bit up
                (int(width * 0.45), int(height * 0.5)),  # Almost middle of the screen
                (int(width * 0.5), int(height * 0.5)),  
                (int(width * 0.1), height)              # Bottom left and a bit to the right
            ]], np.int32)

            right_polygon = np.array([[
                (width, height),                        # Bottom right
                (width, int(height * 0.9)),             # Bottom right and a bit up
                (int(width * 0.55), int(height * 0.5)),  # Almost middle of screen
                (int(width * 0.5), int(height * 0.5)),
                (int(width * 0.9), height)              # Bottom right and a bit to the left
            ]], np.int32)

            cv2.fillPoly(mask, left_polygon, 255)
            cv2.fillPoly(mask, right_polygon, 255)
            masked = cv2.bitwise_and(edges, mask)

            # Uncomment to freeze carla and see the polygon mask
            # plt.imshow(mask)
            # plt.show()

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

            self.left_coords, self.right_coords = None, None
            
            if lines is not None and len(lines) > 0:
                self.left_coords, self.right_coords = self.average_slope_intercept(frame, lines)
                
                # Apply temporal smoothing
                self.left_coords = self.smooth_lines(self.left_coords, self.prev_left_coords)
                self.right_coords = self.smooth_lines(self.right_coords, self.prev_right_coords)

                if self.left_coords is None:
                    self.missing_left += 1
                    if self.missing_left < 20:
                        self.left_coords = self.prev_left_coords
                    else:
                        # after 50 bad frames, clear everything
                        self.prev_left_coords = None
                        self.left_coords      = None
                else:
                    self.missing_left      = 0
                    self.prev_left_coords  = self.left_coords
                
                if self.right_coords is None:
                    self.missing_right += 1
                    if self.missing_right < 20: # This number repersents the number of frames the line will be kept
                        self.right_coords = self.prev_right_coords
                    else:
                        # after 50 bad frames, clear everything
                        self.prev_right_coords = None
                        self.right_coords      = None
                else:
                    # Update previous coordinates
                    self.missing_right = 0
                    self.prev_right_coords = self.right_coords
                
                # Draw lane area/line/center lane line
                if self.display_lane_lines:
                    self.draw_lane_lines(lane_image, self.left_coords, self.right_coords)
                if self.left_coords is not None and self.right_coords is not None:
                    if self.display_lane_overlay:
                        self.draw_lane_area(lane_image, self.left_coords, self.right_coords)
                    if self.display_center_lane_line:
                        self.draw_center_lane(lane_image, self.left_coords, self.right_coords)

            return lane_image, gray, edges, masked, self.left_coords, self.right_coords
            
        except Exception as e:
            print(f"Error in process_image: {e}")
            # Return original frame with empty debug images
            return frame, np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0]), np.zeros_like(frame[:,:,0])
        
    def draw_lane_lines(self, image, left_coords, right_coords):
        """Draw lanes detected lane"""
        if left_coords is not None:
            x1, y1, x2, y2 = left_coords
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green
            
        if right_coords is not None:
            x1, y1, x2, y2 = right_coords
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 4)  # Green


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
    
    def draw_center_lane(self, frame, left_coords, right_coords, color=(0, 255, 0), thickness=5):
        """Draw the projected center lane line between detected lanes"""
        try:
            if self.left_coords is not None and self.right_coords is not None:
                # Create points for the lane line
                left_x1, left_y1, left_x2, left_y2 = left_coords
                right_x1, right_y1, right_x2, right_y2 = right_coords

                mid_bottom = ((left_x1 + right_x1)//2, left_y1)
                mid_top    = ((left_x2 + right_x2)//2, right_y2)
                
                overlay = frame.copy()
                cv2.line(overlay, mid_bottom, mid_top, color, thickness)
                cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        except Exception as e:
            print(f"Error drawing center line: {e}")