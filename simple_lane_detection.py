import numpy as np
import cv2

class SimpleLaneDetector:
    def __init__(self, image_size):
        self.img_size = image_size

    def average_slope_intercept(self, frame, lines):
        left_fit = []
        right_fit = []

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            line_parameters = np.polyfit([x1, x2], [y1, y2], 1)
            slope, intercept = line_parameters
            if abs(slope) < 0.3:
                pass
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_avg = np.average(left_fit, axis=0)
        right_avg = np.average(right_fit, axis=0)
        
        if left_avg.all():
            left_coords = self.make_coordinates(frame, left_avg)
        else:
            left_coords = []
        
        if right_avg.all():
            right_coords = self.make_coordinates(frame, right_avg)
        else:
            right_coords = []
        
        return [left_coords, right_coords]

    def make_coordinates(self, frame, line_params):
        try:
            slope, intercept = line_params
            y1 = frame.shape[0]
            y2 = int(y1 * 0.7)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return np.array([x1, y1, x2, y2])
        except Exception:
            return np.array([0,0,0,0])
        

    def process_image(self, image):
        frame = cv2.resize(image, self.img_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width, height),
            (int(width * 0.6), int(height * 0.55)),
            (int(width * 0.6), int(height * 0.55))
        ]], np.int32)
        mask = cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi / 180, threshold=100,
                                minLineLength=40, maxLineGap=150)

        if lines is not None:
            left_fitted_lines, right_fitted_lines = self.average_slope_intercept(frame, lines)
            if left_fitted_lines.all() != 0:
                x1, y1, x2, y2 = left_fitted_lines
                try:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                except OverflowError:
                    pass
            if right_fitted_lines.all() != 0:
                x1, y1, x2, y2 = right_fitted_lines
                try:
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                except OverflowError:
                    pass
                

        return frame, gray, edges, masked