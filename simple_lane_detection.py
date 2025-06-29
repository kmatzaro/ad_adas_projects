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
            parameters = np.polyfit([x1, x2], [y1, y2], 1)
            slope, intercept = parameters
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_avg = np.average(left_fit, axis=0)
        right_avg = np.average(right_fit, axis=0)

        left_coords = self.make_coordinates(frame, left_avg)
        right_coords = self.make_coordinates(frame, right_avg)

        return [left_coords, right_coords]

    def make_coordinates(self, frame, line_params):
        slope, intercept = line_params
        y1 = frame.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def process_image(self, image):
        frame = cv2.resize(image, self.img_size)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height), (width, height),
            (int(width * 0.55), int(height * 0.6)),
            (int(width * 0.45), int(height * 0.6))
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, 2, np.pi / 180, threshold=50,
                                minLineLength=40, maxLineGap=5)
        if lines is not None:
            fitted_lines = self.average_slope_intercept(frame, lines)
            for line in fitted_lines:
                x1, y1, x2, y2 = line
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        return frame, gray, edges, masked