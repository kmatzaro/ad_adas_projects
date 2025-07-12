import cv2
import numpy as np

class BEVTransformer:
    def __init__(self, src_pts: np.ndarray, dst_size: tuple):
        """
        src_pts: 4x2 array in image coordinates,
        dst_size: (width, height) of the BEV image
        """
        w, h = dst_size
        dst_pts = np.float32([
            [0,   h],   
            [w,   h],   
            [w,   0],   
            [0,   0]
        ], dtype=np.float32)
        self.M   = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.Minv= cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.dst_size = dst_size

    def warp(self, image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(image, self.M, self.dst_size)

    def unwarp(self, bev_image: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(bev_image, self.Minv, 
                                   (bev_image.shape[1], bev_image.shape[0]))