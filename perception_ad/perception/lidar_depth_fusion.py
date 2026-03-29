import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from perception.object_detection import DetectedObject


class LiDARDepthFusion:
    """
    Projects LiDAR points into the front-camera image plane and assigns
    a median depth estimate to each detected object's bounding box.

    Math overview
    -------------
    1. Build camera intrinsic matrix K from FOV and image dimensions.
    2. Build a 4×4 rigid-body extrinsic matrix T_lidar_to_cam by composing
       the sensor-to-vehicle transforms from config:
           T_L2C = inv(T_cam_to_veh) @ T_lidar_to_veh
    3. For each LiDAR point P in sensor frame:
           P_cam = T_L2C @ [x, y, z, 1]^T
       Keep only points with P_cam[2] > 0.5 m (in front of camera, skip car body).
       Project: u = fx * P_cam[0]/P_cam[2] + cx
                v = fy * P_cam[1]/P_cam[2] + cy
    4. For each bounding box, collect all projected (u, v, depth) points
       that fall inside it and set distance_estimate = median(depths).
       Median is used to be robust against noise from background/foreground leakage.
    """

    def __init__(self, config: Dict):
        cam_cfg   = config['sensors']['front_camera']
        lidar_cfg = config['sensors']['lidar']

        img_w = cam_cfg['image_size_x']
        img_h = cam_cfg['image_size_y']
        fov_deg = cam_cfg['fov']

        # Intrinsic matrix
        fx = (img_w / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        fy = fx
        cx = img_w / 2.0
        cy = img_h / 2.0
        self.K = np.array([[fx,  0, cx],
                           [ 0, fy, cy],
                           [ 0,  0,  1]], dtype=np.float64)
        self.img_w = img_w
        self.img_h = img_h

        # Extrinsic: LiDAR sensor frame → camera sensor frame
        self._T_lidar_to_cam = self._build_extrinsic(
            lidar_cfg['transform'], cam_cfg['transform']
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assign_depths(
        self,
        detections: List[DetectedObject],
        points_raw: Optional[np.ndarray],
    ) -> List[DetectedObject]:
        """
        Populate distance_estimate on each DetectedObject.

        Args:
            detections:  list of DetectedObject (track_id already set).
            points_raw:  (N, 4) float32 array [x, y, z, intensity] in LiDAR frame,
                         or None if LiDAR data is not yet available.

        Returns:
            The same list with distance_estimate filled where possible.
        """
        if not detections or points_raw is None or len(points_raw) == 0:
            return detections

        projected = self._project_lidar_to_image(points_raw)
        if len(projected) == 0:
            return detections

        proj_arr = np.array(projected)          # (M, 3): [u, v, depth]
        u_all = proj_arr[:, 0]
        v_all = proj_arr[:, 1]
        d_all = proj_arr[:, 2]

        for obj in detections:
            x1, y1, x2, y2 = obj.bbox
            mask = (u_all >= x1) & (u_all < x2) & (v_all >= y1) & (v_all < y2)
            depths = d_all[mask]
            obj.distance_estimate = float(np.median(depths)) if len(depths) > 0 else None

        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_lidar_to_image(
        self, points_raw: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Transform LiDAR points to camera frame and project onto image plane.

        Returns a list of (u, v, depth_m) for points that fall within the
        image bounds and are at least 0.5 m in front of the camera.
        """
        if len(points_raw) == 0:
            return []

        # Homogeneous coordinates (N, 4)
        ones = np.ones((len(points_raw), 1), dtype=np.float64)
        pts_hom = np.hstack([points_raw[:, :3].astype(np.float64), ones])

        # Transform to camera frame: (N, 4)
        pts_cam = (self._T_lidar_to_cam @ pts_hom.T).T

        # Keep only points in front of camera
        mask = pts_cam[:, 2] > 0.5
        pts_cam = pts_cam[mask]
        if len(pts_cam) == 0:
            return []

        # Perspective projection
        p = self.K @ pts_cam[:, :3].T          # (3, N)
        u = p[0] / p[2]
        v = p[1] / p[2]
        depth = pts_cam[:, 2]

        # Keep only points within image bounds
        in_bounds = (u >= 0) & (u < self.img_w) & (v >= 0) & (v < self.img_h)
        return list(zip(
            u[in_bounds].tolist(),
            v[in_bounds].tolist(),
            depth[in_bounds].tolist(),
        ))

    def _build_extrinsic(self, lidar_tf: Dict, cam_tf: Dict) -> np.ndarray:
        """
        Compose LiDAR-to-camera extrinsic matrix.
        Both transforms are sensor-to-vehicle offsets (same reference frame),
        so: T_L2C = inv(T_cam) @ T_lidar
        """
        T_lidar = self._tf_to_matrix(lidar_tf)
        T_cam   = self._tf_to_matrix(cam_tf)
        return np.linalg.inv(T_cam) @ T_lidar

    @staticmethod
    def _tf_to_matrix(tf: Dict) -> np.ndarray:
        """
        Convert a CARLA-style transform dict to a 4×4 homogeneous matrix.
        Rotation convention: UE4 uses pitch=Y, yaw=Z, roll=X (ZYX Euler).
        """
        loc = tf['location']
        rot = tf['rotation']
        tx, ty, tz = loc['x'], loc['y'], loc['z']
        p = math.radians(rot.get('pitch', 0.0))
        y = math.radians(rot.get('yaw',   0.0))
        r = math.radians(rot.get('roll',  0.0))

        Rz = np.array([[ math.cos(y), -math.sin(y), 0],
                       [ math.sin(y),  math.cos(y), 0],
                       [           0,            0, 1]], dtype=np.float64)
        Ry = np.array([[ math.cos(p), 0, math.sin(p)],
                       [           0, 1,           0],
                       [-math.sin(p), 0, math.cos(p)]], dtype=np.float64)
        Rx = np.array([[1,           0,            0],
                       [0, math.cos(r), -math.sin(r)],
                       [0, math.sin(r),  math.cos(r)]], dtype=np.float64)
        R = Rz @ Ry @ Rx

        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3,  3] = [tx, ty, tz]
        return M