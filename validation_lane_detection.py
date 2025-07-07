import numpy as np
import cv2
import csv
import os
import shutil

class LaneValidator:
    def __init__(self, config, world, camera_actor, vehicle, lane_detector):
        """
        Paramters:
            world          : carla.World object
            camera_actor   : carla.Actor (the RGB camera)
            lane_detector  : a LaneDetector instance
            output_dir     : where to save logs and overlay frames
        """
        self.config        = config['validation']
        self.world         = world
        self.camera        = camera_actor
        self.vehicle       = vehicle
        self.lane_detector = lane_detector
        self.map           = world.get_map()
        self.output_dir    = self.config['output_dir']

        # Cleanup any existing directory
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
            print(f"Cleaned up and prepared output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        # Build camera intrinsics once
        self.img_w = lane_detector.img_size['image_width'] 
        self.img_h = lane_detector.img_size['image_height']
        print('Passed')
        fov = float(self.camera.attributes['fov'])
        focal = self.img_w / (2.0 * np.tan(fov * np.pi / 360.0))
        self.K = np.array([
            [ focal, 0    , self.img_w/2 ],
            [ 0    , focal, self.img_h/2 ],
            [ 0    , 0    , 1       ]
        ])

    def sample_ground_truth(self, vehicle, dist=0.1, num_pts=100):
        """Sample `num_pts` waypoints spaced `dist` meters along the ego lane."""
        ego_loc = vehicle.get_transform().location
        wp = self.map.get_waypoint(ego_loc)
        pts = []
        for _ in range(num_pts):
            pts.append(wp.transform.location)
            next_wps = wp.next(dist)
            if not next_wps:
                break
            wp = next_wps[0]
        return pts

    def project_to_image(self, location):
        """Project a carla.Location to 2D pixel coords, or None if behind camera."""
        # world -> camera
        X = np.array([location.x, location.y, location.z, 1.0])
        M = np.array(self.camera.get_transform().get_inverse_matrix())
        cam_coords = M @ X
        cam_coords = [cam_coords[1], -cam_coords[2], cam_coords[0]]
        uvw = self.K @ cam_coords
        u, v = int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2]) # Normalize
        return (u, v) # (u,v) are 2D pixel coords

    def compute_metrics(self, det_midpoints, gt_pixels,):
        """Given lists of (u,v) detected and ground-truth points, compute errors."""
        errors = []
        for u_gt, v_gt in gt_pixels:
            # match by closest row
            diffs = [abs(v_gt - v_det) for _, v_det in det_midpoints]
            idx = np.argmin(diffs)
            u_det, v_det = det_midpoints[idx]
            errors.append(abs(u_gt - u_det))
        errors = np.array(errors)
        return {
            "mean_error": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            f"pct_within_{self.config['threshold_px']}px": float(np.mean(errors < self.config['threshold_px'])) * 100
        }

    def validate_frame(self, image, left_coords, right_coords, vehicle, frame_id, num_points_interpolate = 30):
        """Run validation on a single frame:
           - sample GT, project, detect midpoints, compute, visualize, and log.
        """
        # 1) sample & project GT up to 60% from the bottom to match prediction
        gt3d = self.sample_ground_truth(vehicle)
        raw_gt2d = [p for p in (self.project_to_image(pt) for pt in gt3d) if p]

        y_min = int(self.config['y_min_pct'] * self.img_h)
        gt2d = [(u, v) for (u, v) in raw_gt2d if v >= y_min]

         # 2) build detected mid‐lane pts directly from coords
        det_mid = []
        if left_coords is not None and right_coords is not None:
            x1_l, y1, x2_l, y2 = left_coords
            x1_r, _,  x2_r, _  = right_coords

            # pick N points interpolated between y1→y2
            ploty = np.linspace(y1, y2, num=num_points_interpolate).astype(int)
            for y in ploty:
                # linear interpolation on each line:
                t = (y - y1) / (y2 - y1)
                xl = int(x1_l + t * (x2_l - x1_l))
                xr = int(x1_r + t * (x2_r - x1_r))
                det_mid.append(((xl + xr)//2, y))
        
        # 3) compute metrics only if you have a mid‐lane
        if det_mid:
            metrics = self.compute_metrics(det_mid, gt2d)
        else:
            # log NaNs or zeros, or simply skip
            metrics = {
            "mean_error": None,
            "rmse":       None,
            "pct_within_10px": None
            }
        # return metrics for logging
        metrics["frame"] = frame_id
        metrics["len_det_mid"] = len(det_mid)
        return metrics, gt2d, det_mid

    def draw_detected_vs_gt(self, image, gt2d, det_mid, frame_id, save_frame=True):
        # visualize overlay
        # draw GT in yellow
        for u,v in gt2d: cv2.circle(image, (u,v), 3, (0,255,255), -1)
        # draw midpoints in red
        if len(det_mid) > 0:
            for u,v in det_mid: cv2.circle(image, (u,v), 3, (0,0,255), -1)
            if save_frame:
            # save visualization
                cv2.imwrite(os.path.join(self.output_dir, f"frame_{frame_id:04d}.png"), image)

        

    def run_validation(self, sim_time, image, frame_id, capture_times, logs, left_coords, right_coords):
        """Validate each, and write CSV."""
        metrics, gt2d, det_mid = self.validate_frame(image, left_coords, right_coords, self.vehicle, frame_id)

        if self.config['draw_det_vs_gt'] and det_mid:
            self.draw_detected_vs_gt(image, gt2d, det_mid, frame_id, save_frame=False)

        if sim_time >= capture_times and frame_id < self.config['num_captures']:
            self.draw_detected_vs_gt(image, gt2d, det_mid, frame_id, save_frame=True)
            logs.append(metrics)

            frame_id += 1
            capture_times += self.config['interval_seconds']
            
            # write CSV
            keys = list(metrics.keys())
            with open(os.path.join(self.output_dir, self.config['log_csv']), "w", newline="") as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(logs)
            
            if frame_id == self.config['num_captures']:
                print(f"Validation complete: results in {self.output_dir}/{self.config['log_csv']}")
        return frame_id, capture_times, logs