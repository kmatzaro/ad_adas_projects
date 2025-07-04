import numpy as np
import cv2
import csv
import os

class LaneValidator:
    def __init__(self, world, camera_actor, vehicle, lane_detector, output_dir="validation"):
        """
        Paramters:
        -----------
        world          : carla.World object
        camera_actor   : carla.Actor (the RGB camera)
        lane_detector  : your SimpleLaneDetector instance
        output_dir     : where to save logs and overlay frames
        """
        self.world         = world
        self.camera        = camera_actor
        self.vehicle       = vehicle
        self.lane_detector = lane_detector
        self.map           = world.get_map()
        self.output_dir    = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Build camera intrinsics once
        img_w, img_h = lane_detector.img_size
        fov = float(self.camera.attributes['fov'])
        self.K = np.array([
            [ img_w/(2*np.tan(np.deg2rad(fov)/2)), 0, img_w/2 ],
            [ 0, img_w/(2*np.tan(np.deg2rad(fov)/2)), img_h/2 ],
            [ 0, 0, 1 ]
        ])

    def sample_ground_truth(self, vehicle, dist=1.0, num_pts=50):
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
        if cam_coords[2] <= 0.01:
            return None
        uvw = self.K @ cam_coords[:3]
        u, v = int(uvw[0]/uvw[2]), int(uvw[1]/uvw[2])
        return (u, v)

    def compute_metrics(self, det_midpoints, gt_pixels, threshold_px=10):
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
            f"pct_within_{threshold_px}px": float(np.mean(errors < threshold_px)) * 100
        }

    def validate_frame(self, frame_rgb, det_left, det_right, vehicle, frame_id):
        """Run validation on a single frame:
           - sample GT, project, detect midpoints, compute, visualize, and log.
        """
        # 1) sample & project GT
        gt3d = self.sample_ground_truth(vehicle)
        gt2d = [p for p in (self.project_to_image(pt) for pt in gt3d) if p]

        # 2) build detected mid-lane pts along image rows
        h = self.lane_detector.img_size[1]
        ploty = np.linspace(int(h*0.6), h-1, num=30)
        left_fit = None if det_left is None else np.polyfit([det_left[1], det_left[3]], [det_left[0], det_left[2]], 1)
        right_fit= None if det_right is None else np.polyfit([det_right[1],det_right[3]],[det_right[0],det_right[2]],1)
        det_mid = []
        for y in ploty:
            if left_fit is None or right_fit is None:
                break
            xl = np.polyval(left_fit, y)
            xr = np.polyval(right_fit, y)
            det_mid.append((int((xl+xr)/2), int(y)))

        # 3) compute metrics
        metrics = self.compute_metrics(det_mid, gt2d)

        # 4) visualize overlay
        vis = frame_rgb.copy()
        # draw GT in yellow
        for u,v in gt2d:
            cv2.circle(vis, (u,v), 3, (0,255,255), -1)
        # draw midpoints in red
        for u,v in det_mid:
            cv2.circle(vis, (u,v), 3, (0,0,255), -1)

        # save visualization
        cv2.imwrite(os.path.join(self.output_dir, f"frame_{frame_id:04d}.png"), vis)

        # return metrics for logging
        metrics["frame"] = frame_id
        return metrics

    def run_validation(self, sim_time, current_frame, frame_id, capture_times, logs, num_frames=10, time_intervals=10.0, log_csv="metrics.csv"):
        """Validate each, and write CSV."""
        if current_frame and sim_time >= capture_times and frame_id <= num_frames:
            frame_rgb = current_frame["result"]
            left, right = self.lane_detector.prev_left_coords, self.lane_detector.prev_right_coords
            m = self.validate_frame(frame_rgb, left, right, self.vehicle, frame_id)
            logs.append(m)

            frame_id += 1
            capture_times += time_intervals

            # write CSV
            keys = list(m.keys())
            with open(os.path.join(self.output_dir, log_csv), "w", newline="") as f:
                writer = csv.DictWriter(f, keys)
                writer.writeheader()
                writer.writerows(logs)
            
            if frame_id == num_frames:
                print(f"Validation complete: results in {self.output_dir}/{log_csv}")
        return frame_id, capture_times, logs