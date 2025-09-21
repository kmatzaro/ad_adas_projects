from dataclasses import dataclass
from typing import Optional, Dict
from collections import deque
import time
import numpy as np


@dataclass
class TimingMetrics:
    average_detection_time: float
    min_detection_time: float
    max_detection_time: float
    current_fps_est: int
    real_time_performance: Optional[bool]


class PerformanceMonitor:
    """
    Performance monitoring for CARLA pipeline
    
    In real-time systems, you must monitor performance to detect:
    - Processing bottlenecks
    - Frame drops
    - Memory leaks
    - System degradation over time
    """
    
    def __init__(self, config, window_size=1000):
        # Rolling window prevents memory growth and gives recent performance
        self.window_size = window_size
        self.FPS = config['carla']['FPS']

        self.metrics = {
            'front_camera': {
                'front_camera_lane_detection_time_ms': deque(maxlen=window_size),
                'front_camera_object_detection_time_ms': deque(maxlen=window_size),
                'total_front_camera_time_ms': deque(maxlen=window_size),
                'frame_count': 0
            },
            'rear_camera': {
                'rear_camera_object_detection_time_ms': deque(maxlen=window_size),
                'frame_count': 0
            },
            'left_camera': {
                'left_camera_object_detection_time_ms': deque(maxlen=window_size),
                'frame_count': 0
            },
            'right_camera': {
                'right_camera_object_detection_time_ms': deque(maxlen=window_size),
                'frame_count': 0
            }
        }
        
    def add_frame_data(self, camera_id, timing_metrics: Dict):
        """Add performance data for current frame"""
        if 'error' not in timing_metrics.keys():
            for key, val in timing_metrics.items():
                self.metrics[camera_id][key].append(val)
                self.metrics[camera_id]['frame_count'] += 1
        
        # Log every 60 frames = 1 second at 60fps, provides regular feedback
        if self.metrics[camera_id]['frame_count'] % 60 == 0:
            self._log_performance_summary(camera_id)
       
    def get_performance_stats(self, detection_times) -> TimingMetrics:
        """Statistics for performance tracking"""

        if not detection_times:
            return TimingMetrics(0, 0, 0, 0, False)
        
        detection_times_list = list(detection_times)

        # Average detection time
        avg_detection_time_ms = np.average(detection_times_list)

        # Min/max detection time
        min_detection_time = np.min(detection_times_list)
        max_detection_time = np.max(detection_times_list)

        # FPS estimate
        fps_estimate = 1 / avg_detection_time_ms * 1000

        # Real time performance
        if avg_detection_time_ms <= 1/self.FPS * 1000:
            real_time_performance = True
        else:
            real_time_performance = False

        return TimingMetrics(
            average_detection_time = avg_detection_time_ms,
            min_detection_time = min_detection_time,
            max_detection_time = max_detection_time,
            current_fps_est = int(fps_estimate),
            real_time_performance = real_time_performance
        )

    def _log_performance_summary(self, camera_id: str):
        """Log function to print performance stats"""

        print("=" * 50)
        print(f"Enhanced Perception Performance over ({self.window_size} frames) for {camera_id}:")

        for name, timing_data in self.metrics[camera_id].items():
            if timing_data and name != 'frame_count':
                performance_stats = self.get_performance_stats(timing_data)
                print(f"  {name:18s}: avg={performance_stats.average_detection_time:.1f}ms  ({performance_stats.min_detection_time:.1f}-{performance_stats.max_detection_time:.1f}ms)")
                
        # Real-time status
        real_time_budget = 1000 / self.FPS
        if camera_id == 'front_camera':
            avg_perception = np.mean(self.metrics[camera_id][f"total_{camera_id}_time_ms"])
        else:
            avg_perception = np.mean(self.metrics[camera_id][f"{camera_id}_object_detection_time_ms"])
        status = "GOOD" if avg_perception < real_time_budget else " SLOW"
        print(f"  Real-time Status: {status} (target: <{real_time_budget:.1f}ms)")