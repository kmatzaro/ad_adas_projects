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
        self.callback_times = deque(maxlen=window_size)
        self.lane_detection_times = deque(maxlen=window_size)
        self.object_detection_times = deque(maxlen=window_size)
        self.total_perception_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.FPS = config['carla']['FPS']
        self.start_time = time.time()
        
    def add_frame_data(self, callback_time, perception_time: Dict):
        """Add performance data for current frame"""
        self.callback_times.append(callback_time)
        self.lane_detection_times.append(perception_time['lane_detection_time_ms'])
        self.object_detection_times.append(perception_time['object_detection_time_ms'])
        self.total_perception_times.append(perception_time['total_end_time'])
        self.frame_count += 1
        
        # Log every 60 frames = 1 second at 60fps, provides regular feedback
        if self.frame_count % 60 == 0:
            self._log_performance_summary()
       
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

    def _log_performance_summary(self):
        """Log function to print performance stats"""

        timing_components = {
        "Lane Detection": self.lane_detection_times,
        "Object Detection": self.object_detection_times, 
        "Total Perception": self.total_perception_times,
        "Callback Overhead": self.callback_times
        }

        print("=" * 50)
        print(f"Enhanced Perception Performance ({len(self.total_perception_times)} frames):")

        for name, timing_data in timing_components.items():
            if timing_data:
                performance_stats = self.get_performance_stats(timing_data)
                print(f"  {name:18s}: avg={performance_stats.average_detection_time:.1f}ms  ({performance_stats.min_detection_time:.1f}-{performance_stats.max_detection_time:.1f}ms)")
        
        # Real-time status
        if self.total_perception_times:
            real_time_budget = 1000 / self.FPS
            avg_perception = np.mean(self.total_perception_times)
            status = "GOOD" if avg_perception < real_time_budget else " SLOW"
            print(f"  Real-time Status: {status} (target: <{real_time_budget:.1f}ms)")